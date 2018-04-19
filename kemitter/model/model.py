import sys
from abc import ABC, abstractmethod
import numpy as np


class Model(ABC):
    """Abstract base class for all solver models.

    This class defines the common interface used by each individual model. Individual model classes
    differ in how they implement their ``run()`` method, but users interface with them using the methods
    and attributes defined here.

    Notes:
        ``Model`` should not be instantiated on its own. Only its children, which implement a ``run()`` method
        should be used directly.

    Attributes:
        solver_result (ndarray): 1D array containing the solved variable values returned by the solver in its raw form.
        background (float): the solved constant background term.
        rates (dict): contains 1D arrays with the solved wavelength-dependent emission rates for each basis type.
        total_emission (ndarray): 1D array containing the solved total emission rates, i.e. the sum of each
            value in ``Model.rates``.
        percent_emission (dict): contains 1D arrays corresponding to the percent contribution of each basis type to
            the total emission at each wavelength.
        counts (dict): contains 1D arrays representing the solved wavelength-dependent total counts for each basis type.
        basis_names (list of str): names of the basis types, denoting the keys to access specific rates, counts,
            and percent_emission vectors.
    """
    def __init__(self):
        self.__pol_children = None  # list of PolDataSets
        self.solver_result = None
        self.background = None
        self.rates = None
        self.total_emission = None
        self.percent_emission = None
        self.counts = None
        self.basis_names = None

    @property
    def is_empty(self):
        """bool: Whether or not the model contains any polarized data to process."""
        return not self.__pol_children

    @property
    def polarization_angles(self):
        """list of int or float: the various polarization angles contained in the model data sets."""
        if not self.is_empty:
            angles = []
            for child in self.__pol_children:
                angles.append(child.pol_angle)
            return angles

    @property
    def bases(self):
        """list of Basis: the various bases contained in the model data sets in their full object format."""
        if not self.is_empty:
            return [self.data_set(angle).basis for angle in self.polarization_angles]

    @property
    def observations(self):
        """list of Observation: the various observations contained in the model data sets in their full object format."""
        if not self.is_empty:
            return [self.data_set(angle).observation for angle in self.polarization_angles]

    @property
    def basis_matrices(self):
        """list of csc_matrix: list of the sparse basis matrices stored in the ``Basis`` objects of
        the model data sets."""
        if not self.is_empty:
            return [self.data_set(angle).basis.basis_matrix for angle in self.polarization_angles]

    @property
    def n_polarizations(self):
        """int: the number of polarized data sets contained in the model."""
        return len(self.__pol_children)

    def data_set(self, pol_angle):
        """Getter function for a particular polarized data set.

        Args:
            pol_angle (int or float): the polarization angle of the desired polarized data set, in degrees.

        Returns (PolDataSet):
            The polarized data set object, containing references to the corresponding observations and
            bases, and if solved the polarized image fits and counts.
        """
        for child in self.__pol_children:
            if child.pol_angle == pol_angle:
                return child
        raise ValueError('No child with polarization angle {0}'.format(pol_angle))

    def remove_data_set(self, pol_angle):
        """Removes a polarized data set form the model.

        Args:
            pol_angle (int or float): the polarization angle of the polarized data set to remove, in degrees.
        """
        if pol_angle in self.polarization_angles:
            child = self.data_set(pol_angle)
            self.__pol_children.remove(child)

    def reset(self):
        """Clears all polarized data sets from the model, after confirmation.

        Warnings:
            This operation will clear any built bases or solver results contained in the model.
        """
        yes = ['y', 'yes']
        no = ['n', 'no', '']
        sys.stdout.write('WARNING: This action will clear all loaded data, calculated bases, and models.\n'
                         '    Would you still like to proceed? [y/N] ')
        choice = input().lower()
        if choice in yes:
            self.__pol_children = None
        elif choice in no:
            return
        else:
            print('Invalid answer. Exiting without resetting...\n')
            return

    @abstractmethod
    def run(self, bases, observations):
        """Abstract method for all child ``run()`` methods

        This method is called by each individual solver routine to handle data pre-processing and basis building.

        Processing proceeds by first loading bases and observations into proper polarized data sets. In this step,
        arguments are checked to ensure polarization angles match in value and order. Any bases that have not been
        built already are built with their corresponding ``build()`` method. Then processing is returned
        to the individual ``run()`` implementation of the calling solver.

        Args:
            bases (list of Basis): The basis objects (built or not), to be used for fitting.
            observations (list of Observations): The observation objects to be used for fitting.
        """
        self._load_into_pol_data_sets(bases, observations)

        for b in bases:
            if not b.is_built:
                b.build()
        print('\n============ Starting the kemitter ' + self.name + ' solver ============')

    def build_bases(self):
        for angle in self.polarization_angles:
            active_basis = self.data_set(angle).basis
            if not active_basis.is_built:
                active_basis.build()

    def visualize(self):
        pass

    def _load_into_pol_data_sets(self, bases, observations):
        # check to ensure the same number of bases and observations have been provided.
        if not isinstance(bases, list):
            bases = [bases]
        if not isinstance(observations, list):
            observations = [observations]
        if len(bases) != len(observations):
            raise ValueError('Number of bases and observations do not match')

        # extract the polarization angles from the bases and ensure their order and values match in the observations
        pol_angles = []
        for i in range(len(bases)):
            pol_angles.append(bases[i].pol_angle)

        for i in range(len(pol_angles)):
            if observations[i].pol_angle != pol_angles[i]:
                raise ValueError('Polarization angles in basis and observation do not match in position {0}.'.format(i))

        # check that bases are of the same type
        for i in range(len(pol_angles) - 1):
            if bases[i].basis_names != bases[i+1].basis_names:
                raise ValueError('Basis names ' + str(bases[i].basis_names) + ' and ' + str(bases[i+1].basis_names) +
                                 ' do not match.')
        self.basis_names = bases[0].basis_names

        # load information into polarized data sets (PolDataSet objects), and add to list of children data sets.
        # At this step, also define the observation specific parameters for
        # the basis (wavelength and momentum grid size information)
        self.__pol_children = []
        for i in range(len(pol_angles)):
            self.__pol_children.append(PolDataSet(pol_angles[i], observations[i], bases[i]))
            bases[i].define_observation_parameters(observations[i].wavelength, observations[i].momentum_pixel_count)

    def _process_result(self, result_val):
        w_count = self.bases[0].basis_parameters.wavelength_count
        orig_w_count = self.bases[0].basis_parameters.orig_wavelength_count
        ux_count = self.bases[0].basis_parameters.ux_count
        # find proper x_indices accounting for padded wavelength
        self.rates = {}
        self.counts = {}
        self.percent_emission = {}
        self.total_emission = np.zeros((orig_w_count, 1))
        for i, name in enumerate(self.basis_names):
            begin_ind = i * w_count
            end_ind = begin_ind + w_count - 1
            if w_count != orig_w_count:
                begin_ind += int(np.floor((ux_count - 1) / 2))
                end_ind -= int(np.floor(ux_count / 2))
            self.rates[name] = result_val[begin_ind:(end_ind+1)]
            self.total_emission += self.rates[name]
            self.counts[name] = np.zeros_like(self.rates[name])
            for i in range(len(self.bases)):
                self.counts[name] += np.array(self.bases[i].basis_matrix[:,begin_ind:end_ind+1].multiply(self.rates[name].T).sum(axis=0)).T

        for name in self.basis_names:
            self.percent_emission[name] = self.rates[name] / self.total_emission

        for angle in self.polarization_angles:
            self.data_set(angle).fit = self.data_set(angle).basis.basis_matrix @ result_val
            self.data_set(angle).counts = np.zeros((orig_w_count,1))
            for name in self.basis_names:
                self.data_set(angle).counts += self.counts[name]


class PolDataSet(object):
    """Utility class for storing and organizing polarized data (bases, observation, fits).

    Used to keep data of the same polarization angle packaged together for batch processing and access.

    Warnings:
        Polarized data sets should only be accessed and modified through the parent model's ``data_set()`` method.

    Attributes:
        pol_angle (int or float): the polarization angle shared by all of the various data set components.
        observation (Observation): the experimental observation object
        basis (Basis): the theoretically basis object
        fit (ndarray): reconstructed fit to observation data once solved by the model solver
        counts (ndarray): the wavelength-dependent counts at this particular polarization angle, across all basis types.
    """
    def __init__(self, pol_angle, obs, basis):
        self.pol_angle = pol_angle
        self.observation = obs
        self.basis = basis
        self.fit = None
        self.counts = None
