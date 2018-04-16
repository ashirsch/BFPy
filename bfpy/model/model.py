import sys
from abc import ABC, abstractmethod
import numpy as np


class Model(ABC):

    def __init__(self):
        self.__pol_children = None  # list of PolDataSets
        self.solver_result = None
        self.background = None
        self.rates = None
        self.total_emission = None
        self.percent_emission = None
        self.counts = None
        self.fits = None
        self.basis_names = None

    @property
    def polarization_angles(self):
        angles = []
        for child in self.__pol_children:
            angles.append(child.pol_angle)
        return angles

    @property
    def bases(self):
        if not self.is_empty:
            return [self.data_set(angle).basis for angle in self.polarization_angles]

    @property
    def observations(self):
        if not self.is_empty:
            return [self.data_set(angle).observation for angle in self.polarization_angles]

    @property
    def basis_matrices(self):
        if not self.is_empty:
            return [self.data_set(angle).basis.basis_matrix for angle in self.polarization_angles]

    @property
    def n_polarizations(self):
        return len(self.__pol_children)

    def data_set(self, pol_angle):
        for child in self.__pol_children:
            if child.pol_angle == pol_angle:
                return child
        raise ValueError('No child with polarization angle {0}'.format(pol_angle))

    def remove_data_set(self, pol_angle):
        child = self.data_set(pol_angle)
        self.__pol_children.remove(child)

    @property
    def is_empty(self):
        return not self.__pol_children

    def is_defined(self):
        pass

    def reset(self):
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
        self._load_into_pol_data_sets(bases, observations)

        for b in bases:
            if not b.is_built:
                b.build()

    def build_bases(self):
        for angle in self.polarization_angles:
            active_basis = self.data_set(angle).basis
            if not active_basis.is_built:
                active_basis.build()

    def visualize(self):
        pass

    def _load_into_pol_data_sets(self, bases, observations):
        if not isinstance(bases, list):
            bases = [bases]
        if not isinstance(observations, list):
            observations = [observations]
        if len(bases) != len(observations):
            raise ValueError('Number of bases and observations do not match')

        # pull out polarization angles
        pol_angles = []
        for i in range(len(bases)):
            pol_angles.append(bases[i].pol_angle)

        # load data into PolDataSets
        for i in range(len(pol_angles)):
            if observations[i].pol_angle != pol_angles[i]:
                raise ValueError('Polarization angles in basis and observation do not match in position {0}.'.format(i))

        for i in range(len(pol_angles) - 1):
            if bases[i].basis_names != bases[i+1].basis_names:
                raise ValueError('Basis names ' + str(bases[i].basis_names) + ' and ' + str(bases[i+1].basis_names) +
                                 ' do not match.')
        self.basis_names = bases[0].basis_names

        self.__pol_children = []
        for i in range(len(pol_angles)):
            self.__pol_children.append(PolDataSet(pol_angles[i], observations[i], bases[i]))
            bases[i].define_observation_parameters(observations[i].wavelength, observations[i].angular_pixel_count)

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

    def __init__(self, pol_angle, obs, basis):
        self.pol_angle = pol_angle
        self.observation = obs
        self.basis = basis
        self.fit = None
        self.counts = None
