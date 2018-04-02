#!/usr/bin/env python
import sys

from bfpy.obsrv import observation
from bfpy.bases import basis


class BFPSession(object):

    def __init__(self, verbose=True, pol_children=None):
        if pol_children is None:
            pol_children = []
        self.__pol_children = pol_children  # PolDataSet
        self.model = None                   # cvxpy Model or subclass of such
        self.vis = None                     # PlotSet
        self.verbose = verbose

    @property
    def polarization_angles(self):
        angles = []
        for child in self.__pol_children:
            angles.append(child.pol_angle)
        return angles

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
            self.__pol_children.clear()
            self.model = None
            self.basis_parameters = None
            self.vis = None
        elif choice in no:
            return
        else:
            print('Invalid answer. Exiting without resetting...')
            return

    def load(self, pol_angle):
        # check if polarization has already been loaded into session
        if pol_angle in self.polarization_angles:
            yes = ['y', 'yes']
            no = ['n', 'no', '']
            sys.stdout.write('WARNING: An Observation with polarization angle {0} is already defined.\n'
                             '    Would you like to overwrite this observation? [y/N] '.format(pol_angle))
            choice = input().lower()
            if choice in yes:
                self.remove_data_set(pol_angle)
            elif choice in no:
                return
            else:
                print('Invalid answer. Exiting without overwriting...')
                return
        # create new polarized data set
        active_data_set = PolDataSet(pol_angle)
        # launches interactive loading
        success = active_data_set.observation._load()
        if success:
            print('Successfully loaded {0} deg. data.'.format(pol_angle))
            self.__pol_children.append(active_data_set)
        else:
            print('Unable to load observation.')

    def load_from_file(self, pol_angle, filename):
        # make and append PolDataSet and load
        pass

    def load_from_array(self, pol_angle, numpy_data):
        # make and append PolDataSet and load
        pass

    def define_fit(self, model, **kwargs):
        # sets model
        pass

    def run(self, open_slit, pad_lambda, fit_all_frames, debug=True):
        pass

    def build_bases(self):
        for angle in self.polarization_angles:
            if self.data_set(angle).basis.is_defined:
                self.data_set(angle).basis.build()

    def visualize(self):
        pass


class PolDataSet(object):

    def __init__(self, pol_angle=None):
        self.observation = observation.Observation(pol_angle)
        self.basis = basis.Basis(pol_angle)
        self.pol_angle = pol_angle

    def define_basis(self, basis_type, n0, n1, n2o, n2e, n3,
                     ux_range, uy_range, ux_count, uy_count,
                     d, s, l, pad_w=False, trim_w=True):
        """
        :type basis_type: str
        :type n0: float
        :type n1: float
        :type n2o: float
        :type n2e: float
        :type n3: float
        :type ux_range: tuple
        :type uy_range: tuple
        :type ux_count: int
        :type uy_count: int
        :type d: float
        :type s: float
        :type l: float
        :type pad_w: bool
        :type trim_w: bool
        """
        self.basis.basis_parameters = basis.BasisParameters(basis_type,
                                                            n0, n1, n2o, n2e, n3,
                                                            ux_range, uy_range,
                                                            ux_count, uy_count,
                                                            d, s, l,
                                                            self.pol_angle)
        if self.observation.loaded:
            self.basis.basis_parameters.set_wavelength(self.observation.wavelength)
