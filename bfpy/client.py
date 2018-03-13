#!/usr/bin/env python
import sys

from bfpy.obsrv import observation
from bfpy.basis import basis


class BFPSession(object):

    def __init__(self, verbose=True, pol_children=None):
        if pol_children is None:
            pol_children = []
        self.__pol_children = pol_children  # PolDataSet
        self.model = None                   # cvxpy Model or subclass of such
        self.vis = None                     # PlotSet
        self.verbose = verbose

    def is_empty(self):
        return not self.__pol_children

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

    def is_defined(self):
        pass

    def n_polarizations(self):
        return len(self.__pol_children)

    def load(self, pol_angle):
        # check if polarization has already been loaded into session
        if not self.is_empty():
            for child in self.__pol_children:
                if pol_angle == child.pol_angle:
                    yes = ['y', 'yes']
                    no = ['n', 'no', '']
                    sys.stdout.write('WARNING: An Observation with polarization angle {0} is already defined.\n'
                                     '    Would you like to overwrite this observation? [y/N] '.format(pol_angle))
                    choice = input().lower()
                    if choice in yes:
                        self.__pol_children.remove(child)
                        continue
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

    def define_basis(self, basis_type, n0, n1, n2, n3, **kwargs):
        # sets basis parameters
        pass

    def define_fit(self, model, **kwargs):
        # sets model
        pass

    def run(self, open_slit, pad_lambda, fit_all_frames, debug=True):
        pass

    def visualize(self):
        pass


class PolDataSet(object):

    def __init__(self, pol_angle=None):
        self.observation = observation.Observation(pol_angle)
        self.basis = basis.Basis()
        self.pol_angle = pol_angle
