#!/usr/bin/env python
from .obsrv import observation
from .basis import basis


class BFPSession(object):

    def __init__(self, pol_children=None):
        if pol_children is None:
            pol_children = []
        self.__pol_children = pol_children  # PolDataSet
        self.model = None                   # cvxpy Model or subclass of such
        self.basis_parameters = None        # BasisParameters class or subclass of such
        self.vis = None                     # PlotSet

    def is_empty(self):
        pass

    def reset(self):
        pass

    def is_defined(self):
        pass

    def n_polarizations(self):
        return len(self.__pol_children)

    def load(self, pol_angle):
        # launches interactive loading
        pass

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
