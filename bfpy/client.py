#!/usr/bin/env python
from .obsrv import observation
from .basis import basis


class BFPSession(object):

    def __init__(self, pol_children=None):
        if pol_children is None:
            pol_children = []
        self.__pol_children = pol_children  # PolDataSet
        self.model = None                   # mosek.Model() or subclass of such
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

    def load_interactive(self):
        # launches interactive loading
        pass

    def load_from_file(self, pol_type, filename):
        # make and append PolDataSet and load
        pass

    def load_from_array(self, pol_type, numpy_data):
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

    def __init__(self, pol=None):
        self.observation = observation.Observation()
        self.basis = basis.Basis()
        self.pol_type = pol
