import numpy as np
from .fields import fields
from .builders import EDIsoBuilder

class BasisFactory:

    @staticmethod
    def make_builder(parameters):
        if parameters.basis_type == "EDIso":
            return EDIsoBuilder(parameters)
        else:
            print("Bad builder type: " + parameters.basis_type)
            pass


class BasisParameters:
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
    :type wavelength: numpy.ndarray
    """

    def __init__(self, basis_type,
                 n0, n1, n2o, n2e, n3,
                 ux_range, uy_range,
                 ux_count, uy_count,
                 d, s, l,
                 wavelength,
                 wavelength_count,
                 pol_angle):
        self.basis_type = basis_type
        self.n0         = n0
        self.n1         = n1
        self.n2o        = n2o
        self.n2e        = n2e
        self.n3         = n3
        self.ux_range   = ux_range
        self.uy_range   = uy_range
        self.ux_count   = ux_count
        self.uy_count   = uy_count
        self.d          = d
        self.s          = s
        self.l          = l
        self.wavelength = wavelength
        self.wavelength_count = wavelength_count
        self.pol_angle = pol_angle
