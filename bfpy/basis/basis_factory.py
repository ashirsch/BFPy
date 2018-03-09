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
    :type wavelength_count: int
    :type pol_angle: float
    :type pad_w: bool
    """

    def __init__(self, basis_type,
                 n0, n1, n2o, n2e, n3,
                 ux_range, uy_range,
                 ux_count, uy_count,
                 d, s, l,
                 wavelength,
                 wavelength_count,
                 pol_angle,
                 pad_w=False):
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
        self.pad_w = pad_w

        if self.pad_w:
            self._pad_wavelength()

    def _pad_wavelength(self):
        pre_wavelength_spacing = np.abs(self.wavelength[1] - self.wavelength[0])
        pre_padding = self.wavelength[0] + pre_wavelength_spacing * np.arange(-np.floor((self.ux_count-1)/2), 0)

        post_wavelength_spacing = np.abs(self.wavelength[-1] - self.wavelength[-2])
        post_padding = self.wavelength[-1] + post_wavelength_spacing * np.arange(1, np.floor(self.ux_count/2) + 1)

        self.wavelength = np.hstack((pre_padding, self.wavelength, post_padding))
        self.wavelength_count = len(self.wavelength)
