import time
import numpy as np

from bfpy.basis.builders.ediso import EDIsoBuilder


class Basis(object):
    def __init__(self, pol_angle):
        self.basis_matrix = None
        self.built = False
        self.basis_parameters = None
        self.__pol_angle = pol_angle

    def build(self):
        builder = self._make_builder()

        t0 = time.time()
        self.basis_matrix = builder.build()
        t1 = time.time()

        self.built = True
        print(self.basis_parameters.basis_type + " basis successfully built.")
        print("Elapsed build time: {0:.3f}".format(t1-t0))

    def _make_builder(self):
        if self.basis_parameters.basis_type == "EDIso":
            return EDIsoBuilder(self.basis_parameters)
        else:
            print("Invalid builder type: " + self.basis_parameters.basis_type)
            pass

    @property
    def is_defined(self):
        defined = True
        defined = defined and self.basis_parameters._verify_state()
        defined = defined and self.__pol_angle is not None
        return defined


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
                 pol_angle,
                 wavelength=None,
                 wavelength_count=None,
                 pad_w=False,
                 trim_w=True):
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
        self.pol_angle  = np.radians(pol_angle)
        self.pad_w      = pad_w
        self.trim_w     = trim_w
        self.orig_wavelength = wavelength
        self.orig_wavelength_count = wavelength_count

        if self.pad_w:
            self._pad_wavelength()

    def _pad_wavelength(self):
        pre_wavelength_spacing = np.abs(self.wavelength[1] - self.wavelength[0])
        pre_padding = self.wavelength[0] + pre_wavelength_spacing * np.arange(-np.floor((self.ux_count-1)/2), 0)

        post_wavelength_spacing = np.abs(self.wavelength[-1] - self.wavelength[-2])
        post_padding = self.wavelength[-1] + post_wavelength_spacing * np.arange(1, np.floor(self.ux_count/2) + 1)

        self.wavelength = np.hstack((pre_padding, self.wavelength, post_padding))
        self.wavelength_count = len(self.wavelength)

    def _set_wavelength(self, wavelength, pad_w=False, trim_w=True):
        self.wavelength = wavelength
        self.wavelength_count = len(wavelength)
        self.orig_wavelength = wavelength
        self.orig_wavelength_count = len(wavelength)
        self.pad_w = pad_w
        self.trim_w = trim_w

        if self.pad_w:
            self._pad_wavelength()

    def _verify_state(self):
        assert self.basis_type
        assert self.n0 is not None and self.n0 > 0
        assert self.n1 is not None and self.n1 > 0
        assert self.n2o is not None and self.n2o > 0
        assert self.n2e is not None and self.n2e> 0
        assert self.n3 is not None and self.n3 > 0
        assert self.ux_range and self.ux_range[0] <= self.ux_range[1]
        assert self.uy_range and self.uy_range[0] <= self.uy_range[1]
        assert self.ux_count is not None and self.ux_count > 0
        assert self.uy_count is not None and self.uy_count > 0
        assert self.d is not None and self.d >= 0
        assert self.s is not None and self.s >= 0
        assert self.l is not None and self.l >= 0
        assert self.wavelength is not None
        assert self.wavelength_count == len(self.wavelength)
        assert self.pol_angle is not None
        assert self.pad_w is not None
        assert self.trim_w is not None
        assert self.orig_wavelength is not None
        assert self.orig_wavelength_count == len(self.orig_wavelength)
        return True
