import time
from abc import ABC, abstractmethod
import numpy as np
import scipy.sparse as sp


class Basis(ABC):
    """Abstract base class for bases"""
    def __init__(self):
        self.basis_names = None
        self.basis_matrix = None
        self.is_built = False
        self.basis_parameters = None
        self.pol_angle = None
        super().__init__()

    @abstractmethod
    def build(self):
        pass

    def define_observation_parameters(self, wavelength, k_count, open_slit=True):
        if self.basis_parameters is not None:
            self.basis_parameters.uy_count = k_count
            self.basis_parameters.ux_count = k_count if open_slit else 1
            self.basis_parameters.set_wavelength(wavelength)
        else:
            print("Geometric and optical parameters must be defined prior to loading of "
                  "observation-dependent parameters.")

    @property
    def is_defined(self):
        defined = True
        defined = defined and self.basis_parameters._verify_state()
        defined = defined and self.pol_angle is not None
        return defined

    def sparse_column_major_offset(self, matrix):
        # flatten data matrix by column
        flattened_length = self.basis_parameters.ux_count * self.basis_parameters.uy_count * \
                          self.basis_parameters.wavelength_count
        flat_matrix = np.reshape(matrix, (flattened_length,), order='F')
        # load into sparse matrix
        return sp.csc_matrix((flat_matrix, (self.sparse_rows(), self.sparse_cols())))

    def basis_trim(self, matrix):
        begin_ind = int(self.basis_parameters.uy_count * np.floor(self.basis_parameters.ux_count/2))
        if self.basis_parameters.pad_w:
            begin_ind += int(self.basis_parameters.uy_count * np.floor((self.basis_parameters.ux_count - 1)/2))

        final_pix_row_ind = matrix.shape[0] - 1
        end_ind = int(final_pix_row_ind - self.basis_parameters.uy_count * np.floor((self.basis_parameters.ux_count - 1)/2))
        if self.basis_parameters.pad_w:
            end_ind -= int(self.basis_parameters.uy_count * np.floor(self.basis_parameters.ux_count/2))

        return matrix[begin_ind:(end_ind+1), :]

    def sparse_rows(self): # , ux_count, uy_count, wavelength_count):
        single_wavelength_rows = np.arange(0, self.basis_parameters.ux_count * self.basis_parameters.uy_count)
        offset = np.arange(0, self.basis_parameters.uy_count * self.basis_parameters.wavelength_count, self.basis_parameters.uy_count)
        rows = single_wavelength_rows.reshape(self.basis_parameters.ux_count * self.basis_parameters.uy_count, 1) + offset
        return rows.flatten('F').astype(int)

    def sparse_cols(self): # , ux_count, uy_count, wavelength_count):
        single_wavelength_cols = np.arange(0, self.basis_parameters.wavelength_count)
        return np.repeat(single_wavelength_cols, self.basis_parameters.ux_count * self.basis_parameters.uy_count).astype(int)


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
    :type d: float
    :type s: float
    :type l: float
    :type pol_angle: float
    :type ux_count: int
    :type uy_count: int
    :type wavelength: numpy.ndarray
    :type wavelength_count: int
    :type pad_w: bool
    :type trim_w: bool
    """

    def __init__(self, basis_type,
                 n0, n1, n2o, n2e, n3,
                 ux_range, uy_range,
                 d, s, l,
                 pol_angle,
                 ux_count=None, uy_count=None,
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
        self.pol_angle_rad  = np.radians(pol_angle)
        self.pad_w      = pad_w
        self.trim_w     = trim_w
        self.orig_wavelength = wavelength
        self.orig_wavelength_count = wavelength_count

    def set_wavelength(self, wavelength):
        self.wavelength = wavelength
        self.wavelength_count = len(wavelength)
        self.orig_wavelength = wavelength
        self.orig_wavelength_count = len(wavelength)

        if self.pad_w:
            self._pad_wavelength()

    def _pad_wavelength(self):
        pre_wavelength_spacing = np.abs(self.wavelength[1] - self.wavelength[0])
        pre_padding = self.wavelength[0] + pre_wavelength_spacing * np.arange(-np.floor((self.ux_count-1)/2), 0)

        post_wavelength_spacing = np.abs(self.wavelength[-1] - self.wavelength[-2])
        post_padding = self.wavelength[-1] + post_wavelength_spacing * np.arange(1, np.floor(self.ux_count/2) + 1)

        self.wavelength = np.hstack((pre_padding, self.wavelength, post_padding))
        self.wavelength_count = len(self.wavelength)

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
        assert self.pol_angle_rad is not None
        assert self.pad_w is not None
        assert self.trim_w is not None
        assert self.orig_wavelength is not None
        assert self.orig_wavelength_count == len(self.orig_wavelength)
        return True
