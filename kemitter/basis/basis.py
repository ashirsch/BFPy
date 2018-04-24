from abc import ABC, abstractmethod
import numpy as np
import scipy.sparse as sp


class Basis(ABC):
    """Abstract base class for all basis types.

    This class defines the common interface used for defining sample geometries and building emission bases.
    Individual basis classes differ in how they implement their ``build()`` method, but users interface with them with
    the methods and attributes defined here.

    Notes:
        ``Basis`` should not be instantiated on its own. Only its children, which implement a ``build()`` method
        should be used directly.

    Attributes:
        basis_names (list of str): denotes the types and column-wise order of bases present in the built basis matrix.
        basis_matrix (csc_matrix): sparse matrix containing basis functions. Each column corresponds to a basis function
            of a particular basis type at a particular wavelength.
        is_built (bool): whether or not a basis matrix has been successfully built
        pol_angle (int or float): the polarization angle of the basis, in degrees.
        basis_parameters (BasisParameters): the parameter object containing information about sample geometry, optical
            properties, and observation-dependent information. Used by submodules for constructing emission bases.

    Warnings:
        Directly modifying the ``basis_parameters`` object after it's construciton is dangerous and should be
        considered deprecated. Use the child class's initializer and ``define_observation_parameters()`` methods
        instead to alter basis parameters.
    """
    def __init__(self):
        self.basis_names = None
        self.basis_matrix = None
        self.is_built = False
        self.pol_angle = None
        self.basis_parameters = None
        super().__init__()

    @property
    def is_defined(self):
        """bool: whether or not all basis parameters have been properly defined and the basis is ready to be built."""
        defined = True
        defined = defined and self.basis_parameters._verify_state()
        defined = defined and self.pol_angle is not None
        return defined

    @abstractmethod
    def build(self):
        """Abstract method for basis building. Implemented by each basis individually"""
        pass

    def define_observation_parameters(self, wavelength, k_count, open_slit=True):
        """Allows setting of basis parameters that depend on intended observation fitting.

        Args:
            wavelength (ndarray): 1D array of wavelength mapping values.
            k_count (int): the image size in the momentum dimension. In open slit case, denotes the resolution of
                the momentum-space basis functions by referring to the grid edge size. That is, each basis function
                will be calculated on a ``k_count X k_count`` sized grid.
            open_slit (bool): whether observation and corresponding basis should be of wide-angle type (ux_count > 1).
        """
        if self.basis_parameters is not None:
            self.basis_parameters.uy_count = k_count
            self.basis_parameters.ux_count = k_count if open_slit else 1
            self.basis_parameters.set_wavelength(wavelength)
        else:
            print("Geometric and optical parameters must be defined prior to loading of "
                  "observation-dependent parameters.")

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


class BasisParameters(object):
    """Parameter object for basis class

    Stores all necessary information about sample geometry, optical properties, and observation-dependent
    conditions such as measurement wavelength and numerical aperture for basis construction. Also holds flags
    for basis building procedures based on user preferences.

    Attributes:
        basis_type (str): the type of basis to be built with the object
        n0 (float): refractive index of the 0 layer (typically vacuum, 1.0)
        n1 (float): refractive index of the 1 layer
        n2o (float): in-plane refractive index of the emitter layer
        n2e (float): out-of-plane refractive index of the emitter layer
        n3 (float): refractive index of the substrate layer (typically quartz, 1.5)
        ux_range (tuple of float): the minimum and maximum normalized wavenumbers in the x direction
            e.g. (-NA, NA) for open slit
        uy_range (tuple of float): the minimum and maximum normalized wavenumbers in the y direction
            e.g. (-NA, NA) for open slit
        d (float): distance from emitter center to 2-3 layer interface, in nanometers
        s (float): distance from emitter center to 1-2 layer interface, in nanometers
        l (float): thickness of n1 layer
        pol_angle_rad (float): the polarization angle, given in degrees (stored in radians)
        ux_count (int): the number of samples in the x-momentum dimension
        uy_count (int): the number of samples in the y-momentum dimension
        wavelength (ndarray):  1D array of wavelength mapping values (with or without padding)
        wavelength_count (int): the number of wavelength values to be calculated (with or without padding)
        pad_w (bool): construct basis by padding wavelength values near edges of image (False by default)
        trim_w (bool): trim the basis matrix to the desired image dimensions by wavelength (True by default)
        orig_wavelength (ndarry): 1D array of wavelength mapping values (without padding)
        orig_wavelength_length (int): the number of wavelength values in the observation image (without padding)

    Warnings:
        Polarization angle is entered in the initializer in degrees, like other areas of the public interface,
        but is then converted to radians for basis calculations. Functions that use basis parameters should refer to
        the ``pol_angle_rad`` attribute when making any calculation with polarization angle, NOT the degree value stored
        in the basis object itself.
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
        """Initializer for basis parameter class"""
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
        """Setter for the wavelength mapping values

        Args:
            wavelength (ndarray):  1D array of wavelength mapping values. May be padded upon setting.
        """
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
