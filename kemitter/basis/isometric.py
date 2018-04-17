import sys
import time
import numpy as np
from numba import vectorize
import scipy.sparse as sp
from .basis import Basis, BasisParameters
from .fields import field


class IsometricEmitter(Basis):

    def __init__(self, pol_angle,
                 dipoles=('ED', 'MD'),
                 n0=1.0, n1=1.0, n2=1.0, n3=1.0,
			     d=10.0, s=10.0, l=0.0,
			     NA=1.3,
			     pad_w=False, trim_w=True,
                 wavelength=None, k_count=None, open_slit=True):

        try:
            assert n0 > 0
            assert n1 > 0
            assert n2 > 0
            assert n3 > 0
            assert NA > 0
            assert d >= 0
            assert s >= 0
            assert l >= 0
        except AssertionError:
            print("Invalid argument given.")
            return
        super().__init__()
        self.pol_angle = pol_angle
        self.dipoles = dipoles
        self.basis_names = list(self.dipoles)
        self.basis_parameters = BasisParameters(basis_type="ISOMETRIC",
                                                n0=n0, n1=n1, n2o=n2, n2e=n2, n3=n3,
                                                ux_range=(-NA, NA), uy_range=(-NA, NA),
                                                d=d, s=s, l=l,
                                                pol_angle=self.pol_angle,
                                                pad_w=pad_w,
                                                trim_w=trim_w)
        if wavelength is not None and k_count is not None:
            self.define_observation_parameters(wavelength, k_count, open_slit)

    def build(self):
        if not self.is_defined:
            raise RuntimeError("Basis is not well-defined. Ensure that all parameters are assigned properly.")

        print('\n============ Starting the kemitter ' + self.basis_parameters.basis_type + ' builder ============')
        print('Basis information:')
        print('    polarization angle: {0:d}'.format(self.pol_angle))
        print('    wavelengths:        {0:d}'.format(self.basis_parameters.orig_wavelength_count))
        print('    k grid size:        {0:d}'.format(self.basis_parameters.ux_count))
        t0 = time.time()
        print('\nCalculating fields:')
        field_set = field.Field(self.basis_parameters)
        field_set.calculate_fields(self.dipoles)
        isometric_bases = []
        sys.stdout.write('Forming sparse emission basis: ')
        if "ED" in self.dipoles:
            ed_isometric = isometric_emission(self.basis_parameters.pol_angle_rad,
                                              field_set.xpol.ED.x, field_set.ypol.ED.x,
                                              field_set.xpol.ED.y, field_set.ypol.ED.y,
                                              field_set.xpol.ED.z, field_set.ypol.ED.z)
            ed_isometric = self.sparse_column_major_offset(ed_isometric)
            isometric_bases.append(ed_isometric)
        if "MD" in self.dipoles:
            md_isometric = isometric_emission(self.basis_parameters.pol_angle_rad,
                                              field_set.xpol.MD.x, field_set.ypol.MD.x,
                                              field_set.xpol.MD.y, field_set.ypol.MD.y,
                                              field_set.xpol.MD.z, field_set.ypol.MD.z)
            md_isometric = self.sparse_column_major_offset(md_isometric)
            isometric_bases.append(md_isometric)
        # for each ed / md
        # reshape so that each wavelength patter is vectorized (ux_count*uy_count, wavelength_count) (each column a wavelength)
        # load into sparse matrix with spatial offset
        # concatenate ED and MD sparse matrices
        if len(isometric_bases) > 1:
            basis = sp.hstack(isometric_bases)
        else:
            basis = isometric_bases[0]
        # trim basis based on lambda range TODO: work on x-y flipping
        if self.basis_parameters.trim_w:
            basis = self.basis_trim(basis)
        # return the single sparse matrix
        self.basis_matrix = basis
        self.is_built = True
        t1 = time.time()
        sys.stdout.write('DONE\n')
        print('Elapsed time: {0:.2f} s'.format(t1-t0))


@vectorize("float64(float64,complex128,complex128,complex128,complex128,complex128,complex128)",
           target='parallel', nopython=True)
def isometric_emission(pol_angle, xpol_x, ypol_x, xpol_y, ypol_y, xpol_z, ypol_z):
    isometric = np.square(np.abs(np.cos(pol_angle) * ypol_x +
                                 np.sin(pol_angle) * xpol_x)) + \
                np.square(np.abs(np.cos(pol_angle) * ypol_y +
                                 np.sin(pol_angle) * xpol_y)) + \
                np.square(np.abs(np.cos(pol_angle) * ypol_z +
                                 np.sin(pol_angle) * xpol_z))
    return isometric
