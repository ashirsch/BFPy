import sys
import time
import numpy as np
from numba import vectorize
import scipy.sparse as sp
from .basis import Basis, BasisParameters
from .fields import field


class OrientedEmitter(Basis):

    def __init__(self, pol_angle,
                 dipoles=('ED'),
                 n0=1.0, n1=1.0, n2o=1.0, n2e=1.0, n3=1.0,
			     d=10.0, s=10.0, l=0.0,
			     NA=1.3,
			     pad_w=False, trim_w=True,
                 wavelength=None, k_count=None, open_slit=True):
        super().__init__()
        try:
            assert n0 > 0
            assert n1 > 0
            assert n2o > 0
            assert n2e > 0
            assert n3 > 0
            assert NA > 0
            assert d >= 0
            assert s >= 0
            assert l >= 0
        except AssertionError:
            print("Invalid argument given.")
            return
        self.pol_angle = pol_angle
        self.dipoles = dipoles
        self.basis_names = [orientation + dip for dip in self.dipoles for orientation in ('IP', 'OP')]
        self.basis_parameters = BasisParameters(basis_type="ORIENTED",
                                                n0=n0, n1=n1, n2o=n2o, n2e=n2e, n3=n3,
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
        sys.stdout.write('Forming sparse emission basis: ')
        oriented_bases = []
        if "ED" in self.dipoles:
            in_plane = in_plane_emission(self.basis_parameters.pol_angle_rad,
                                         field_set.xpol.ED.x, field_set.ypol.ED.x,
                                         field_set.xpol.ED.y, field_set.ypol.ED.y)
            in_plane = self.sparse_column_major_offset(in_plane)
            out_plane = out_plane_emission(self.basis_parameters.pol_angle_rad,
                                           field_set.xpol.ED.z, field_set.ypol.ED.z)
            out_plane = self.sparse_column_major_offset(out_plane)
            oriented_bases.append(in_plane)
            oriented_bases.append(out_plane)

        basis = sp.hstack(oriented_bases)

        if self.basis_parameters.trim_w:
            basis = self.basis_trim(basis)

        self.basis_matrix = basis
        self.is_built = True
        t1 = time.time()
        sys.stdout.write('DONE\n')
        print('Elapsed time: {0:.2f} s'.format(t1 - t0))


@vectorize("float64(float64,complex128,complex128,complex128,complex128)",
           target='parallel', nopython=True)
def in_plane_emission(pol_angle, xpol_x, ypol_x, xpol_y, ypol_y):
    in_plane = np.square(np.abs(np.cos(pol_angle) * ypol_x +
                                np.sin(pol_angle) * xpol_x)) + \
               np.square(np.abs(np.cos(pol_angle) * ypol_y +
                                np.sin(pol_angle) * xpol_y))
    return in_plane


@vectorize("float64(float64,complex128,complex128)",
           target='parallel', nopython=True)
def out_plane_emission(pol_angle, xpol_z, ypol_z):
    out_plane = np.square(np.abs(np.cos(pol_angle) * ypol_z +
                                 np.sin(pol_angle) * xpol_z))
    return out_plane
