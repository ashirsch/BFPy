import numpy as np
from numba import vectorize, jit, prange
from . import fresnel as frs
from . import dipole as dip

class Field(object):
    """
    :type basis_parameters: BasisParameters
    """
    def __init__(self, basis_parameters):
        self.xpol = Field.PolFieldSet()
        self.ypol = Field.PolFieldSet()
        self.u    = Field.WavenumberSet()
        self.__bp = basis_parameters

    class PolFieldSet(object):
        def __init__(self):
            self.ED = None
            self.MD = None
            self.EQ = None

        class PolDipoleField(object):
            def __init__(self, dipole):
                self.x = None
                self.y = None
                self.z = None
                self.dipole = dipole

        def is_empty(self):
            return ((self.ED is None) and (self.MD is None) and (self.EQ is None))

    class WavenumberSet(object):
        def __init__(self):
            self.x   = None
            self.y   = None
            self.z0  = None
            self.z1  = None
            self.z2s = None
            self.z2p = None
            self.z3  = None

    def calculate_fields(self, dipoles):
        # calculate normalized wavenumbers
        self._calculate_wavenumbers()
        Tsxy, Tsz, Tpxy, Tpz = self._calculate_transmission_coeffs()
        if "ED" in dipoles:
            self.ypol.ED = Field.PolFieldSet.PolDipoleField("ED")
            self.xpol.ED = Field.PolFieldSet.PolDipoleField("ED")

            self.ypol.ED.x = dip._ypol_edx(self.u.x, self.u.y, self.u.z2s, self.u.z3,
                                           Tpxy, Tsz, self.__bp.n2o, self.__bp.n3)
            self.ypol.ED.y = dip._ypol_edy(self.u.x, self.u.y, self.u.z2s, self.u.z3,
                                           Tpxy, Tsz, self.__bp.n2o, self.__bp.n3)
            self.ypol.ED.z = dip._ypol_edz(self.u.x, self.u.y, self.u.z2s, self.u.z3,
                                           Tpz, self.__bp.n2o, self.__bp.n3)

            # TODO: for closed slit case, will need special treatment for xpol; for now permutation will suffice
            self.xpol.ED.x = np.transpose(self.ypol.ED.x, (1, 0 ,2))
            self.xpol.ED.y = np.transpose(self.ypol.ED.y, (1, 0, 2))
            self.xpol.ED.z = np.transpose(self.ypol.ED.z, (1, 0, 2))
            print("generated ED fields")
        if "MD" in dipoles:
            self.ypol.MD = Field.PolFieldSet.PolDipoleField("MD")
            self.xpol.MD = Field.PolFieldSet.PolDipoleField("MD")

            self.ypol.MD.x = dip._ypol_mdx(self.u.x, self.u.y, self.u.z2p, self.u.z3,
                                       Tsxy, Tpz, self.__bp.n2o, self.__bp.n3)
            self.ypol.MD.y = dip._ypol_mdy(self.u.x, self.u.y, self.u.z2p, self.u.z3,
                                       Tsxy, Tpz, self.__bp.n2o, self.__bp.n3)
            self.ypol.MD.z = dip._ypol_mdz(self.u.x, self.u.y, self.u.z2s, self.u.z3,
                                       Tsz, self.__bp.n2o, self.__bp.n3)

            # TODO: for closed slit case, will need special treatment for xpol; for now permutation will suffice
            self.xpol.MD.x = -np.transpose(self.ypol.MD.x, (1, 0, 2))
            self.xpol.MD.y = -np.transpose(self.ypol.MD.y, (1, 0, 2))
            self.xpol.MD.z = -np.transpose(self.ypol.MD.z, (1, 0, 2))
            print("generated MD fields")
        # TODO: EQ expansion
        # apply angular limit mask
        self._apply_mask()


    def _calculate_wavenumbers(self):
        ux_span = np.linspace(self.__bp.ux_range[0], self.__bp.ux_range[1], self.__bp.ux_count, dtype=np.complex128)
        uy_span = np.linspace(self.__bp.uy_range[0], self.__bp.uy_range[1], self.__bp.uy_count, dtype=np.complex128)

        self.u.x, self.u.y = np.meshgrid(ux_span, uy_span)

        self.u.z0  = oop_wave_number(self.u.x, self.u.y, self.__bp.n0)
        self.u.z1  = oop_wave_number(self.u.x, self.u.y, self.__bp.n1)
        self.u.z2s = oop_wave_number(self.u.x, self.u.y, self.__bp.n2o)
        self.u.z2p = oop_wave_number_birefringent(self.u.x, self.u.y, self.__bp.n2o, self.__bp.n2e)
        self.u.z3  = oop_wave_number(self.u.x, self.u.y, self.__bp.n3)

    def _calculate_transmission_coeffs(self):
        rs10 = frs.single_interface_reflection_s(self.u.z0, self.u.z1)
        rs21 = frs.single_interface_reflection_s(self.u.z1, self.u.z2s)
        rs23 = frs.single_interface_reflection_s(self.u.z3, self.u.z2s)  # PAY ATTENTION TO ARGUMENT ORDER
        ts23 = frs.single_interface_transmission_s(self.u.z2s, self.u.z3)

        rp10 = frs.single_interface_reflection_p(self.u.z0, self.u.z1, self.__bp.n0, self.__bp.n1)
        rp21 = frs.single_interface_reflection_p(self.u.z1, self.u.z2p, self.__bp.n1, self.__bp.n2o)
        rp23 = frs.single_interface_reflection_p(self.u.z3, self.u.z2p, self.__bp.n3, self.__bp.n2o)  # PAY ATTENTION TO ARGUMENT ORDER
        tp23 = frs.single_interface_transmission_p(self.u.z2p, self.u.z3, self.__bp.n2o, self.__bp.n3)

        Rs   = frs.total_interface_reflection(rs21, rs10, self.u.z1, self.__bp.wavelength, self.__bp.l)
        Tsxy = frs.total_interface_transmission_xy(ts23, rs23, self.u.z2s,
                                                   self.__bp.wavelength, self.__bp.d, self.__bp.s, Rs)
        Tsz  = frs.total_interface_transmission_z(ts23, rs23, self.u.z2s,
                                                  self.__bp.wavelength, self.__bp.d, self.__bp.s, Rs)

        Rp   = frs.total_interface_reflection(rp21, rp10, self.u.z1, self.__bp.wavelength, self.__bp.l)
        Tpxy = frs.total_interface_transmission_xy(tp23, rp23, self.u.z2p,
                                                   self.__bp.wavelength, self.__bp.d, self.__bp.s, Rp)
        Tpz  = frs.total_interface_transmission_z(tp23, rp23, self.u.z2p,
                                                  self.__bp.wavelength, self.__bp.d, self.__bp.s, Rp)
        return Tsxy, Tsz, Tpxy, Tpz

    def _apply_mask(self):
        # find indices in each 2D matrix that is beyond the angular limit
        mask_r, mask_c = np.where(np.sqrt(self.u.x**2 + self.u.y**2) > (self.__bp.uy_range[1] * 1.001))
        for active_field in [self.xpol, self.ypol]:
            for active_dipole in [active_field.ED, active_field.MD, active_field.EQ]:
                if active_dipole is not None:
                    active_dipole.x[mask_r, mask_c, :] = 0
                    active_dipole.y[mask_r, mask_c, :] = 0
                    active_dipole.z[mask_r, mask_c, :] = 0


@vectorize("complex128(complex128,complex128,float64)",
           target='parallel', nopython=True)
def oop_wave_number(ux, uy, n):
    return np.sqrt(n ** 2 - ux ** 2 - uy ** 2)


@vectorize("complex128(complex128,complex128,float64,float64)",
           target='parallel', nopython=True)
def oop_wave_number_birefringent(ux, uy, n_o, n_e):
    return np.sqrt(n_o ** 2 - (n_o / n_e)**2 * (ux ** 2 + uy ** 2))
