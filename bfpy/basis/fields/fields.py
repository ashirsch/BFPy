import numpy as np
from numba import vectorize
import bfpy.basis.fields.fresnel as frs
import time


@vectorize("complex128(complex128,complex128,float64)",
           target='parallel', nopython=True)
def oop_wave_number(ux, uy, n):
    return np.sqrt(n ** 2 - ux ** 2 - uy ** 2)


@vectorize("complex128(complex128,complex128,float64,float64)",
           target='parallel', nopython=True)
def oop_wave_number_birefringent(ux, uy, n_o, n_e):
    return np.sqrt(n_o ** 2 - (n_o / n_e)**2 * (ux ** 2 + uy ** 2))


def main():
    uxspan = np.linspace(-1.3, 1.3, 180, dtype=np.complex128)
    uyspan = np.linspace(-1.3, 1.3, 180, dtype=np.complex128)
    wavelength = np.linspace(554.7395, 684.6601, 1024, dtype=np.float64)

    ux, uy = np.meshgrid(uxspan, uyspan)

    n0 = 1.0
    n1 = 1.0
    n2 = 1.7
    n2e = 1.7
    n3 = 1.5

    d = 10.0
    s = 10.0

    uz0 = oop_wave_number(ux, uy, n0)
    uz1 = oop_wave_number(ux, uy, n1)
    uz2s = oop_wave_number(ux, uy, n2)
    uz2p = oop_wave_number_birefringent(ux, uy, n2, n2e)
    uz3 = oop_wave_number(ux, uy, n3)

    rs10 = frs.single_interface_reflection_s(uz0, uz1)
    rs21 = frs.single_interface_reflection_s(uz1, uz2s)
    rs23 = frs.single_interface_reflection_s(uz3, uz2s) # PAY ATTENTION TO ARGUMENT ORDER
    ts23 = frs.single_interface_transmission_s(uz2s, uz3)

    rp10 = frs.single_interface_reflection_p(uz0, uz1, n0, n1)
    rp21 = frs.single_interface_reflection_p(uz1, uz2p, n1, n2)
    rp23 = frs.single_interface_reflection_p(uz3, uz2p, n3, n2) # PAY ATTENTION TO ARGUMENT ORDER
    tp23 = frs.single_interface_transmission_p(uz2p, uz3, n2, n3)

    t0 = time.time()
    Rs = frs.total_interface_reflection(rs21, rs10, uz1, wavelength, 0.0)
    Tsxy = frs.total_interface_transmission_xy(ts23, rs23, uz2s, wavelength, d, s, Rs)
    Tsz = frs.total_interface_transmission_z(ts23, rs23, uz2s, wavelength, d, s, Rs)
    tr = time.time()

    Rp = frs.total_interface_reflection(rp21, rp10, uz1, wavelength, 0.0)
    Tpxy = frs.total_interface_transmission_xy(tp23, rp23, uz2p, wavelength, d, s, Rp)
    Tpz = frs.total_interface_transmission_z(tp23, rp23, uz2p, wavelength, d, s, Rp)

    print("Rs+Tsxy+Tsz time: {0}".format(tr-t0))

    return Rp, Tpxy, Tpz
