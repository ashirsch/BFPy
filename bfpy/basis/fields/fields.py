import numpy as np
from numba import vectorize
import bfpy.basis.fields.fresnel as frs
import bfpy.basis.fields.frs2 as frs2
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

    rs10 = frs.single_interface_reflect_s(uz0, uz1)
    rs21 = frs.single_interface_reflect_s(uz1, uz2s)
    rs23 = frs.single_interface_reflect_s(uz3, uz2s)  # PAY ATTENTION TO ARGUMENT ORDER
    ts23 = frs.single_interface_transmit_s(uz2s, uz3)

    Rs = np.zeros((180, 180, 1024), dtype=np.complex128)
    Tsxy = np.zeros_like(Rs)
    Tsz = np.zeros_like(Rs)
    t0 = time.time()
    # Rs_1_phase_term = frs.phase_term(uz1, wavelength, 0.0, 2.0)
    # tp = time.time()
    # frs.total_interface_reflect(rs21, rs10, Rs_1_phase_term, Rs)
    # frs.total_interface_reflection_unrolled(rs21, rs10, uz1, wavelength, 0.0, Rs)
    Rs = frs.total_interface_reflection_unrolled_jit_sig_p(rs21, rs10, uz1, wavelength, 0.0)
    # frs2.total_interface_reflection_one_mom(rs21, rs10, uz1, wavelength, 0.0, Rs)
    # frs.total_interface_transmission_xy_unrolled(ts23,rs23,uz2s,wavelength,d,s,Rs,Tsxy)
    # frs.total_interface_transmission_z_unrolled(ts23,rs23,uz2s,wavelength,d,s,Rs,Tsz)
    tr = time.time()

    # print("Phase time: {0}".format(tp-t0))
    print("TIR time: {0}".format(tr-t0))

    return Rs, Tsxy, Tsz
