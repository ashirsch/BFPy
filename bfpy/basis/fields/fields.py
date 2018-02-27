import numpy as np
from numba import vectorize
import bfpy.basis.fields.fresnel as frs


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
    wavelength = np.linspace(554, 700, 1024, dtype=np.float64)

    ux, uy = np.meshgrid(uxspan, uyspan)

    n0 = 1.0
    n1 = 1.0
    n2 = 1.7
    n2e = 1.7
    n3 = 1.5

    uz0 = oop_wave_number(ux, uy, n0)
    uz1 = oop_wave_number(ux, uy, n1)
    uz2s = oop_wave_number(ux, uy, n2)
    uz2s = oop_wave_number_birefringent(ux, uy, n2, n2e)
    uz3 = oop_wave_number(ux, uy, n3)

    rs10 = frs.single_interface_reflect_s(uz0, uz1)
    rs21 = frs.single_interface_reflect_s(uz1, uz2s)

    Rs = np.empty((180, 180, 1024), dtype=np.complex128)
    frs.total_interface_reflect_s(rs21, rs10, uz1, wavelength, 0.0, Rs)

    return Rs
