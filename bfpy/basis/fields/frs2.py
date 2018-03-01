import numpy as np
from numba import guvectorize, jit


@guvectorize("(complex128[:],complex128[:],complex128[:],float64[:],float64, complex128[:,:])",
             '(n),(n),(n),(m),() -> (n,m)', target='cpu', nopython=True)
def total_interface_reflection_one_mom(r21, r10, uz, wavelength, l, R):
    for u_ind in range(uz.shape[0]):
            R[u_ind] = (r21[u_ind] + r10[u_ind] * np.exp(2j * uz[u_ind] / wavelength * 2 * np.pi * l)) / \
                        (1 + r21[u_ind] * r10[u_ind] * np.exp(2j * uz[u_ind] / wavelength * 2 * np.pi * l))
# nope, takes 3 seconds



