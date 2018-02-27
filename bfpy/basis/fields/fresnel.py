import numpy as np
from numba import vectorize, guvectorize


@vectorize("complex128(complex128,complex128)",
           target='parallel', nopython=True)
def single_interface_reflect_s(uz_l, uz_u):
    return (uz_u - uz_l) / (uz_u + uz_l)


@guvectorize("(complex128[:,:],complex128[:,:],complex128[:,:],float64[:],float64, complex128[:,:,:])",
             '(n,n),(n,n),(n,n),(m),() -> (n,n,m)', target='parallel', nopython=True)
def total_interface_reflect_s(rs21, rs10, uz1, wavelength, l, Rs):
    for w in range(len(wavelength)):
        Rs[:, :, w] = (rs21 + rs10 * np.exp(2j * uz1 * wavelength[w] * l)) / \
                      (1 + rs21 * rs10 * np.exp(2j * uz1 * wavelength[w] * l))
