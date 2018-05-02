import numpy as np
from numba import vectorize, jit, prange


@vectorize("complex128(complex128,complex128)",
           target='parallel', nopython=True)
def single_interface_reflection_s(uz_l, uz_u):
    return (uz_u - uz_l) / (uz_u + uz_l)


@vectorize("complex128(complex128,complex128,float64,float64)",
           target='parallel', nopython=True)
def single_interface_reflection_p(uz_l, uz_u, n_l, n_u):
    return (n_l**2 * uz_u - n_u**2 * uz_l) / (n_l**2 * uz_u + n_u ** 2 * uz_l)


@jit("complex128[:,:,:](complex128[:,:],complex128[:,:],complex128[:,:],float64[:],float64)",
     parallel=True, nopython=True)
def total_interface_reflection(r21, r10, uz, wavelength, l):
    R = np.zeros((uz.shape[0],uz.shape[1],len(wavelength)), dtype=np.complex128)
    for ux in prange(uz.shape[0]):
        for uy in prange(uz.shape[1]):
            for w in prange(len(wavelength)):
                R[ux, uy, w] = (r21[ux,uy] + r10[ux,uy] * np.exp(2j * uz[ux,uy] * wavelength[w] * 2 * np.pi * l)) / \
                               (1 + r21[ux,uy] * r10[ux,uy] * np.exp(2j * uz[ux,uy] * wavelength[w] * 2 * np.pi * l))
    return R


@vectorize("complex128(complex128,complex128)",
           target='parallel', nopython=True)
def single_interface_transmission_s(uz_l, uz_u):
    return  (2.0 * uz_l) / (uz_u + uz_l)


@vectorize("complex128(complex128,complex128,float64,float64)",
           target='parallel', nopython=True)
def single_interface_transmission_p(uz_l, uz_u, n_l, n_u):
    return (2.0 * n_u**2 * uz_l) / (n_l**2 * uz_u + n_u**2 * uz_l) * (n_l/n_u)


@jit("complex128[:,:,:](complex128[:,:],complex128[:,:],complex128[:,:],float64[:],float64,float64,complex128[:,:,:])",
     parallel=True, nopython=True)
def total_interface_transmission_xy(t23, r23, uz, wavelength, d, s, R):
    T = np.zeros((uz.shape[0], uz.shape[1], len(wavelength)), dtype=np.complex128)
    for ux in prange(uz.shape[0]):
        for uy in prange(uz.shape[1]):
            for w in prange(len(wavelength)):
                T[ux, uy, w] = ((t23[ux,uy] * np.exp(1j*uz[ux, uy]/wavelength[w]*2*np.pi*d)) /
                                (1 - r23[ux,uy] * R[ux,uy,w] * np.exp((2j*uz[ux,uy]/wavelength[w]*2*np.pi*(d+s))))) * \
                               (1 - R[ux, uy, w] * np.exp((2j*uz[ux,uy]/wavelength[w]*2*np.pi*s)))
    return T


@jit("complex128[:,:,:](complex128[:,:],complex128[:,:],complex128[:,:],float64[:],float64,float64,complex128[:,:,:])",
     parallel=True, nopython=True)
def total_interface_transmission_z(t23, r23, uz, wavelength, d, s, R):
    T = np.zeros((uz.shape[0], uz.shape[1], len(wavelength)), dtype=np.complex128)
    for ux in prange(uz.shape[0]):
        for uy in prange(uz.shape[1]):
            for w in prange(len(wavelength)):
                T[ux,uy,w] = ((t23[ux,uy] * np.exp(1j*uz[ux,uy]/wavelength[w]*2*np.pi*d)) /
                              (1 - r23[ux,uy] * R[ux,uy,w] * np.exp((2j*uz[ux,uy]/wavelength[w]*2*np.pi*(d+s))))) * \
                             (1 + R[ux,uy,w] * np.exp((2j*uz[ux,uy]/wavelength[w]*2*np.pi*s)))
    return T
