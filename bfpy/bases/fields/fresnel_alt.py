import numpy as np
from numba import guvectorize, jit


@guvectorize("(complex128[:],complex128[:],complex128[:],float64[:],float64, complex128[:,:])",
             '(n),(n),(n),(m),() -> (n,m)', target='cpu', nopython=True)
def total_interface_reflection_one_mom(r21, r10, uz, wavelength, l, R):
    for u_ind in range(uz.shape[0]):
            R[u_ind] = (r21[u_ind] + r10[u_ind] * np.exp(2j * uz[u_ind] / wavelength * 2 * np.pi * l)) / \
                        (1 + r21[u_ind] * r10[u_ind] * np.exp(2j * uz[u_ind] / wavelength * 2 * np.pi * l))
# nope, takes 3 seconds


# Tsxy = ts23.*exp(1i*kz2*d)./(1-rs23.*Rs.*exp(2i*kz2*(d+s))).*(1-Rs.*exp(2i*kz2*s));
@guvectorize("(complex128[:,:],complex128[:,:],complex128[:,:],float64[:],float64,float64,complex128[:,:,:],complex128[:,:,:])",
             '(n,n),(n,n),(n,n),(m),(),(),(n,n,m) -> (n,n,m)', target='parallel', nopython=True)
def total_interface_transmission_xy(t23, r23, uz, wavelength, d, s, R, T):
    for w in range(len(wavelength)):
        T[:,:,w] = ((t23*np.exp(1j*uz/wavelength[w]*2*np.pi*d)) /
                    (1-r23*R[:,:,w]*np.exp((2j*uz/wavelength[w]*2*np.pi*(d +s))))) * \
                   (1 - R[:,:,w]*np.exp((2j*uz/wavelength[w]*2*np.pi*s)))


# Tsz = ts23.*exp(1i*kz2*d)./(1-rs23.*Rs.*exp(2i*kz2*(d+s))).*(1+Rs.*exp(2i*kz2*s));
@guvectorize("(complex128[:,:],complex128[:,:],complex128[:,:],float64[:],float64,float64,complex128[:,:,:],complex128[:,:,:])",
             '(n,n),(n,n),(n,n),(m),(),(),(n,n,m) -> (n,n,m)', target='parallel', nopython=True)
def total_interface_transmission_z(t23, r23, uz, wavelength, d, s, R, T):
    for w in range(len(wavelength)):
        T[:,:,w] = ((t23*np.exp(1j*uz/wavelength[w]*2*np.pi*d)) /
                    (1-r23*R[:,:,w]*np.exp((2j*uz/wavelength[w]*2*np.pi*(d+s))))) * \
                   (1 + R[:,:,w]*np.exp((2j*uz/wavelength[w]*2*np.pi*s)))


@guvectorize("(complex128[:,:],complex128[:,:],complex128[:,:],float64[:],float64,float64,complex128[:,:,:],complex128[:,:,:])",
             '(n,n),(n,n),(n,n),(m),(),(),(n,n,m) -> (n,n,m)', target='parallel', nopython=True)
def total_interface_transmission_xy_unrolled(t23, r23, uz, wavelength, d, s, R, T):
    for ux in range(uz.shape[0]):
        for uy in range(uz.shape[1]):
            for w in range(len(wavelength)):
                T[ux,uy,w] = ((t23[ux,uy] * np.exp(1j * uz[ux,uy] / wavelength[w] * 2 * np.pi * d)) /
                            (1 - r23[ux,uy] * R[ux,uy,w] * np.exp((2j * uz[ux,uy] / wavelength[w]*2*np.pi*(d+s))))) * \
                           (1 - R[ux,uy,w]*np.exp((2j*uz[ux,uy]/wavelength[w]*2*np.pi*s)))


@guvectorize("(complex128[:,:],complex128[:,:],complex128[:,:],float64[:],float64,float64,complex128[:,:,:],complex128[:,:,:])",
             '(n,n),(n,n),(n,n),(m),(),(),(n,n,m) -> (n,n,m)', target='parallel', nopython=True)
def total_interface_transmission_z_unrolled(t23, r23, uz, wavelength, d, s, R, T):
    for ux in range(uz.shape[0]):
        for uy in range(uz.shape[1]):
            for w in range(len(wavelength)):
                T[ux,uy,w] = ((t23[ux,uy]*np.exp(1j*uz[ux,uy]/wavelength[w]*2*np.pi*d)) /
                            (1-r23[ux,uy]*R[ux,uy,w]*np.exp((2j*uz[ux,uy]/wavelength[w]*2*np.pi*(d +s))))) * \
                           (1 + R[ux,uy,w]*np.exp((2j*uz[ux,uy]/wavelength[w]*2*np.pi*s)))
# Unrolling went from 19 seconds for Rs, Tsxy, Tsz to 9.8 seconds

# recalculating phase term does not significantly impact performance
# it is generally better for numba to re-do calculations than to make many and move matrices around in python/memory
@guvectorize("(complex128[:,:],complex128[:,:],complex128[:,:],float64[:],float64, complex128[:,:,:])",
             '(n,n),(n,n),(n,n),(m),() -> (n,n,m)', target='parallel', nopython=True)
def total_interface_reflection(r21, r10, uz, wavelength, l, R):
    for w in range(len(wavelength)):
        R[:, :, w] = (r21 + r10 * np.exp(2j * uz * wavelength[w] * 2 * np.pi * l)) / \
                     (1 + r21 * r10 * np.exp(2j * uz * wavelength[w] * 2 * np.pi * l))
# Takes 2.54 seconds with (180,180,1024)


@guvectorize("(complex128[:,:],complex128[:,:],complex128[:,:],float64[:],float64, complex128[:,:,:])",
             '(n,n),(n,n),(n,n),(m),() -> (n,n,m)', target='parallel', nopython=True)
def total_interface_reflection_unrolled(r21, r10, uz, wavelength, l, R):
    for ux in range(uz.shape[0]):
        for uy in range(uz.shape[1]):
            for w in range(len(wavelength)):
                R[ux, uy, w] = (r21[ux,uy] + r10[ux,uy] * np.exp(2j * uz[ux,uy] / wavelength[w] * 2 * np.pi * l)) / \
                             (1 + r21[ux,uy] * r10[ux,uy] * np.exp(2j * uz[ux,uy] / wavelength[w] * 2 * np.pi * l))
# Takes 1.06 seconds


@jit(parallel=True, nopython=True)
def total_interface_reflection_unrolled_jit(r21, r10, uz, wavelength, l):
    R = np.zeros((uz.shape[0],uz.shape[1],len(wavelength)), dtype=np.complex128)
    for w in range(len(wavelength)):
        for ux in range(uz.shape[0]):
            for uy in range(uz.shape[1]):
                R[ux, uy, w] = (r21[ux,uy] + r10[ux,uy] * np.exp(2j * uz[ux,uy] * wavelength[w] * 2 * np.pi * l)) / \
                             (1 + r21[ux,uy] * r10[ux,uy] * np.exp(2j * uz[ux,uy] * wavelength[w] * 2 * np.pi * l))
    return R
# Takes 1.78 seconds


@jit("complex128[:,:,:](complex128[:,:],complex128[:,:],complex128[:,:],float64[:],float64)",parallel=True, nopython=True)
def total_interface_reflection_unrolled_jit_sig(r21, r10, uz, wavelength, l):
    R = np.zeros((uz.shape[0],uz.shape[1],len(wavelength)), dtype=np.complex128)
    for w in range(len(wavelength)):
        for uy in range(uz.shape[1]):
            for ux in range(uz.shape[0]):
                R[ux, uy, w] = (r21[ux,uy] + r10[ux,uy] * np.exp(2j * uz[ux,uy] * wavelength[w] * 2 * np.pi * l)) / \
                             (1 + r21[ux,uy] * r10[ux,uy] * np.exp(2j * uz[ux,uy] * wavelength[w] * 2 * np.pi * l))
    return R
# takes 1.21 seconds



# NUMBA IS FASTER WHEN IT CAN VECTORIZE ALL AT ONCE - OVERHEAD OF MOVING AROUND BIG MATRICES IS VERY HIGH
# The following code (broken up and wrapped to allow single calculation of phase term) is slower than
# the above function total_interface_reflection(...)
#
# def phase_term(uz, wavelength, l, refl):
#     phase_shape = (uz.shape[0], uz.shape[1], len(wavelength))
#     phase = np.zeros(phase_shape, dtype=np.complex128)
#     _calculate_phase_term(uz, wavelength, l, refl, phase)
#     return phase
#
#
# @guvectorize("(complex128[:,:],float64[:],float64,float64,complex128[:,:,:])",
#              '(n,n),(m),(),() -> (n,n,m)', target='parallel', nopython=True)
# def _calculate_phase_term(uz, wavelength, l, refl, phase):
#     for w in range(len(wavelength)):
#         phase[:,:,w] = np.exp(refl * 1j * uz * wavelength[w] * 2.0 * np.pi * l)
#
#
# @guvectorize("(complex128[:,:],complex128[:,:],complex128[:,:,:], complex128[:,:,:])",
#              '(n,n),(n,n),(n,n,m) -> (n,n,m)', target='parallel', nopython=True)
# def total_interface_reflect(r21, r10, phase_term, R):
#     for w in range(phase_term.shape[2]):
#         R[:, :, w] = (r21 + r10 * phase_term[:, :, w]) / (1 + r21 * r10 * phase_term[:, :, w])
