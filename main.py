import time
import numba
import numpy as np
from bfpy.basis import basis
from bfpy.basis.basis_factory import BasisParameters
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("starting numba with {0} threads".format(numba.config.NUMBA_NUM_THREADS))

    h_bp = BasisParameters(basis_type="EDIso",
                         n0=1.0, n1=1.0, n2o=1.7, n2e=1.7, n3=1.5,
                         ux_range=(-1.3,1.3), uy_range=(-1.3,1.3),
                         ux_count=180, uy_count=180,
                         d=10.0, s=10.0, l=0.0,
                         wavelength=np.linspace(554.7395, 684.6601, 1024, dtype=np.float64),
                         wavelength_count=1024,
                         pol_angle=0,
                         pad_w=True)
    v_bp = BasisParameters(basis_type="EDIso",
                           n0=1.0, n1=1.0, n2o=1.7, n2e=1.7, n3=1.5,
                           ux_range=(-1.3, 1.3), uy_range=(-1.3, 1.3),
                           ux_count=180, uy_count=180,
                           d=10.0, s=10.0, l=0.0,
                           wavelength=np.linspace(554.7395, 684.6601, 1024, dtype=np.float64),
                           wavelength_count=1024,
                           pol_angle=np.pi/2,
                           pad_w=True)

    t0 = time.time()
    # ========= TEST CODE HERE =========
    h_basis = basis.Basis()
    h_basis.build(basis_parameters=h_bp)
    v_basis = basis.Basis()
    v_basis.build(basis_parameters=v_bp)
    # ==================================
    t1 = time.time()
    print("Done with elapsed time: {0:.3f} sec".format(t1-t0))
    first_col = v_basis.basis_matrix[:,1865].todense()
    col = np.reshape(first_col, (180,1024), order='F')
    plt.imshow(col)
    plt.show()
