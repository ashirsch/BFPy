import time
import numba
import numpy as np
import bfpy
import bfpy.vis.visualization as bfpvis

print("starting numba with {0} threads".format(numba.config.NUMBA_NUM_THREADS))

h_basis = bfpy.basis.IsometricEmitter(pol_angle=90, dipoles=("ED", "MD"))
h_basis.define(n2=1.7, n3=1.5)
h_basis.define_observation_parameters(np.linspace(554, 660, 1024, dtype=np.float64), 180)

t0 = time.time()
# ========= TEST CODE HERE =========
h_basis.build()
# ==================================
t1 = time.time()
print("Done with elapsed time: {0:.3f} sec".format(t1-t0))

bfpvis.basis_func_plot(h_basis, 400)
bfpvis.basis_func_plot(h_basis, 1424)
# bfpvis.basis_func_plot(v_basis, 400)
# bfpvis.basis_func_plot(v_basis, 1424)
