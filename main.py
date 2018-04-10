import time
import numba
import numpy as np
import bfpy
import bfpy.vis.visualization as bfpvis

print("starting numba with {0} threads".format(numba.config.NUMBA_NUM_THREADS))

h_basis = bfpy.basis.IsometricEmitter(pol_angle=90, dipoles=("ED", "MD"), n2=1.7, n3=1.5)
h_basis.define_observation_parameters(np.linspace(554, 660, 1024, dtype=np.float64), 180)

or_basis = bfpy.basis.OrientedEmitter(pol_angle=0, n2e=1.7, n2o=1.6, n3=1.5, NA=1.5)
or_basis.define_observation_parameters(np.linspace(660, 740, 1024, dtype=np.float64), 180)

t0 = time.time()
# ========= TEST CODE HERE =========
h_basis.build()
or_basis.build()
# ==================================
t1 = time.time()
print("Done with elapsed time: {0:.3f} sec".format(t1-t0))

bfpvis.basis_func_plot(h_basis, 400)
bfpvis.basis_func_plot(h_basis, 1424)
bfpvis.basis_func_plot(or_basis, 400)
bfpvis.basis_func_plot(or_basis, 1424)
