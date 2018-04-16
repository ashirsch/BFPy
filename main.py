import numpy as np
import bfpy
import pickle

h_obs = bfpy.Observation()
h_obs.load()

v_obs = bfpy.Observation()
v_obs.load()

h_basis = bfpy.basis.IsometricEmitter(pol_angle=90, n2=1.7, n3=1.5)
v_basis = bfpy.basis.IsometricEmitter(pol_angle=0, n2=1.7, n3=1.5)

# ========= TEST CODE HERE =========
model = bfpy.model.Quadratic()
model.run([h_basis, v_basis], [h_obs, v_obs])
# ==================================

with open('final_test_sess.p', 'wb') as f:
    pickle.dump(model, f)
