import numpy as np
import kemitter
import pickle

# h_obs = kemitter.Observation()
# h_obs.load()
#
# v_obs = kemitter.Observation()
# v_obs.load()
#
# h_basis = kemitter.basis.IsometricEmitter(pol_angle=90, n2=1.7, n3=1.5)
# v_basis = kemitter.basis.IsometricEmitter(pol_angle=0, n2=1.7, n3=1.5)
#
# with open('final_test_in.p', 'wb') as f:
#     pickle.dump((h_basis, v_basis, h_obs, v_obs), f)

with open('final_test_in.p', 'rb') as f:
    h_basis, v_basis, h_obs, v_obs = pickle.load(f)

# ========= TEST CODE HERE =========
model = kemitter.model.Quadratic(alpha=0.025)
model.run([h_basis, v_basis], [h_obs, v_obs])
# ==================================

with open('final_test_sess.p', 'wb') as f:
    pickle.dump(model, f)
