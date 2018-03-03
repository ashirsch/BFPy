import time
import numba
import numpy as np
from bfpy.basis.fields import fields
from bfpy.basis.basis_factory import BasisParameters

if __name__ == "__main__":
    print("starting numba with {0} threads".format(numba.config.NUMBA_NUM_THREADS))

    bp = BasisParameters("EDIso", 1.0, 1.0, 1.7, 1.7, 1.5,
                         (-1.3,1.3), (-1.3,1.3),
                         180, 180,
                         10.0, 10.0, 0.0,
                         np.linspace(554.7395, 684.6601, 1024, dtype=np.float64))

    t0 = time.time()
    # ========= TEST CODE HERE =========
    field = fields.Field(basis_parameters=bp)
    field.calculate_fields(["ED", "MD"])
    # ==================================
    t1 = time.time()
    print("Done with elapsed time: {0} sec".format(t1-t0))
