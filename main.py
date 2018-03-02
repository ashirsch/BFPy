from bfpy.basis.fields import fields
import time
import numba

if __name__ == "__main__":
    print("starting numba with {0} threads".format(numba.config.NUMBA_NUM_THREADS))
    t0 = time.time()
    # ========= TEST CODE HERE =========
    Rp, Tpxy, Tpz = fields.main()
    print(Rp[123,32,785])
    print(Tpxy[123,32,785])
    print(Tpz[123,32,785])
    # ==================================
    t1 = time.time()
    print("Done with elapsed time: {0} sec".format(t1-t0))
