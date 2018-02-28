from bfpy.basis.fields import fields
import time
import numba

if __name__ == "__main__":
    print("starting numba with {0} threads".format(numba.config.NUMBA_NUM_THREADS))
    t0 = time.time()
    # ========= TEST CODE HERE =========
    Rs, Tsxy, Tsz = fields.main()
    print(Rs[0,0,0])
    print(Tsxy[0,0,0])
    print(Tsz[0,0,0])
    # ==================================
    t1 = time.time()
    print("Done with elapsed time: {0} sec".format(t1-t0))
