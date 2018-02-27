from bfpy.basis.fields import fields
import time

if __name__ == "__main__":
    print('starting')
    t0 = time.time()
    # ========= TEST CODE HERE =========
    Rs = fields.main()
    # ==================================
    t1 = time.time()
    print("Done with elapsed time: {0} sec".format(t1-t0))
