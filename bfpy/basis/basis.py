import time
from .basis_factory import BasisFactory


class Basis(object):
    def __init__(self):
        self.basis_matrix = None
        self.built = False
        self.basis_parameters = None

    def build(self):
        builder = BasisFactory.make_builder(self.basis_parameters)

        t0 = time.time()
        self.basis_matrix = builder.build()
        t1 = time.time()

        self.built = True
        print(self.basis_parameters.basis_type + " basis successfully built.")
        print("Elapsed build time: {0:.3f}".format(t1-t0))
