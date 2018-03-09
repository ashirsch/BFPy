import time
from .basis_factory import BasisFactory


class Basis(object):
    def __init__(self):
        self.basis_matrix = None
        self.built = False

    def build(self, basis_parameters):
        builder = BasisFactory.make_builder(basis_parameters)

        t0 = time.time()
        self.basis_matrix = builder.build()
        t1 = time.time()

        self.built = True
        print(basis_parameters.basis_type + " basis successfully built.")
        print("Elapsed build time: {0:.3f}".format(t1-t0))
