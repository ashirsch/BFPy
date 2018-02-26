from .basis_factory import BasisFactory


class Basis(object):
    def __init__(self):
        self.basis_matrix = None

    def build(self, basis_parameters):
        builder = BasisFactory.make_builder(basis_parameters)
        self.basis_matrix = builder.build()
