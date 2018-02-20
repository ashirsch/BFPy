import numpy as np


class BasisFactory:

    @staticmethod
    def make_builder(basis_type, parameters):
        if basis_type == "EDIsoBuilder":
            return EDIsoBuilder(parameters)
        else:
            print("Bad builder type: " + basis_type)
            pass


class EDIsoBuilder:
    """
    :type basis_type: str
    :type basis_parameters: BasisParameters
    """

    def __init__(self, parameters):
        self.basis_type = "EDIso"
        self.basis_parameters = parameters

    def build(self):
        basis = np.eye(3)
        basis[0, 0] = self.basis_parameters.n0
        basis[1, 1] = self.basis_parameters.n1
        basis[2, 2] = self.basis_parameters.n2
        return basis


class BasisParameters:
    """
    :type n0: float
    :type n1: float
    :type n2: float
    """

    def __init__(self, n0, n1, n2):
        self.n0 = n0
        self.n1 = n1
        self.n2 = n2
