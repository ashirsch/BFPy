import numpy as np


class BasisFactory:

    @staticmethod
    def make_builder(parameters):
        if parameters.basis_type == "EDIso":
            return EDIsoBuilder(parameters)
        else:
            print("Bad builder type: " + parameters.basis_type)
            pass


class EDIsoBuilder:
    """
    :type basis_parameters: BasisParameters
    """

    def __init__(self, basis_parameters):
        self.basis_parameters = basis_parameters

    def build(self):
        basis = np.eye(3)
        basis[0, 0] = self.basis_parameters.n0
        basis[1, 1] = self.basis_parameters.n1
        basis[2, 2] = self.basis_parameters.n2
        return basis


class BasisParameters:
    """
    :type basis_type: str
    :type n0: float
    :type n1: float
    :type n2: float
    """

    def __init__(self, basis_type, n0, n1, n2):
        self.basis_type = basis_type
        self.n0 = n0
        self.n1 = n1
        self.n2 = n2
