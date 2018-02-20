import basis_factory as bf

if __name__ == "__main__":
    print("Available types:")
    print(bf.BasisFactory.__subclasses__())

    builder = bf.BasisFactory.make_builder("EDIsoBuilder", bf.BasisParameters(1, 2, 3))
    basis = builder.build()

    print(basis)
