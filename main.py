from bfpy.basis import basis_factory as bf

if __name__ == "__main__":
    builder = bf.BasisFactory.make_builder(bf.BasisParameters("EDIso", 1, 2, 3))
    basis = builder.build()

    print(basis)
