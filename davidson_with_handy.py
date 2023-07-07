def davidson_with_handy(transformer, conditioner, number_of_eigen, longer_of_orbs = 6, number_of_electrons = 6, tolerance = 1e-8):
    """takes the transformer function as defined in the candy paper that converts the initial configuration interaction vector into a new one to escape the expensive matrix multiplication step, a free conditioner, which mediates the digital of the hamiltonian matrics, the integer length of the agen system decayed, and the tolerance for the algorithm to converge."""
    # stored by saving the step of the expensive matrix voltaic ton with the transformer function