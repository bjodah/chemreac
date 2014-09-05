# -*- coding: utf-8 -*-

import numpy as np


def get_coeff_mtx(substances, stoichs):
    """
    Parameters
    ==========
    substances: sequence of keys in stoichs dict pairs
    stoichs: sequence of pairs of dicts
        pairs of reactant and product dicts mapping substance keys
        to stoichiometric coefficients (integers)

    Returns
    =======
    2 dimensional array of shape (len(substances), len(stoichs))

    """
    A = np.zeros((len(substances), len(stoichs)))
    for ri, sb in enumerate(substances):
        for ci, (reac, prod) in enumerate(stoichs):
            A[ri,ci] = prod.get(sb, 0) - reac.get(sb, 0)
    return A


def decompose_yield_into_rate_coeffs(yields, stoichs, atol=1e-10):
    """
    Decomposes (radiolytic) yields into linear combination of
    stoichiometric production reactions

    Ak = y

    A is (n_species x n_reactions) matrix, k is "rate coefficient", y is yields

    Parameters
    ==========
    yields: OrderedDict
        specie names as keys and yields as values
    stoichs: list of 2-dict tuples
         giving stoiciometry (1st is reactant, 2nd is products),
         dict keys must match those of `yields`

    Returns
    =======
    1-dimensional array of effective rate coefficients.

    """
    # Sanity check:
    for ys in yields.keys():
        present = False
        for reac, prod in stoichs:
            if ys in reac or ys in prod:
                present = True
        assert present

    sbstncs = yields.keys()
    y = np.array(yields.values())
    A = get_coeff_mtx(sbstncs, stoichs)
    k, residuals, rank, s = np.linalg.lstsq(A, y)
    assert residuals < atol
    return k
