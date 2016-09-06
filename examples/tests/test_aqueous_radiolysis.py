# -*- coding: utf-8 -*-

from aqueous_radiolysis import integrate_rd
from chemreac.util.testing import veryslow


@veryslow
def test_integrate_rd():
    integr = integrate_rd()
    assert integr.info['success']
