# -*- coding: utf-8 -*-
"""
chemreac.core
=============
In ``chemreac.core`` you will find :py:class:`ReactionDiffusion` which
is the class describing the system of ODEs.

"""
from __future__ import (absolute_import, division, print_function)

from collections import defaultdict, OrderedDict
from functools import reduce
import inspect
from itertools import chain
from operator import add
import os

import numpy as np

from .chemistry import mk_sn_dict_from_names, ReactionSystem
from .util.pyutil import monotonic
from .units import get_derived_unit, to_unitless, linspace
from .constants import get_unitless_constant

from ._chemreac import PyReactionDiffusion

Geom_names = {'f': 'Flat', 'c': 'Cylindrical', 's': 'Spherical'}


class ReactionDiffusionBase(object):

    def to_ReactionSystem(self, substance_names=None):
        substance_names = substance_names or self.substance_names
        rxns = []
        for ri in range(self.nr):
            rxn = self.to_Reaction(ri, substance_names)
            rxns.append(rxn)
        return ReactionSystem(rxns, mk_sn_dict_from_names(substance_names))

    @classmethod
    def from_ReactionSystem(cls, rsys, variables=None, fields=None, **kwargs):
        """
        Make a :class:`ReactionDiffusion` from a :class:`chempy.ReactionSystem`.

        Parameters
        ----------
        rsys : chempy.ReactionSystem
        variables : dict
        fields: optional
        unit_registry : dict, optional
        \*\*kwargs :
            Keyword arguments passed on to :class:`ReactionDiffusion`

        """
        from chempy.kinetics.rates import RadiolyticBase
        mass_action_rxns = []
        radiolytic_rxns = []
        for rxn in rsys.rxns:
            for key in chain(rxn.reac, rxn.prod, rxn.inact_reac):
                if key not in rsys.substances:
                    raise ValueError("Unkown substance name: %s" % key)
            if isinstance(rxn.param, RadiolyticBase):
                radiolytic_rxns.append(rxn)
            else:
                mass_action_rxns.append(rxn)

        # Handle radiolytic yields
        if kwargs.get('unit_registry', None) is not None:
            yield_unit = get_unit(kwargs['unit_registry'], 'radyield')
        else:
            yield_unit = 1
        yields = OrderedDict()
        for rxn in radiolytic_rxns:
            for doserate_name, g_val in rxn.param.g_values(variables).items():
                if doserate_name not in yields:
                    yields[doserate_name] = defaultdict(lambda: 0*yield_unit)
                for k in rxn.keys():
                    n, = rxn.net_stoich([k])
                    if k not in yields[doserate_name]:
                        yields[doserate_name][k] = n*g_val
                    else:
                        yields[doserate_name][k] += n*g_val
        g_values = [rsys.as_per_substance_array(v, unit=yield_unit, raise_on_unk=True) for v in yields.values()]
        g_value_parents = []
        for k in yields:
            parent = None
            for r in radiolytic_rxns:
                if parent is None:
                    parent = r.reac
                else:
                    if parent != r.reac:
                        raise ValueError("Mixed parents for %s" % k)
            if parent == {}:
                g_value_parents.append(-1)
            else:
                raise NotImplementedError("Concentraion dependent radiolysis not supported.")

        if fields is None:
            # Each doserate_name gets its own field:
            fields = [[variables['density']*variables[dname]]*kwargs.get('N', 1) for dname in yields]
            if fields == [[]]:
                fields = None

        def _kwargs_updater(key, attr):
            if attr in kwargs:
                return
            try:
                kwargs[attr] = [getattr(rsys.substances[sn], key) for sn in
                                rsys.substances]
            except AttributeError:
                try:
                    kwargs[attr] = [rsys.substances[sn].data[key]
                                    for sn in rsys.substances]
                except KeyError:
                    pass

        for key, attr in zip(
                ['D', 'mobility', 'name', 'latex_name'],
                ['D', 'mobility', 'substance_names',
                 'substance_latex_names']):
            _kwargs_updater(key, attr)

        if kwargs.get('unit_registry', None) is None:
            cb = ReactionDiffusion
        else:
            cb = ReactionDiffusion.nondimensionalisation
        return cb(
            rsys.ns,
            [reduce(add, [[i]*rxn.reac.get(k, 0) for i, k
                          in enumerate(rsys.substances)]) for rxn in mass_action_rxns],
            [reduce(add, [[i]*rxn.prod.get(k, 0) for i, k
                          in enumerate(rsys.substances)]) for rxn in mass_action_rxns],
            [rxn.rate_expr().rate_coeff(variables) for rxn in mass_action_rxns],
            stoich_inact=[reduce(add, [
                [i]*(0 if rxn.inact_reac is None else
                     rxn.inact_reac.get(k, 0)) for i, k in enumerate(rsys.substances)
            ]) for rxn in mass_action_rxns],
            g_values=g_values,
            g_value_parents=g_value_parents,
            fields=fields,
            **kwargs)

    def to_Reaction(self, ri, substance_names=None):
        """
        Convenience method for making a Reaction instance
        for reaction index ri
        """
        from .chemistry import Reaction
        if substance_names is None:
            substance_names = self.substance_names
        return Reaction(
            {substance_names[i]: self.stoich_active[ri].count(i) for
             i in range(self.n)},
            {substance_names[i]: self.stoich_prod[ri].count(i) for
             i in range(self.n)},
            param=self.k[ri],
            inact_reac={
                substance_names[i]: self.stoich_inact[ri].count(i) for
                i in range(self.n)})

    def alloc_fout(self):
        return np.zeros(self.n*self.N)

    def alloc_jout(self, banded=None, order='C', pad=0):
        if pad is True:
            pad = self.n*self.n_jac_diags
        if pad is False:
            pad = 0
        if banded is None:
            banded = self.N > 1
        if order == 'C':
            rpad, cpad = 0, pad
        elif order == 'F':
            rpad, cpad = pad, 0
        else:
            raise ValueError("Order must be 'C' or 'F'")

        if banded:
            nr = 2*(self.n*self.n_jac_diags) + 1 + rpad
            nc = self.n*self.N + cpad
            return np.zeros((nr, nc), order=order)
        else:
            return np.zeros((self.n*self.N + rpad, self.n*self.N + cpad),
                            order=order)

    def alloc_jout_compressed(self):
        from block_diag_ilu import alloc_compressed
        return alloc_compressed(self.N, self.n, self.n_jac_diags)

    @property
    def ny(self):
        return self.N*self.n


def get_unit(unit_registry, key):
    try:
        return get_derived_unit(unit_registry, key)
    except KeyError:
        return dict(
            radyield=unit_registry['amount']/get_unit(unit_registry, 'energy'),
            field=get_unit(unit_registry, 'energy')/(get_unit(
                unit_registry, 'time') * get_unit(unit_registry, 'length')**3)
        )[key]


def g_units(unit_registry, g_value_parents):
    """ Forms the unit of radiolytic yield

    Parameters
    ----------
    g_value_parents: iterable of integers
        parent substance indices (-1 indicates no parent)
    """
    g_units = []
    for parent in g_value_parents:
        if parent == -1:
            g_units.append(get_unit(unit_registry, 'radyield'))
        else:
            g_units.append(get_unit(unit_registry, 'radyield') /
                           get_unit(unit_registry, 'concentration'))
    return g_units


def k_units(unit_registry, reaction_orders):
    return [get_unit(unit_registry, 'concentration')**(
        1-order)/get_unit(unit_registry, 'time')
            for order in reaction_orders]


class ReactionDiffusion(PyReactionDiffusion, ReactionDiffusionBase):
    """
    Object representing the numerical model, with callbacks for evaluating
    derivatives and jacobian.

    Instances provide the methods:

    - :meth:`f`
    - :meth:`dense_jac_rmaj`


    - ``f(t, y, fout)``
    - ``dense_jac_rmaj(t, y, jout)``
    - ``dense_jac_cmaj(t, y, jout)``
    - ``banded_jac_cmaj(t, y, jout)``
    - ``banded_packed_jac_cmaj(t, y, jout)``

    some of which are used by e.g. :func:`~chemreac.integrate.integrate_scipy`

    Additional convenience attributes (not used by underlying C++ class):

    - :py:attr:`substance_names`
    - :py:attr:`substance_latex_names`

    .. note::
        only the four first arguments (up to k) are considered positional


    Parameters
    ----------
    n: integer
        Number of species.
    stoich_active: list of lists of integer indices
        Reactant index lists per reaction.
    stoich_prod: list of lists of integer indices
        Product index lists per reaction.
    k: 1-dimensional array
        Reaction rate coefficients.
    N: integer
        Number of compartments (default: 1 if ``x==None`` else ``len(x)-1``).
    D: sequence of floats
        Diffusion coefficients (of length n).
    z_chg: sequence of integers
        1-dimensional array of ion charges.
    mobility: sequence of floats
        Mobility of ions.
    x: sequence of floats or pair of flats or float, optional
        Compartment boundaries (of length N+1), if x is a pair of floats it is
        expanded into ``linspace(x[0], x[1], N+1)``, if x is a float it is
        expanded into ``linspace(0, x, N+1)``. Default: ``1``.
    stoich_inact: list of lists of integer indices
        List of inactive reactant index lists per reaction.n, default: ``[]``.
    geom: str (letter)
        Any in 'fcsp', i.e. flat, cylindrical, spherical, periodic (flat).
    logy: bool
        f and \*_jac_\* routines operate on log_b(concentration).
    logt: bool
        f and \*_jac_\* routines operate on log_b(time).
    logx: bool
        f and \*_jac_\* routines operate on log_b(space).
    nstencil: integer
        Number of points used in finite difference scheme.
    lrefl: bool
        Reflective left boundary (default: True).
    rrefl: bool
        Reflective right boundary (default: True).
    auto_efield: bool
        Calculate electric field from concentrations (default: False).
    surf_chg: pair of floats
        Total charge of surface (defaut: (0.0, 0.0)).
    eps_rel: float
        Relative permitivity of medium (dielectric constant).
    g_values: sequence of sequences of floats
        Per specie yields (amount / energy) per field type.
    g_value_parents: sequence of integers (optional)
        Indices of parents for each g_value. ``-1`` denotes no concentration
        dependence of g_values. default: ``[-1]*len(g_values)``.
    fields: sequence of sequence of floats
        Per bin field strength (energy / volume / time) per field type.
        May be calculated as product between density and doserate.
    modulated_rxns: sequence of integers
        Indicies of reactions subject to per bin modulation.
    modulation: sequence of sequences of floats
        Per bin modulation vectors for each index in modulated_rxns.
    ilu_limit: float
        Requirement on (average) diagonal dominance for performing ILU
        factorization.
    n_jac_diags: int
        Number of diagonals to include in Jacobian. ``-1`` delegates to
        environment variable ``CHEMREAC_N_JAC_DIAGS``. ``0`` makes it equal to
        ``nstencil - 1 / 2``.
    use_log2: bool
        Use base 2 for logarithmic transform instead of e.
    unit_registry: dict (optional)
        See ``chemreac.units.SI_base_registry`` for an example (default: None).


    Attributes
    ----------
    unit_registry: dict

    """
    # not used by C++ class
    extra_attrs = ['substance_names', 'substance_latex_names']

    # subset of extra_attrs optionally passed by user
    kwarg_attrs = ['substance_names', 'substance_latex_names']

    _substance_names = None
    _substance_latex_names = None

    @property
    def substance_names(self):
        return self._substance_names or list(map(str, range(self.n)))

    @substance_names.setter
    def substance_names(self, names):
        self._substance_names = names

    @property
    def substance_latex_names(self):
        return self._substance_latex_names or list(map(str, range(self.n)))

    @substance_latex_names.setter
    def substance_latex_names(self, latex_names):
        self._substance_latex_names = latex_names

    def __new__(cls, n, stoich_active, stoich_prod, k,  # *, i.e. last pos arg
                N=0, D=None, z_chg=None,
                mobility=None, x=None, stoich_inact=None, geom='f',
                logy=False, logt=False, logx=False, nstencil=None,
                lrefl=True, rrefl=True, auto_efield=False, surf_chg=None,
                eps_rel=1.0, g_values=None, g_value_parents=None,
                fields=None,
                modulated_rxns=None,
                modulation=None,
                unit_registry=None,
                ilu_limit=None,
                n_jac_diags=-1,
                use_log2=False,
                faraday_const=None,
                vacuum_permittivity=None,
                **kwargs):
        if N == 0:
            if x is None:
                N = 1
            else:
                N = len(x)-1
        nstencil = nstencil or (1 if N == 1 else 3)
        if N < nstencil:
            raise ValueError("N must be >= nstencil")

        if z_chg is None:
            z_chg = list([0]*n)
        if mobility is None:
            mobility = np.zeros(n)
        if N > 1:
            assert n == len(D)
            assert n == len(z_chg)
            assert n == len(mobility)
        else:
            if D is None:
                D = np.zeros(n)

        if x is None:
            x = 1.0

        try:
            if len(x) == N+1:
                if not monotonic(x, 1, True):
                    raise ValueError("x must be strictly positive monotonic")
                _x = x
            elif len(x) == 2:
                _x = linspace(x[0], x[1], N+1)
            elif len(x) == 1:
                raise TypeError
            else:
                raise ValueError("Don't know what to do with len(x) == %d" %
                                 len(x))
        except TypeError:
            _x = linspace(0, x, N+1)

        if stoich_inact is None:
            stoich_inact = list([[]]*len(stoich_active))
        if len(stoich_inact) != len(stoich_active):
            raise ValueError("length mismatch")
        if len(stoich_active) != len(stoich_prod):
            raise ValueError("length mismatch")
        if len(stoich_active) != len(k):
            raise ValueError("length mismatch")

        if geom not in 'fcs':
            raise ValueError("Unkown geom: %s" % geom)

        if surf_chg is None:
            surf_chg = (0.0, 0.0)

        # Handle g_values
        if g_values is None:
            g_values = []
        else:
            if fields is not None:
                assert len(g_values) == len(fields)
            for gv in g_values:
                assert len(gv) == n
        if g_value_parents is None:
            g_value_parents = [-1]*len(g_values)
        else:
            if not len(g_values) == len(g_value_parents):
                raise ValueError("g_values and g_value_parents need to be of same length")

        if fields is None:
            fields = [[0.0]*N]*len(g_values)
        else:
            assert len(fields) == len(g_values)
            for fld in fields:
                assert len(fld) == N

        if modulated_rxns is not None:
            if len(modulated_rxns) != len(modulation):
                raise ValueError("len(modulated_rxns) != len(modulation)")
        if modulation is not None:
            if modulated_rxns is None:
                raise ValueError("modulated_rxns missing.")
            if any(len(arr) != N for arr in modulation):
                raise ValueError("An array in modulation of size != N")

        _k = np.asarray(k)
        if _k.ndim != 1:
            raise ValueError("Rates vector has inproper dimension")

        rd = super(ReactionDiffusion, cls).__new__(
            cls, n, stoich_active, stoich_prod, _k,
            N, D, z_chg, mobility, _x, stoich_inact, 'fcs'.index(geom), logy,
            logt, logx, g_values, g_value_parents, fields,
            modulated_rxns or [], modulation or [], nstencil, lrefl, rrefl,
            auto_efield, surf_chg, eps_rel,
            faraday_const or get_unitless_constant(
                unit_registry, 'Faraday_constant'),
            vacuum_permittivity or get_unitless_constant(
                unit_registry, 'vacuum_permittivity'),
            ilu_limit=(float(os.environ.get('CHEMREAC_ILU_LIMIT', 1000)) if
                       ilu_limit is None else ilu_limit),
            n_jac_diags=(int(os.environ.get('CHEMREAC_N_JAC_DIAGS', 1)) if
                         n_jac_diags is -1 else n_jac_diags),
            use_log2=use_log2
        )

        rd.unit_registry = unit_registry

        for attr in cls.kwarg_attrs:
            if attr in kwargs:
                setattr(rd, '_' + attr, kwargs.pop(attr))
        if kwargs:
            raise KeyError("Unkown kwargs: ", kwargs.keys())
        return rd

    def __reduce__(self):
        args = inspect.getargspec(self.__new__).args[1:]
        return (self.__class__, tuple(getattr(self, attr) for attr in args))

    _prop_unit = {
        'mobility': 'electrical_mobility',
        'D': 'diffusion',
        'fields': 'field',
        'k': (k_units, 'reac_orders'),  # special case
        'g_values': (g_units, 'g_value_parents'),
        'x': 'length',
        'surf_chg': 'charge',
    }

    @classmethod
    def nondimensionalisation(cls, n, stoich_active, stoich_prod, k, **kwargs):
        """ Alternative constructor taking arguments with units """
        reac_orders = map(len, stoich_active)
        _k_units = k_units(kwargs['unit_registry'], reac_orders)
        k_unitless = [to_unitless(kv, ku) for kv, ku in zip(k, _k_units)]
        g_values_unitless = [np.asarray([to_unitless(yld, yld_unit) for yld in gv]) for gv, yld_unit in zip(
            kwargs.get('g_values', []), g_units(kwargs['unit_registry'], kwargs.get('g_value_parents', [])))]
        for key, rep in cls._prop_unit.items():
            val = kwargs.pop(key, None)
            if val is not None:
                if isinstance(rep, tuple):
                    pass
                else:
                    kwargs[key] = to_unitless(
                        val, get_unit(kwargs['unit_registry'], rep))

        return cls(n, stoich_active, stoich_prod, k_unitless,
                   g_values=g_values_unitless, **kwargs)

    @property
    def reac_orders(self):
        return map(len, self.stoich_active)

    def with_units(self, prop):
        rep = self._prop_unit[prop]
        if isinstance(rep, tuple):
            return [v*u for v, u in zip(getattr(self, prop), rep[0](
                self.unit_registry, getattr(self, rep[1])))]
        else:
            return np.asarray(getattr(self, prop))*get_unit(
                self.unit_registry, rep)

    def set_with_units(self, prop, val):
        rep = self._prop_unit[prop]
        if isinstance(rep, tuple):
            raise NotImplementedError("Not implemented for %s" % prop)
        else:
            setattr(self, prop, to_unitless(val, get_unit(
                self.unit_registry, rep)))
