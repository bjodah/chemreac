#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Aqueous radiolysis
------------------

:download:`examples/aqueous_radiolysis.py` is an example of a rather large
system of reactions.

::

 $ python analytic_diffusion.py --help

.. exec::
   echo "::\\n\\n"
   python examples/examples/aqueous_radiolysis.py --help | sed "s/^/   /"


Here is an example generated by:

::

 $ python analytic_diffusion.py --doserate 25 --plot \
    --savefig analytic_diffusion.png


.. image:: ../_generated/aqueous_radiolysis.png


"""

from __future__ import absolute_import, division, print_function

# stdlib imports
import json
import os
from math import log10

# external imports
import argh
import numpy as np

# project internal imports
from chemreac import ReactionDiffusion
from chemreac.integrate import Integration
from chemreac.serialization import load
from chemreac.units import (
    kilogram, decimetre, gray, molar, second, metre
)
from chemreac.util.grid import generate_grid
from chemreac.util.plotting import plot_C_vs_t, save_and_or_show_plot


def integrate_rd(
        t0=1e-7, tend=.1, x0=1e-9, xend=0.1, doserate=15.0, N=1000, nt=512,
        nstencil=0, logy=False, logt=False, logx=False,
        name='aqueous_radiolysis', integrator='scipy', iter_type='default',
        linear_solver='default', ilu_limit=1000.0, first_step=0.0,
        n_jac_diags=0, eps_lin=0.0, num_jacobian=False, savefig='None',
        verbose=False, plot=False, plot_jacobians=False, profile_yep=False,
        use_log2=False,
):
    """
    Integrates the reaction system defined by
    :download:`aqueous_radiolysis.json <examples/aqueous_radiolysis.json>`
    """
    null_conc = 1e-24
    if nstencil == 0:
        nstencil = 3 if N > 1 else 1
    mu = 50.0*metre**-1  # linear attenuation
    x = generate_grid(x0, xend, N, logx, use_log2=use_log2)
    expb = (lambda x: 2**x) if use_log2 else np.exp
    _cb = expb if logx else (lambda arg: arg)
    lin_xcenters = _cb(x[:-1]+np.diff(x)/2)*metre
    doserate *= gray / second
    doseratefield = doserate*np.exp(-mu*lin_xcenters)
    rho = 1.0*kilogram*decimetre**-3  # kg/dm3
    rd = load(os.path.join(os.path.dirname(__file__), name+'.json'),
              ReactionDiffusion, N=N, logy=logy, logt=logt, logx=logx,
              fields=[doseratefield*rho, 0*doseratefield*rho], nstencil=nstencil,
              ilu_limit=ilu_limit, n_jac_diags=n_jac_diags, use_log2=use_log2)
    y0_by_name = json.load(open(os.path.join(os.path.dirname(__file__),
                                             name+'.y0.json'), 'rt'))

    # y0 with a H2 gradient
    y0 = np.array([[y0_by_name.get(k, null_conc) if k != 'H2' else
                    1e-3/(i+2) for k in rd.substance_names]
                   for i in range(rd.N)])*molar

    tout = np.logspace(log10(t0), log10(tend), nt)*second
    if profile_yep:
        import yep
        yep.start(os.path.join(os.path.dirname(__file__), name+'.prof'))
    integr = Integration(
        rd, y0, tout, with_jacobian=(not num_jacobian),
        integrator=integrator, iter_type=iter_type,
        linear_solver=linear_solver, first_step=first_step,
        **(dict(eps_lin=eps_lin) if eps_lin != 0.0 else {}))
    if profile_yep:
        yep.stop()

    if verbose:
        from pprint import pprint
        pprint(integr.info)

    if plot:
        import matplotlib.pyplot as plt
        conc_unit = molar
        bt_fmtstr = ("C(t) in bin {{0:.2g}}-{{1:.2g}} "
                     "with local doserate {}")
        ax = plt.subplot(2, 1, 1)
        plot_C_vs_t(integr, bi=0, ax=ax, substances=('H2', 'H2O2'),
                    ttlfmt=bt_fmtstr.format(np.round(rd.fields[0][0], 1)),
                    ylabel="C / "+str(conc_unit))
        ax = plt.subplot(2, 1, 2)
        plot_C_vs_t(integr, bi=N-1, ax=ax, substances=('H2', 'H2O2'),
                    ttlfmt=bt_fmtstr.format(np.round(rd.fields[0][N-1], 1)),
                    ylabel="C / "+str(conc_unit))
        plt.tight_layout()
        save_and_or_show_plot(savefig=savefig)

    if plot_jacobians:
        from chemreac.util.plotting import coloured_spy
        from chemreac.util.banded import get_dense
        import glob
        import matplotlib.pyplot as plt
        d = {}
        absmax = 0.0
        for fname in sorted(glob.glob('jac_*.dat')):
            B = np.fromfile(fname)
            absmax = max(absmax, np.max(np.abs(B)))
            d[fname] = B

        for fname, B in d.items():
            h = 3*rd.n+1
            assert B.size % h == 0
            B = B.reshape(h, B.size/h, order='F')
            coloured_spy(B[rd.n:, :], log=1)
            plt.savefig('B_'+fname+'.png')
            plt.close(plt.gcf())
            D = get_dense(B, rd.n, N, padded=True)
            coloured_spy(D, log=-1, symmetric_colorbar=absmax)
            plt.savefig('D_'+fname+'.png')
            plt.close(plt.gcf())
            print(fname, np.average(B[2*rd.n, :]) /
                  np.average(B[3*rd.n, :-rd.n]))
            print(fname, np.average(B[2*rd.n, :])/np.average(B[rd.n, rd.n:]))


if __name__ == '__main__':
    argh.dispatch_command(integrate_rd)
