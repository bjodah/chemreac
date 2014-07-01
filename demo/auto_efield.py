#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import

from math import log, erf, exp

import argh
import numpy as np

from chemreac import (
    ReactionDiffusion, FLAT, CYLINDRICAL, SPHERICAL, Geom_names
)
from chemreac.integrate import run

def sigm(x, lim=350., n=8):
    # Algebraic sigmoid to avoid overflow/underflow of 'double exp(double)'
    return x/((x/lim)**n+1)**(1./n)

sq2 = 2**0.5
pi = np.pi
sqpi = pi**0.5


def gaussian(x, mu, sigma, logy, logx, geom):
    # Formula for normalization from derived in following mathematica code:
    # $Assumptions = {(sigma | mu) \[Element] Reals, sigma > 0}
    # 1/Integrate[E^(-1/2*((x - mu)/sigma)^2), {x, -Infinity, Infinity}]
    # 1/Integrate[2*pi*x*E^(-1/2*((x - mu)/sigma)^2), {x, 0, Infinity}]
    # 1/Integrate[4*pi*x^2*E^(-1/2*((x - mu)/sigma)^2), {x, 0, Infinity}]
    if geom == FLAT:
        a = 1/sigma/(2*np.pi)**0.5
    elif geom == CYLINDRICAL:
        a = 1/pi/sigma/(2*exp(-mu**2/2/sigma**2)*sigma +\
                        mu*sq2*sqpi*(1 + erf(mu/(sq2*sigma))))
    elif geom == SPHERICAL:
        a = 1/2/pi/sigma/(2*exp(-mu**2/2/sigma**2)*mu*sigma +\
                          sq2*sqpi*(mu**2 + sigma**2)*(1 + erf(mu/sq2/sigma)))
    else:
        raise RuntimeError()

    b = -0.5*((x-mu)/sigma)**2
    if logy:
        return log(a) + b
    else:
        return a*np.exp(b)


def pair_of_gaussians(x, offsets, sigma, logy, logx, geom):
    try:
        sigma0, sigma1 = sigma[0], sigma[1]
    except:
        sigma0 = sigma1 = sigma
    x = np.exp(x) if logx else x
    xspan = (x[-1] - x[0])
    xl = x[0] + offsets[0]*xspan  # lower
    xu = x[0] + offsets[1]*xspan  # upper
    return (
        gaussian(x, xl, sigma0, logy, logx, geom),
        gaussian(x, xu, sigma1, logy, logx, geom)
    )


def integrate_rd(D=0., t0=1e-6, tend=7., x0=0.1, xend=1.0, N=256,
                 base=0.5, offset=0.25, mobility=3e-8, nt=25, geom='f',
                 logt=False, logy=False, logx=False, random=False,
                 nstencil=3, lrefl=False, rrefl=False,
                 num_jacobian=False, method='bdf', plot=False,
                 atol=1e-6, rtol=1e-6, random_seed=42, surf_chg=0.0,
                 sigma_q=101, sigma_skew=0.5):
    assert 0<= base and base <= 1
    assert 0 <= offset and offset <= 1
    if random_seed:
        np.random.seed(random_seed)
    n = 2
    tout = np.linspace(t0, tend, nt)
    geom = {'f': FLAT, 'c': CYLINDRICAL, 's': SPHERICAL}[geom]

    # Setup the grid
    _x0 = log(x0) if logx else x0
    _xend = log(xend) if logx else xend
    x = np.linspace(_x0, _xend, N+1)
    if random:
        x += (np.random.random(N+1)-0.5)*(_xend-_x0)/(N+2)

    # Setup the system
    stoich_reac = []
    stoich_prod = []
    k = []
    bin_k_factor = [[] for _ in range(N)]
    bin_k_factor_span = []

    sys = ReactionDiffusion(
        n, stoich_reac, stoich_prod, k, N,
        D=[D, D],
        z_chg=[1, -1],
        mobility=[mobility, -mobility],
        x=x,
        bin_k_factor=bin_k_factor,
        bin_k_factor_span=bin_k_factor_span,
        geom=geom,
        logy=logy,
        logt=logt,
        logx=logx,
        nstencil=nstencil,
        lrefl=lrefl,
        rrefl=rrefl,
        auto_efield=True,
        surf_chg=surf_chg,
        eps=80.10,  # water at 20 deg C
    )

    # Initial conditions
    sigma = (xend-x0)/sigma_q
    sigma = [(1-sigma_skew)*sigma, sigma_skew*sigma]
    y0 = np.vstack(pair_of_gaussians(sys.xcenters, [base+offset, base-offset],
                                     sigma, logy, logx, geom)).transpose()
    if logy:
        y0 = sigm(y0)
    if plot:
        # Plot initial E-field
        import matplotlib.pyplot as plt
        sys.calc_efield((np.exp(y0) if logy else y0).flatten())
        plt.subplot(4, 1, 3)
        plt.plot(sys.xcenters, sys.efield, label="E at t=t0")
        plt.plot(sys.xcenters, sys.xcenters*0, label="0")


    # Run the integration
    t = np.log(tout) if logt else tout
    yout, info = run(sys, y0.flatten(), t,
                     atol=atol, rtol=rtol,
                     with_jacobian=(not num_jacobian), method=method)
    Cout = np.exp(yout) if logy else yout

    # Plot results
    if plot:
        def _plot(y, ttl=None,  **kwargs):
            plt.plot(sys.xcenters, y, **kwargs)
            plt.xlabel('x / m')
            plt.ylabel('C / M')
            if ttl:
                plt.title(ttl)

        for i in range(nt):
            plt.subplot(4, 1, 1)
            c = 1-tout[i]/tend
            c = (1.0-c, .5-c/2, .5-c/2)
            _plot(Cout[i, :, 0], 'Simulation (N={})'.format(sys.N),
                  c=c, label='$z_A=1$' if i==nt-1 else None)
            _plot(Cout[i, :, 1], c=c[::-1],
                  label='$z_B=-1$' if i==nt-1 else None)
            plt.legend()

            plt.subplot(4, 1, 2)
            delta_y = Cout[i, :, 0] - Cout[i, :, 1]
            _plot(delta_y, 'Diff'.format(sys.N),
                  c=[c[2], c[0], c[1]],
                  label='A-B (positive excess)' if i==nt-1 else None)
            plt.legend(loc='best')
            plt.xlabel("$x~/~m$")
            plt.ylabel(r'Concentration / M')
        ylim = plt.gca().get_ylim()
        if N < 100:
            plt.vlines(sys.x, ylim[0], ylim[1],
                       linewidth=1.0, alpha=0.2, colors='gray')

        plt.subplot(4, 1, 3)
        plt.plot(sys.xcenters, sys.efield)
        plt.plot(sys.xcenters, sys.efield, label="E at t=tend")
        plt.xlabel("$x~/~m$")
        plt.ylabel("$E~/~V\cdot m^{-1}$")
        plt.legend()

        for i in range(3):
            plt.subplot(4, 1, i+1)
            ylim = plt.gca().get_ylim()
            for d in (-1, 1):
                center_loc = [x0+(base+d*offset)*(xend-x0)]*2
                plt.plot(np.log(center_loc) if logx else center_loc,
                         ylim, '--k')
        plt.subplot(4, 1, 4)
        for i in range(n):
            amount = [sys.integrated_conc(Cout[j,:,i]) for j in range(nt)]
            print(np.array(amount))
            plt.plot(tout, amount, c=c[::(1,-1)[i]])  # match colors of lines
        plt.xlabel('Time / s')
        plt.ylabel('Amount / mol')
        plt.tight_layout()
        plt.show()
    return tout, Cout, info, sys

if __name__ == '__main__':
    argh.dispatch_command(integrate_rd)