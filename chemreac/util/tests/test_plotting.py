# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
matplotlib.use('Agg')  # travis-ci has no DISPLAY env var.
import matplotlib.axes

from chemreac import ReactionDiffusion
from chemreac.integrate import run
from chemreac.util.plotting import (
    coloured_spy, plot_jacobian, plot_per_reaction_contribution,
    plot_C_vs_t_in_bin, plot_C_vs_x, plot_C_vs_t_and_x, plot_bin_k_factors
)


def _get_decay_rd(N):
    return ReactionDiffusion(2, [[0]], [[1]], [3.14], N, D=[0.0, 0.0])


def test_coloured_spy():
    N = 6
    t = 0.0
    rd = _get_decay_rd(6)
    y0 = np.array([2.0, 3.0]*N)
    jout = rd.alloc_jout(order='F')
    rd.banded_packed_jac_cmaj(t, y0, jout)
    ax = coloured_spy(np.log(np.abs(jout)))
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_jacobian():
    # A -> B
    rd = _get_decay_rd(1)
    y0 = [3.0, 1.0]
    tout = np.linspace(0, 3.0, 7)
    yout, info = run(rd, y0, tout)
    axes = plot_jacobian(rd, tout, yout, [0, 1])
    for ax in axes:
        assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_per_reaction_contribution():
    rd = _get_decay_rd(1)
    tout = np.linspace(0, 3.0, 7)
    y0 = [3.0, 1.0]
    yout, info = run(rd, y0, tout)
    axes = plot_per_reaction_contribution(rd, tout, yout, [0, 1])
    for ax in axes:
        assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_C_vs_t_in_bin():
    N = 3
    rd = _get_decay_rd(N)
    tout = np.linspace(0, 3.0, 7)
    y0 = [3.0, 1.0]*N
    yout, info = run(rd, y0, tout)
    ax = plot_C_vs_t_in_bin(rd, tout, yout, 0)
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_C_vs_x():
    N = 3
    rd = _get_decay_rd(N)
    tout = np.linspace(0, 3.0, 7)
    y0 = [3.0, 1.0]*N
    yout, info = run(rd, y0, tout)
    ax = plot_C_vs_x(rd, tout, yout, [0, 1], 6)
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_C_vs_t_and_x():
    N = 3
    rd = _get_decay_rd(N)
    tout = np.linspace(0, 3.0, 7)
    y0 = [3.0, 1.0]*N
    yout, info = run(rd, y0, tout)
    ax = plot_C_vs_t_and_x(rd, tout, yout, 0)
    import mpl_toolkits.mplot3d
    assert isinstance(ax, mpl_toolkits.mplot3d.Axes3D)


def test_plot_bin_k_factors():
    # modulation in x means x_center
    # A -> B # mod1 (x**2)
    # C -> D # mod1 (x**2)
    # E -> F # mod2 (sqrt(x))
    # G -> H # no modulation
    k = np.array([3.0, 7.0, 13.0, 22.0])
    N = 5
    n = 8
    nr = 4
    D = np.zeros(n)
    x = np.linspace(3, 7, N+1)
    xc = x[:-1] + np.diff(x)/2
    bkf = [(xc[i]*xc[i], xc[i]**0.5) for i in range(N)]
    bkf_span = [2, 1]
    rd = ReactionDiffusion(
        n,
        [[i] for i in range(0, n, 2)],
        [[i] for i in range(1, n, 2)],
        k=k, N=N, D=D, x=x, bin_k_factor=bkf,
        bin_k_factor_span=bkf_span
    )
    ax = plot_bin_k_factors(rd, indices=[0, 1])
    assert isinstance(ax, matplotlib.axes.Axes)