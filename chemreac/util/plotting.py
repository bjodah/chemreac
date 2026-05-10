# -*- coding: utf-8 -*-

"""
chemreac.util.plotting
----------------------

This module collects convenience functions to create matplotlib plots
of results.
"""

from math import floor, ceil, log

import numpy as np
from chemreac.chemistry import mk_sn_dict_from_names
from chemreac.units import get_derived_unit, to_unitless
from chemreac.util.analysis import solver_linear_error_from_integration
from chemreac.util.banded import get_jac_row_from_banded
from chemreac.util.pyutil import set_dict_defaults_inplace
import matplotlib.pyplot as plt


def _init_axes(ax=None):
    if ax is None:
        ax = plt.axes()
    else:
        if not hasattr(ax, 'plot'):
            ax = plt.axes(**ax)
    return ax


def save_and_or_show_plot(show=None, savefig='None', **kwargs):
    """ Save and/or show current matplotlib figure

    Parameters
    ----------
    show: bool or None
        Show plot, when None only show when savefig is not used
        default: None
    savefig: string
        path to output file of figure. If extension is html, mpld3
        will be used to generate a d3 backed html output.
    \\*\\*kwargs:
        keyword arguments passed on to ``matplotlib.pyplot.savefig``
    """
    if savefig is not None and savefig != 'None':
        if savefig.endswith('.html'):
            # Export using mpld3
            import mpld3
            open(savefig, 'wt').write(mpld3.fig_to_html(plt.gcf()))
        else:
            plt.savefig(savefig, **kwargs)

        if show is None:
            show = False
    else:
        if show is None:
            show = True

    if show:
        plt.show()


# if True:
#     SymLogNorm if np.any(A<0) else LogNorm
# if isinstance(log, int):
#     SymLogNorm(10**log)
# note: "norm" in kwargs overrides this.

def coloured_spy(A, cmap_name='coolwarm', log=False,
                 symmetric_colorbar=False, **kwargs):
    """
    Convenience function for using matplotlib to
    generate a spy plot for inspecting e.g. a jacobian
    matrix or its LU decomposition.

    Parameters
    ----------
    A: 2D array
        Array to inspect, populated e.g. by jacobian callback.
    cmap_name: string (default: coolwarm)
        name of matplotlib colormap to use, kwargs["cmap"] overrides this.
    log: bool or int
        when True: LogNorm/SymLogNorm is used,
        when integer: SymlogNorm(10**log). Note that "norm" in kwargs
        override this.
    symmetric_colorbar: bool or float
        to make divergent colormaps pass through zero as intended.
        if float: max abolute value of colormap (linear)

    Returns
    -------
    Pair (tuple) of axes plotted to (spy, colorbar)

    .. note:: colorbar does not play nicely with
            SymLogNorm why a custom colorbar axes is drawn.

    """
    from matplotlib.ticker import MaxNLocator
    from matplotlib.cm import get_cmap
    from matplotlib.colors import LogNorm, SymLogNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    A = np.asarray(A)
    if 'cmap' not in kwargs:
        kwargs['cmap'] = get_cmap(cmap_name)

    plt.figure()
    ax_imshow = plt.subplot(111)

    if log is not False and 'norm' not in kwargs:
        if (isinstance(symmetric_colorbar, (float, int)) and
           symmetric_colorbar is not False):
            Amin = -symmetric_colorbar
            Amax = symmetric_colorbar
        else:
            Amin = np.min(A[np.where(A != 0)])
            Amax = np.max(A[np.where(A != 0)])

        if symmetric_colorbar is True:
            Amin = -max(-Amin, Amax)
            Amax = -Amin

        if log is True:
            if np.any(A < 0):
                log = int(np.round(np.log10(Amax) - 13))
            else:
                minlog = int(floor(np.log10(Amin)))
                maxlog = int(ceil(np.log10(Amax)))
                tick_locations = [10**x for x in range(minlog, maxlog+1)]
                kwargs['norm'] = LogNorm()
        elif isinstance(log, int):
            tick_locations = []
            if Amin < 0:
                minlog = int(ceil(np.log10(-Amin)))
                tick_locations.extend([-(10**x) for x in range(
                    minlog, log-1, -1)])
                tick_locations.extend([0])
            else:
                tick_locations.extend([0])
                minlog = int(floor(np.log10(Amin)))
                tick_locations.extend([10**x for x in range(minlog, log+1)])

            if Amax < 0:
                pass  # Ticks already reach 0
            else:
                maxlog = int(ceil(np.log10(Amax)))
                tick_locations.extend([10**x for x in range(log, maxlog+1)])
            kwargs['norm'] = SymLogNorm(10**log)
        else:
            raise TypeError("log kwarg not understood: {}".format(log))
    else:
        tick_locations = np.linspace(np.min(A), np.max(A), 10)

    ax_imshow.imshow(A, interpolation='none', **kwargs)
    ya = ax_imshow.get_yaxis()
    ya.set_major_locator(MaxNLocator(integer=True))
    xa = ax_imshow.get_xaxis()
    xa.set_major_locator(MaxNLocator(integer=True))

    divider = make_axes_locatable(ax_imshow)
    ax_colorbar = divider.append_axes("right", size="5%", pad=0.05)

    def colorbar(ticks=None, norm=None, log10threshy=None):
        if isinstance(norm, (LogNorm, SymLogNorm)):
            levels = np.concatenate([
                np.linspace(ticks[i], ticks[i+1], 9, endpoint=False) for i
                in range(len(ticks)-1)
            ]+[np.array([ticks[-1]])]).flatten()
        else:
            levels = ticks
        xc = [np.zeros_like(levels), np.ones_like(levels)]
        yc = [levels, levels]
        ax_colorbar.contourf(xc, yc, yc, levels=levels, norm=norm,
                             cmap=kwargs['cmap'])
        if isinstance(norm, LogNorm):
            ax_colorbar.set_yscale('log')
        elif isinstance(norm, SymLogNorm):
            ax_colorbar.set_yscale('symlog', linthresh=10**log)
        ax_colorbar.yaxis.tick_right()
        ax_colorbar.set_xticks([])
        ax_colorbar.set_yticks(ticks)

    colorbar(ticks=tick_locations, norm=kwargs.get('norm', None))
    plt.tight_layout()
    return ax_imshow, ax_colorbar


def _get_jac_row_over_t(rd, tout, yout, indices, bi=0):
    # Note that you really need yout - not Cout
    Jout = rd.alloc_jout(banded=True)
    row_out = np.zeros((yout.shape[0], len(indices), rd.n))
    for i, y in enumerate(yout):
        rd.banded_jac_cmaj(tout[i], y.flatten(), Jout)
        Jtmp = Jout[rd.n*rd.n_jac_diags:, bi*rd.n:(bi + 1)*rd.n]
        row_out[i, :, :] = get_jac_row_from_banded(Jtmp, indices, rd.n)
    return row_out


def _get_per_rxn_out(rd, tout, yout, specie_indices):
    # Note that you really need yout - not Cout
    out = np.empty((yout.shape[0], len(specie_indices), rd.nr))
    for ti, y in enumerate(yout):
        flat_y = y.flatten()
        for j, si in enumerate(specie_indices):
            rd.per_rxn_contrib_to_fi(tout[ti], flat_y, si, out[ti, j, :])
    return out


DEFAULT = dict(
    c=('tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
       'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'black'),
    ls=('-', '--', ':', '-.')
)


def _plot_analysis(cb, labels, rd, tout, yout, indices, axes=None,
                   titles=None, linthresh=1e-10, logx=True,
                   legend_kwargs=None, ls=None, c=None):
    """
    Parameters
    ----------
    cb: callback
        callback with signature (rd, tout, yout, indices) returning
        3-dimensional array with shape (tout.size, len(axes), len(labels))
    labels: sequence of strings
        labels of individual plots
    rd: ReactionDiffusion instance
    tout: 1-dimensional array of floats
    yout: solver output
    indices: 4th argument for callback
    axes: sequence of matplotlib Axes instances
        (default: len(indices) number of subplot axes)
    titles: titles per axis
    linthresh: float
        symlog option 'linthresh' (default: 1e-10)
    logx: set x scale to 'log'
    legend_kwargs: dict
        dict of kwargs to pass to matplotlib legend function
        (default: {'loc': None, 'prop': {'size': 12}}), set
        to False to suppress legend.
    ls: sequence of strings
        linestyles
    c: sequence of strings
        colors
    """
    legend_kwargs = legend_kwargs or {}
    set_dict_defaults_inplace(legend_kwargs,
                              dict(loc=None, prop={'size': 12}))
    if axes is None:
        axes = [plt.subplot(len(indices), 1, i+1) for i in range(
            len(indices))]
    else:
        assert len(axes) == len(indices)
    ls = ls or DEFAULT['ls']
    c = c or DEFAULT['c']
    row_out = cb(rd, tout, yout, indices)
    for i, ax in enumerate(axes):
        ax.set_yscale('symlog', linthresh=linthresh)
        if logx:
            ax.set_xscale('log')
        istl = 0  # style counter
        for j, lbl in enumerate(labels):
            if np.all(np.abs(row_out[:, i, j]) < linthresh):
                continue
            ax.plot(tout, row_out[:, i, j], label=lbl, c=c[istl % len(c)],
                    ls=ls[istl % len(ls)])
            istl += 1
        if legend_kwargs is not False:
            ax.legend(**legend_kwargs)
        if titles:
            ax.set_title(titles[i])
        ax.set_xlabel("t / s")
    return axes


def plot_jacobian(rd, tout, yout, substances, **kwargs):
    """
    Plots time evolution of Jacobian values (useful when investigating
    numerical instabilities).

    Parameters
    ----------
    rd: ReactionDiffusion
        system at hand
    tout: array_like
        output time from integration
    yout: array_like
        output data from integration (differs from Cout for logy=True)
    substances: iterable of int or string
        indices or names of substances to plot jacobian values for
    """
    indices = [si if isinstance(si, int) else rd.substance_names.index(si) for
               si in substances]
    print_names = rd.substance_latex_names or rd.substance_names
    axes = _plot_analysis(_get_jac_row_over_t, print_names, rd, tout, yout,
                          indices, titles=[
                              print_names[i] for i in indices], **kwargs)
    for ax in axes:
        ax.set_ylabel(
            "$\\frac{\\partial r_{tot}}{\\partial C_i}~/~s^{-1}$")
    return axes


def plot_jacobian_from_integration(integr, substances, **kwargs):
    pass
    # a = integr.rd.unit_registry
    # return plot_jacobian(integr.rd, )  # TODO: finish this


def plot_per_reaction_contribution(integr, substances, equilibria=None,
                                   field_yields=False, **kwargs):
    """
    Plots contributions to concentration derivatives of selected
    substances from individual reactions.

    Parameters
    ----------
    integr: Integration instance
    substances: sequence of Substance instances
    equilibria: set of tuples of reaction indices (optional)
        When passed only net effect of equilibria reaction will be plotted
    field_yields: bool (default: False)
        If ``True`` contributions from g_values times field will be shown
    **kwargs: kwargs passed on to _plot_analysis

    Returns
    -------
    list of matplotlib.axes.Axes instances
    """
    rd = integr.rd
    if rd.N != 1:
        # should be quite straight forward to implement
        raise NotImplementedError

    indices = [ri if isinstance(ri, int) else rd.substance_names.index(ri)
               for ri in substances]
    if rd.substance_latex_names is not None:
        print_names = rd.substance_latex_names
        use_tex = True
    else:
        print_names = rd.substance_names
        use_tex = False
    substances = mk_sn_dict_from_names(rd.substance_names,
                                       latex_name=rd.substance_latex_names)

    if field_yields:
        # Let's use negative reaction indices for each field -1: 0, -2: 1
        rxn_indices = range(-len(rd.fields), 0)
    else:
        rxn_indices = []

    if equilibria is not None:
        equilibria_participants = []
        for rxnidxs in equilibria:
            equilibria_participants.extend(rxnidxs)
            rxn_indices.append(rxnidxs)
        for ri in range(rd.nr):
            if ri not in equilibria_participants:
                rxn_indices.append(ri)
    else:
        rxn_indices += range(rd.nr)

    labels = []
    for rxni in rxn_indices:
        if isinstance(rxni, int):
            if rxni >= 0:
                rxn = rd.to_Reaction(rxni)
                labels.append('R' + str(rxni) + ': ' +
                              ('$' + rxn.latex(substances) + '$') if
                              use_tex else str(rxn))
            else:
                # Field production!
                fi = -rxni - 1
                if rd.g_value_parents[fi] == -1:
                    # No parents
                    parent = ''
                else:
                    parent = (print_names[rd.g_value_parents[fi]] +
                              '$\\rightsquigarrow$' if use_tex else ' ~~~> ')
                labels.append('G_' + str(fi) + ': ' + parent + ', '.join(
                    [print_names[si] for si in range(rd.n) if
                     rd.g_values[fi][si] != 0]))
        else:
            rxn = rd.to_Reaction(rxni[0])
            labels.append(
                'R(' + ', '.join(map(str, rxni)) + '): ' +
                ('$' + rxn.latex(dict(zip(
                    rd.substance_names, print_names))) + '$') if use_tex
                else str(rxn))

    def cb(rd_, tout_, yout_, specie_indices_):
        bi = 0  # bin index, N=1 only implemented for now
        per_rxn = _get_per_rxn_out(rd_, tout_, yout_, specie_indices_)
        out = np.zeros((yout_.shape[0], len(specie_indices_),
                        len(rxn_indices)))
        for i, rxns in enumerate(rxn_indices):
            if isinstance(rxns, int):
                if rxns >= 0:
                    out[:, :, i] = per_rxn[:, :, rxns]
                else:
                    fi = -rxns - 1
                    for sii, si in enumerate(specie_indices_):
                        out[:, sii, i] = rd.g_values[fi][si]*rd.fields[fi][bi]
            else:
                for rxn_idx in rxns:
                    out[:, :, i] += per_rxn[:, :, rxn_idx]
        return out

    axes = _plot_analysis(cb, labels, rd, integr.tout, integr.yout, indices,
                          titles=['$'+print_names[i]+'$' for i in indices],
                          **kwargs)
    for ax in axes:
        ax.set_ylabel(r"Reaction rate / $M\cdot s^{-1}$")
    return axes


def _init_ax_substances_labels(rd, ax, substances, labels, xscale, yscale):
    # helper func..
    ax = ax or plt.subplot(1, 1, 1)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if substances is None:
        substance_idxs = range(rd.n)
    else:
        substance_idxs = [
            s if isinstance(s, int) else rd.substance_names.index(s) for
            s in substances]
    latex_names_None = rd.substance_latex_names in (None, [None]*rd.n)
    if labels is None:
        try:
            if not latex_names_None:
                names = ['$\\mathrm{'+n+'}$' for n in rd.substance_latex_names]
            else:
                names = rd.substance_names
        except AttributeError:
            names = list(map(str, substance_idxs))
        labels = [names[i] for i in substance_idxs]
    else:
        assert len(labels) == len(substance_idxs)
    return ax, substance_idxs, labels


def plot_C_vs_t(integr, **kwargs):
    return plot_C_vs_t_in_bin(
        integr.rd, integr.with_units('tout'), integr.with_units('Cout'), **kwargs)


def plot_C_vs_t_in_bin(
        rd, tout, Cout, bi=0, ax=None, labels=None,
        xscale='log', yscale='log', substances=None,
        ttlfmt=None, legend_kwargs=None,
        ls=None, c=None, xlabel=None, ylabel=None):
    """
    Plots bin local concentration as function of time for selected
    substances.

    Parameters
    ----------
    rd: ReactionDiffusion
    tout: 1D array of floats
    Cout: concentration trajectories from solver
    bi: bin index
    ax: Axes instance
    labels: sequence of strings
    xscale: matplotlib scale choice (e.g. 'log', 'symlog')
    yscale: matplotlib scale choice (e.g. 'log', 'symlog')
    substances: sequence of indies or names of substances
    ttlfmt: string formatted with bin boundaries (set to empty to suppress)
    legend_kwargs: dict
        kwargs passed to matplotlib legend function,
        (default: {'loc': None, 'prop': {'size': 12}}), set
        to False to suppress legend.
    ls: sequence of strings
        linestyles
    c: sequence of strings
        colors

    Returns
    =======
    Axes instance

    """
    if ttlfmt is None:
        if rd.N == 1:
            ttlfmt = "C(t)"
        else:
            ttlfmt = (r"C(t) in bin: {0:.2g} m $\langle$" +
                      r" x $\langle$ {1:.2g} m")
    legend_kwargs = legend_kwargs or {}
    set_dict_defaults_inplace(legend_kwargs,
                              dict(loc='upper left'))
    ls = ls or DEFAULT['ls']
    c = c or DEFAULT['c']
    ax, substances, labels = _init_ax_substances_labels(
        rd, ax, substances, labels, xscale, yscale)
    for i, lbl in zip(substances, labels):
        ax.plot(tout, Cout[:, bi, i], label=lbl,
                ls=ls[i % len(ls)], c=c[i % len(c)])
    try:
        ax.set_xlabel(xlabel or "t / " + str(tout.dimensionality.latex),
                      {'fontsize': 16})
        ax.set_ylabel(ylabel or "C / " + str(Cout.dimensionality.latex),
                      {'fontsize': 16})
    except AttributeError:
        pass
    if ttlfmt:
        ax.set_title(ttlfmt.format(rd.x[bi], rd.x[bi+1]))
    if legend_kwargs is not False:
        ax.legend(**legend_kwargs)
    return ax


def plot_C_vs_x(rd, tout, Cout, substances, ti, ax=None, labels=None,
                xscale='log', yscale='log', basetitle="C(x)", ls=None, c=None):
    """
    Plots concentration as function of x for selected
    substances at time index 'ti'.

    Parameters
    ----------
    rd: ReactionDiffusion
    tout: 1D array of floats
    Cout: concentration trajectories from solver
    substances: sequence of indies or names of substances
    ti: int
        time index
    ax: Axes instance
    labels: sequence of strings
    xscale: matplotlib scale choice (e.g. 'linear', 'log', 'symlog')
    yscale: matplotlib scale choice (e.g. 'linear', 'log', 'symlog')
    basetitle: string

    Returns
    =======
    Axes instance
    """
    ax, substances, labels = _init_ax_substances_labels(
        rd, ax, substances, labels, xscale, yscale)
    ls = ls or DEFAULT['ls']
    c = c or DEFAULT['c']
    x_edges = np.repeat(rd.x, 2)[1:-1]
    for i, lbl in zip(substances, labels):
        y_edges = np.repeat(Cout[ti, :, i], 2)
        ax.plot(x_edges, y_edges, label=lbl, ls=ls[i % len(ls)], c=c[i % len(c)])
    ax.set_xlabel("x / m")
    ax.set_ylabel("C / M")
    ax.set_title(basetitle+" at t = {0:.3g} s".format(tout[ti]))
    ax.legend(loc='best', prop={'size': 12})
    return ax


def plot_faded_time(integr, substance, rgb=(1., .5, .5), log_color=False,
                    yscale=None):
    si = (substance if isinstance(substance, int) else
          integr.rd.substance_names.index(substance))
    for ti in range(integr.tout.size):
        if log_color:
            span = log(integr.tout[-1]) - log(integr.tout[0])
            c = (log(integr.tout[ti]) - log(integr.tout[0]))/span
        else:
            c = integr.tout[ti]/integr.tout[-1]
        c = c*rgb[0], c*rgb[1], c*rgb[2]
        plt.plot(integr.rd.xcenters, integr.Cout[ti, :, si], c=c)
    if yscale is not None:
        plt.gca().set_yscale(yscale)


def plot_C_vs_t_and_x(rd, tout, Cout, substance, ax=None, log10=False,
                      **plot_kwargs):
    """
    Plots 3D surface of concentration as function of time and x for a
    selected substance.

    Parameters
    ----------
    rd: ReactionDiffusion
    tout: 1D array of floats
    Cout: concentration trajectories from solver
    substance: int or string
        index or name of substance
    ax: Axes instance
    log10: bool
        Use log logarithmic (base 10) axis
    **plot_kwargs:
        passed onto plot_surface

    Returns
    =======
    Axes3D instance
    """
    # it would be nice to accpet kwargs
    #    xscale='log', yscale='log', zscale='log'
    # but it's currently not supported by matplotlib:
    # http://matplotlib.1069221.n5.nabble.com/
    #     plot-surface-fails-with-log-axes-td10206.html
    substance = (substance if isinstance(substance, int) else
                 rd.substance_names.index(substance))
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    ax = ax or plt.subplot(1, 1, 1, projection='3d')
    if not isinstance(ax, Axes3D):
        raise ValueError("Need Axes3D instance as axes object.")

    xtC = [rd.xcenters, tout, Cout]
    x_, t_, C_ = list(map(np.log10, xtC)) if log10 else xtC
    X, T = np.meshgrid(x_, t_)
    if 'cmap' not in plot_kwargs:
        plot_kwargs['cmap'] = cm.copper
    ax.plot_surface(X, T, C_[:, :, substance],
                    **plot_kwargs)

    fmtstr = "$log_{{10}}$({})" if log10 else "{}"
    ax.set_xlabel(fmtstr.format('x / m'))
    ax.set_ylabel(fmtstr.format('time / s'))
    ax.set_zlabel(fmtstr.format('C / M'))
    latex_names_None = rd.substance_latex_names in (None, [None]*rd.n)
    if rd.substance_names:
        if not latex_names_None:
            name = '$' + rd.substance_latex_names[substance] + '$'
        else:
            name = rd.substance_names[substance]
        ax.set_title('['+name+'] vs. t and x')

    return ax


def plot_fields(rd, ax=None, indices=None, rho=None):
    """
    Convenience function to inspect fields in of
    ReactionDiffusion instance

    Parameters
    ----------
    rd: ReactionDiffusion
    ax: Axes instance or dict
        if ax is a dict it is used as keyword arguments passed to
        matplotlib.pyplot.axes (default: None)
    indices: sequence of integers
        what field strengths sequences to plot
    rho: float (optional)
        density, with consistent unit. If passed doserate
        will be plotted instead.
    """
    ax = _init_axes(ax)
    indices = indices or range(len(rd.fields))
    factors = np.array(rd.fields)
    if rho is not None:
        factors = factors/rho
    x_edges = np.repeat(rd.x, 2)[1:]
    for i in indices:
        y_edges = np.pad(np.repeat(factors[i, :], 2), (0, 1), 'constant')
        ax.plot(x_edges, y_edges, label=i)
    return ax


def plot_solver_linear_error(
        integration, Cref=0, ax=None, x=None, ti=slice(None), bi=0, si=0,
        plot_kwargs=None, fill_between_kwargs=None, scale_err=1.0, fill=True,
        **kwargs):
    """
    Parameters
    ----------
    integration: chemreac.integrate.Integration
        result from integration.
    Cref: array or float
        analytic solution to compare with
    ax: Axes instance or dict
        if ax is a dict it is used as key word arguments passed to
        matplotlib.pyplot.axes (default: None)
    x: array
        (optional) x-values, when None it is deduced to be
        either t or x (when ti or bi are slices repecitvely)
        (default: None)
    ti: slice
        time indices
    bi: slice
        bin indices
    si: integer
        specie index
    plot_kwargs: dict
        keyword arguments passed to matplotlib.pyplot.plot (default: None)
    fill_between_kwargs: dict
        keyword arguments passed to matplotlib.pyplot.fill_between
        (default: None)
    scale_err: float
        value with which errors are scaled. (default: 1.0)
    fill: bool
        whether or not to fill error span
    \\*\\*kwargs
        common keyword arguments of plot_kwargs and fill_between_kwargs,
        e.g. 'color', (default: None).

    See Also
    --------
    plot_solver_linear_excess_error
    """
    ax = _init_axes(ax)
    Cerr = integration.Cout - to_unitless(Cref, get_derived_unit(
        integration.rd.unit_registry, 'concentration'))
    if x is None:
        if isinstance(ti, slice):
            x = integration.tout[ti]
        elif isinstance(bi, slice):
            x = integration.rd.xcenters[bi]
        else:
            raise NotImplementedError("Failed to deduce x-axis.")

    plot_kwargs = plot_kwargs or {}
    set_dict_defaults_inplace(plot_kwargs, kwargs)
    plt.plot(np.asarray(x), np.asarray(scale_err*Cerr[ti, bi, si]),
             **plot_kwargs)

    if fill:
        le_l, le_u = solver_linear_error_from_integration(
            integration, ti=ti, bi=bi, si=si)
        Cerr_u = le_u - Cref[ti, bi, si]
        Cerr_l = le_l - Cref[ti, bi, si]
        fill_between_kwargs = fill_between_kwargs or {}
        set_dict_defaults_inplace(fill_between_kwargs, {'alpha': 0.2}, kwargs)
        plt.fill_between(
            np.asarray(x),
            np.asarray(scale_err*Cerr_l),
            np.asarray(scale_err*Cerr_u), **fill_between_kwargs)
    return ax


def plot_solver_linear_excess_error(integration, Cref, ax=None, x=None,
                                    ti=slice(None), bi=0, si=0, **kwargs):
    """
    Plots the excess error commited by the intergrator, divided by the span
    of the tolerances (atol + rtol*|y_i|).

    Parameters
    ----------
    integration: chemreac.integrate.Integration
        result from integration.
    Cref: array or float
        analytic solution to compare with
    ax: Axes instance or dict
        if ax is a dict it is used as \\*\\*kwargs passed to
        matplotlib.pyplot.axes (default: None)
    x: array
        (optional) x-values, when None it is deduced to be
        either t or x (when ti or bi are slices repecitvely)
        (default: None)
    ti: slice
        time indices
    bi: slice
        bin indices
    si: integer
        specie index
    plot_kwargs: dict
        keyword arguments passed to matplotlib.pyplot.plot (default: None)
    fill_between_kwargs: dict
        keyword arguments passed to matplotlib.pyplot.fill_between
        (default: None)
    scale_err: float
        value with which errors are scaled. (default: 1.0)
    fill: bool
        whether or not to fill error span
    \\*\\*kwargs:
        common keyword arguments of plot_kwargs and fill_between_kwargs,
        e.g. 'color', (default: None).

    See Also
    --------
    plot_solver_linear_error
    """
    if x is None:
        if isinstance(ti, slice):
            x = integration.tout[ti]
        elif isinstance(bi, slice):
            x = integration.rd.xcenters[bi]
        else:
            raise NotImplementedError("Failed to deduce x-axis.")
    ax = _init_axes(ax)
    le_l, le_u = solver_linear_error_from_integration(integration, ti, bi, si)
    Eexcess_l = Cref[ti, bi, si] - le_l  # Excessive if negative
    Eexcess_u = Cref[ti, bi, si] - le_u  # Excessive if positive
    u_conc = get_derived_unit(integration.rd.unit_registry, 'concentration')
    Eexcess_l[np.argwhere(Eexcess_l >= 0)] = 0 * u_conc
    Eexcess_u[np.argwhere(Eexcess_u <= 0)] = 0 * u_conc
    fused = np.concatenate((Eexcess_l[..., np.newaxis],
                            Eexcess_u[..., np.newaxis]), axis=-1)
    indices = np.argmax(abs(fused), axis=-1)
    Eexcess = fused[np.indices(indices.shape), indices][0, ...]
    le_span = le_u - le_l
    ax.plot(np.asarray(integration.tout),
            np.asarray(Eexcess/le_span),
            **kwargs)
    return ax
