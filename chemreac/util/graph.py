# -*- coding: utf-8 -*-

import os
import subprocess
import shutil
import tempfile


"""
Convenince functions for representing reaction systems as graphs.
"""


def rsys2dot(rsys, substances=None, tex=False, rprefix='r', rref0=1,
             nodeparams='[label={} shape=diamond]'):
    """
    Returns list of lines of DOT (graph description language)
    formated graph.

    Parameters
    ==========
    rsys: ReactionSystem
    substances: sequence of Substance instances
    tex: bool (default False)
        If set True, output will be LaTeX formated
    (Substance need to have latex_name attribute set)
    rprefix: string
        Reaction enumeration prefix, default: r
    rref0: integer
        Reaction enumeration inital counter value, default: 1
    nodeparams: string
        DOT formated param list, default: [label={} shape=diamond]

    Returns
    =======
    list of lines of DOT representation of the graph representation.

    """
    lines = ['digraph ' + str(rsys.name) + '{\n']
    ind = '  '  # indentation

    def add_vertex(sn, num, reac):
        snum = str(num) if num > 1 else ''
        name = getattr(substances[sn], 'latex_name' if tex else 'name')
        lines.append(ind + '"{}" -> "{}" [label ="{}"];\n'.format(
            *((name, rid, snum) if reac else (rid, name, snum))
        ))

    for ri, rxn in enumerate(rsys.rxns):
        rid = rprefix + str(ri+rref0)
        lines.append(ind + '{')
        lines.append(ind*2 + 'node ' + nodeparams.format(rid))
        lines.append(ind*2 + rid)
        lines.append(ind + '}\n')
        for sn, num in rxn.reactants.items():
            if num == 0:
                continue
            add_vertex(sn, num, True)
        for sn, num in rxn.products.items():
            if num == 0:
                continue
            add_vertex(sn, num, False)
    lines.append('}\n')
    return lines


def rsys2graph(rsys, substances, fname, output_dir=None, prog=None, save=False,
               **kwargs):
    """
    Convenience function to call `rsys2dot` and write output to file
    and render the graph

    Parameters
    ----------
    rsys: ReactionSystem
    substances: sequence of Substance instances
    outpath: path to graph to be rendered
    prog: command to render DOT file (default: dot)
    **kwargs: parameters to pass along to `rsys2dot`

    Exapmles
    --------

    >>> rsys2graph(rsys, sbstncs, '/tmp/out.png')  # doctest: +SKIP
    """
    lines = rsys2dot(rsys, substances, **kwargs)
    created_tempdir = False
    try:
        if output_dir is None:
            output_dir = tempfile.mkdtemp()
            created_tempdir = True
        basename, ext = os.path.splitext(os.path.basename(fname))
        outpath = os.path.join(output_dir, fname)
        dotpath = os.path.join(output_dir, basename + '.dot')
        with open(dotpath, 'wt') as ofh:
            ofh.writelines(lines)
        if ext == '.tex':
            cmds = [prog or 'dot2tex']
        else:
            cmds = [prog or 'dot', '-T'+outpath.split('.')[-1]]
        p = subprocess.Popen(cmds + [dotpath, '-o', outpath])
        retcode = p.wait()
        if retcode:
            fmtstr = "{}\n returned with exit status {}"
            raise RuntimeError(fmtstr.format(' '.join(cmds), retcode))
        return outpath
    finally:
        if save is True or save == 'True':
            pass
        else:
            if save is False or save == 'False':
                if created_tempdir:
                    shutil.rmtree(output_dir)
            else:
                # interpret save as path to copy pdf to.
                shutil.copy(outpath, save)
