#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
import pprint
import re
import shutil
import subprocess
import sys
import warnings

from setuptools import setup, Extension
try:
    import cython
except ImportError:
    _HAVE_CYTHON = False
else:
    _HAVE_CYTHON = True
    assert cython  # silence pep8

pkg_name = 'chemreac'
url = 'https://github.com/chemreac/' + pkg_name
license = 'BSD'


def _path_under_setup(*args):
    return os.path.join('.', *args)


release_py_path = _path_under_setup(pkg_name, '_release.py')
config_py_path = _path_under_setup(pkg_name, '_config.py')
env = None  # silence pyflakes, 'env' is actually set on the next line
exec(open(config_py_path).read())
for k, v in list(env.items()):
    env[k] = os.environ.get('%s_%s' % (pkg_name.upper(), k), v)

_version_env_var = '%s_RELEASE_VERSION' % pkg_name.upper()
RELEASE_VERSION = os.environ.get(_version_env_var, '')


_version_env_var = '%s_RELEASE_VERSION' % pkg_name.upper()
RELEASE_VERSION = os.environ.get(_version_env_var, '')

if len(RELEASE_VERSION) > 1:
    if RELEASE_VERSION[0] != 'v':
        raise ValueError("$%s does not start with 'v'" % _version_env_var)
    TAGGED_RELEASE = True
    __version__ = RELEASE_VERSION[1:]
else:  # set `__version__` from _release.py:
    TAGGED_RELEASE = False
    exec(open(release_py_path).read())
    if __version__.endswith('git'):
        try:
            _git_version = subprocess.check_output(
                ['git', 'describe', '--dirty']).rstrip().decode('utf-8')
        except subprocess.CalledProcessError:
            warnings.warn("A git-archive is being installed - version information incomplete.")
        else:
            if 'develop' not in sys.argv:
                warnings.warn("Using git to derive version: dev-branches may compete.")
                _ver_tmplt = r'\1.post\2' if os.environ.get('CONDA_BUILD', '0') == '1' else r'\1.post\2+\3'
                __version__ = re.sub(r'v([0-9.]+)-(\d+)-(\S+)', _ver_tmplt, _git_version)  # .dev < '' < .post

_WITH_DEBUG = env['WITH_DEBUG'] == '1'
_WITH_OPENMP = env['WITH_OPENMP'] == '1'
_WITH_DATA_DUMPING = env['WITH_DATA_DUMPING'] == '1'

# Source distributions contain rendered sources
_common_requires = ['numpy>=1.16.6', 'block_diag_ilu>=0.5.1', 'pycvodes>=0.13.1', 'finitediff>=0.6.3']
install_requires = _common_requires + ['chempy>=0.7.11', 'quantities>=0.12.1']
package_include = os.path.join(pkg_name, 'include')


_src = {ext: _path_under_setup(pkg_name, '_%s.%s' % (pkg_name, ext)) for ext in "cpp pyx".split()}
if os.path.exists(_src["pyx"]):
    # Possible that a new release of Python needs a re-rendered Cython source,
    # or that we want to include possible bug-fix to Cython.
    USE_CYTHON = True
    if os.path.exists(_src['cpp']):
        os.unlink(_src['cpp'])  # ensure c++ source is re-generated.
else:
    USE_CYTHON = False

ext_modules = []


if len(sys.argv) > 1 and '--help' not in sys.argv[1:] and sys.argv[1] not in (
        '--help-commands', 'egg_info', 'clean', '--version'):
    import numpy as np
    import finitediff as fd
    import pycvodes as pc
    from pycvodes._libs import get_libs as pc_get_libs
    import block_diag_ilu as bdi

    rendered_path = 'src/chemreac.cpp'
    template_path = rendered_path + '.mako'

    if os.path.exists(template_path):
        from mako.template import Template
        from mako.exceptions import text_error_template
        subsd = {'WITH_OPENMP': _WITH_OPENMP}
        try:
            rendered = Template(open(template_path, 'rt').read()).render(**subsd)
        except:
            sys.stderr.write(text_error_template().render_unicode())
            raise
        else:
            open(rendered_path, 'wt').write(rendered)

    sources = [_src["pyx" if USE_CYTHON else "cpp"]]
    ext_modules.append(Extension('{0}._{0}'.format(pkg_name), sources))

    if USE_CYTHON:
        from Cython.Build import cythonize
        cythonize_inc_path = [
            package_include,
            pc.get_include(),
            os.path.join('external', 'anyode', 'cython_def')
        ]
        print(f"{cythonize_inc_path=}")
        ext_modules = cythonize(ext_modules, include_path=cythonize_inc_path)
        if not os.path.exists(os.path.join('external', 'anyode', 'cython_def', 'anyode.pxd')):
            raise FileNotFoundError("No anyode.pxd?")
    ext_modules[0].include_dirs += [
        np.get_include(), fd.get_include(), bdi.get_include(),
        pc.get_include(), package_include, os.path.join('external', 'anyode', 'include')
    ]
    ext_modules[0].sources = [rendered_path] + ext_modules[0].sources
    ext_modules[0].language = 'c++'
    ext_modules[0].extra_compile_args = ['-std=c++20'] + (['-fopenmp'] if _WITH_OPENMP else [])
    ext_modules[0].define_macros += (
        ([('CHEMREAC_WITH_DEBUG', None)] if _WITH_DEBUG else []) +
        ([('CHEMREAC_WITH_DATA_DUMPING', None)] if _WITH_DATA_DUMPING else []) +
        ([('BLOCK_DIAG_ILU_WITH_OPENMP', None)] if os.environ.get('BLOCK_DIAG_ILU_WITH_OPENMP', '') == '1' else [])
    )
    ext_modules[0].libraries += [l for l in (pc_get_libs().split(',') + os.environ.get(
        'CHEMREAC_LAPACK', "lapack,blas").split(",")) if l != ""] + ['m']

modules = [
    pkg_name+'.util',
]

tests = [
    pkg_name+'.tests',
    pkg_name+'.util.tests',
]

classifiers = [
    "Development Status :: 3 - Alpha",
    'License :: OSI Approved :: BSD License',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Mathematics',
]

with io.open(_path_under_setup(pkg_name, '__init__.py'), 'rt', encoding='utf-8') as f:
    short_description = f.read().split('"""')[1].split('\n')[1]
if not 10 < len(short_description) < 255:
    warnings.warn("Short description from __init__.py proably not read correctly.")
long_description = io.open(_path_under_setup('README.rst'),
                           encoding='utf-8').read()
if not len(long_description) > 100:
    warnings.warn("Long description from README.rst probably not read correctly.")
_author, _author_email = io.open(_path_under_setup('AUTHORS'), 'rt', encoding='utf-8').readline().split('<')

setup_kwargs = dict(
    name=pkg_name,
    version=__version__,
    description=short_description,
    long_description=long_description,
    author=_author.strip(),
    author_email=_author_email.split('>')[0].strip(),
    url=url,
    license=license,
    keywords=["chemical kinetics", "Smoluchowski equation",
              "advection-diffusion-reaction"],
    packages=[pkg_name] + modules + tests,
    include_package_data=True,
    package_data={
        'chemreac.tests': ['*.json', '*.txt']
    },
    ext_modules=ext_modules,
    classifiers=classifiers,
    setup_requires = ['cython==3.0.12'], #<-- https://github.com/cython/cython/issues/6981
    install_requires=install_requires,
    extras_require={'all': [
        'argh', 'pytest', 'scipy>=1.17.1', 'matplotlib', 'mpld3',
        'sym>=0.3.4', 'sympy>=1.1.1,!=1.2', 'pyodeint>=0.10.10', 'pygslodeiv2>=0.9.8', 'batemaneq>=0.2.2',
        'sphinx', 'sphinx_rtd_theme', 'numpydoc', 'pyodesys>=0.14.6'
    ]},
    python_requires='>=3.6',
)

if __name__ == '__main__':
    try:
        if TAGGED_RELEASE:
            # Same commit should generate different sdist files
            # depending on tagged version (see RELEASE_VERSION)
            # this will ensure source distributions contain the correct version
            shutil.move(release_py_path, release_py_path+'__temp__')
            open(release_py_path, 'wt').write(
                "__version__ = '{}'\n".format(__version__))
        shutil.move(config_py_path, config_py_path+'__temp__')
        with open(config_py_path, 'wt') as fh:
            fh.write("env = {}\n".format(pprint.pformat(env)))
        setup(**setup_kwargs)
    finally:
        if TAGGED_RELEASE:
            shutil.move(release_py_path+'__temp__',
                        release_py_path)
        shutil.move(config_py_path+'__temp__',
                    config_py_path)
