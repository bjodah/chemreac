#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

from distutils.core import setup, Command

pkg_name = 'chemreac'
# read __version__ and __doc__ attributes:
exec(open(pkg_name+'/release.py').read())
try:
    major, minor, micro = map(int, __version__.split('.'))
except ValueError:
    IS_RELEASE=False
else:
    IS_RELEASE=True

with open(pkg_name+'/__init__.py') as f:
    long_description = f.read().split('"""')[1]

DEBUG = True if os.environ.get('USE_DEBUG', False) else False
USE_OPENMP = True if os.environ.get('USE_OPENMP', False) else False
LLAPACK = os.environ.get('LLAPACK', 'lapack')

CONDA_BUILD = os.environ.get('CONDA_BUILD', '0') == '1'
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
ON_DRONE = os.environ.get('DRONE', 'false') == 'true'
ON_TRAVIS = os.environ.get('TRAVIS', 'flse') == 'true'

if CONDA_BUILD:
    open('__conda_version__.txt', 'w').write(__version__)

if ON_DRONE or ON_TRAVIS:
    # 'fast' implies march=native which fails on current version of docker.
    options = ['pic', 'warn']
else:
    options = ['pic', 'warn', 'fast']

cmdclass_ = {}

if on_rtd or '--help' in sys.argv[1:] or sys.argv[1] in (
        '--help-commands', 'egg_info', 'clean', '--version'):
    # Enbale pip to probe setup.py before all requirements are installed
    ext_modules_ = []
else:
    import pickle
    from pycodeexport.dist import pce_build_ext, PCEExtension
    import numpy as np
    cmdclass_['build_ext'] = pce_build_ext
    subsd = {'USE_OPENMP': USE_OPENMP}
    sources = [
        'src/chemreac_template.cpp',
        'src/finitediff/finitediff/fornberg.f90',
        'src/finitediff/finitediff/c_fornberg.f90',
        'chemreac/_chemreac.pyx',
    ]

    ext_modules_ = [
        PCEExtension(
            "chemreac._chemreac",
            sources=sources,
            template_regexps=[
                (r'^(\w+)_template.(\w+)$', r'\1.\2', subsd),
            ],
            pycompilation_compile_kwargs={
                'per_file_kwargs': {
                    'src/chemreac.cpp': {
                        'std': 'c++0x',
                        # 'fast' doesn't work on drone.io
                        'options': options +
                        (['openmp'] if USE_OPENMP else []),
                        'define': ['DEBUG'] +
                        (['DEBUG'] if DEBUG else []),
                    },
                    'src/chemreac_sundials.cpp': {
                        'std': 'c++0x',
                        'options': options
                    },
                },
                'options': options,
            },
            pycompilation_link_kwargs={
                'options': (['openmp'] if USE_OPENMP else []),
                'std': 'c++0x',
                'libraries': ['sundials_cvode', LLAPACK, 'sundials_nvecserial'],
            },
            include_dirs=['src/', 'src/finitediff/finitediff/',
                          np.get_include()],
            logger=True,
        )
    ]

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

setup(
    name=pkg_name,
    version=__version__,
    description='Python extension for reaction diffusion.',
    author='Björn Dahlgren',
    author_email='bjodah@DELETEMEgmail.com',
    url='https://github.com/bjodah/' + pkg_name,
    packages=[pkg_name] + modules + tests,
    cmdclass=cmdclass_,
    ext_modules=ext_modules_,
)
