#!/bin/bash
#
# Test script for the Continuous Integration server drone.io

#LOGFILE=`pwd`/'ci.log'

# Define the version of Python that should be tested
PY_VERSION="2.7"

scripts/aptget.ubuntu.12.04LTS.sh

if [[ "$PY_VERSION" == "2.7" ]]; then
    # This saves us some downloading for this version
    wget http://repo.continuum.io/miniconda/Miniconda-3.5.2-Linux-x86_64.sh -O miniconda.sh;
else
    wget http://repo.continuum.io/miniconda/Miniconda3-3.5.2-Linux-x86_64.sh -O miniconda.sh;
fi
chmod +x miniconda.sh
./miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
hash -r
conda config --set always_yes yes --set changeps1 no
conda update conda
conda info -a
conda create -n test-env python=${PY_VERSION} numpy=1.8.1 scipy=0.14 argh=0.23.1 cython=0.19.1 mako=0.5.0 quantities=0.10.1 pytest=2.5.2
source activate test-env
pip install https://github.com/bjodah/pycompilation/archive/v0.2.21.tar.gz
pip install https://github.com/sympy/sympy/archive/master.zip
python setup.py install
py.test
