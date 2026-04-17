#!/bin/bash
set -euxo pipefail
source /opt-3/cpython-v3.*-apt-deb/bin/activate
INSTALL_PIP_FLAGS="--cache-dir ./ci-cache/pip-cache --upgrade"  # --user

export PYODESYS_CVODE_FLAGS="-isystem $SUNDBASE/include ${CFLAGS:-}"
export PYODESYS_CVODE_LDFLAGS="-Wl,--disable-new-dtags -Wl,-rpath,$SUNDBASE/lib -L$SUNDBASE/lib ${LDFLAGS:-}"

for pypkg in pyodeint pygslodeiv2 pycompilation pycodeexport batemaneq finitediff block_diag_ilu pycvodes pykinsol sym pyodesys chempy; do
    case $pypkg in
        sym)
            pypkg_fqn="git+https://github.com/bjodah/sym@jun21#egg=sym"
            ;;
        pyodeint)
            pypkg_fqn="git+https://github.com/bjodah/pyodeint@sep21#egg=pyodeint"
            ;;
        pygslodeiv2)
            pypkg_fqn="git+https://github.com/bjodah/pygslodeiv2@cython-except-plus#egg=pygslodeiv2"
            ;;
         block_diag_ilu)
             pypkg_fqn="git+https://github.com/bjodah/block_diag_ilu"
             ;;
         pycompilation)
             pypkg_fqn="git+https://github.com/bjodah/pycompilation@master#egg=pycompilation"
             ;;
         pycodeexport)
             pypkg_fqn="git+https://github.com/bjodah/pycodeexport@master#egg=pycodeexport"
             ;;
         pycvodes)
             pypkg_fqn="git+https://github.com/bjodah/pycvodes@may21#egg=pycvodes"
             ;;
         pyodesys)
             pypkg_fqn="git+https://github.com/bjodah/pyodesys@bdf2#egg=pyodesys"
             ;;
         pykinsol)
             pypkg_fqn="git+https://github.com/bjodah/pykinsol@jan25#egg=pykinsol"
             ;;
         chempy)
             pypkg_fqn="git+https://github.com/bjodah/chempy@nov20#egg=chempy"
             ;;
         *)
             pypkg_fqn=$pypkg
             ;;
    esac
    if [[ $pypkg == "pycvodes" || $pypkg == "pykinsol" ]]; then
        env \
            CFLAGS="$PYODESYS_CVODE_FLAGS" \
            LDFLAGS="$PYODESYS_CVODE_LDFLAGS" \
            python -m pip install $INSTALL_PIP_FLAGS $pypkg_fqn
    else
        python -m pip install $INSTALL_PIP_FLAGS $pypkg_fqn
    fi
    python -m pytest -k "not pool_discontinuity_approx" --pyargs $pypkg
done

