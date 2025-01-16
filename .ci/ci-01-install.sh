#!/bin/bash
set -euxo pipefail
source /opt-3/cpython-v3.*-apt-deb/bin/activate
INSTALL_PIP_FLAGS="--cache-dir ./ci-cache/pip-cache --upgrade"  # --user
for pypkg in pyodeint pygslodeiv2 pycompilation pycodeexport batemaneq finitediff block_diag_ilu pycvodes pyodesys chempy; do
    case $pypkg in
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
             pypkg_fqn="git+https://github.com/bjodah/pycompilation@use-importlib-rather-than-imp#egg=pycompilation"
             ;;
         pycodeexport)
             pypkg_fqn="git+https://github.com/bjodah/pycodeexport@qulify-extension-name-and-new-ci#egg=pycodeexport"
             ;;
         pycvodes)
             pypkg_fqn="git+https://github.com/bjodah/pycvodes@may21#egg=pycvodes"
             ;;
         pyodesys)
             pypkg_fqn="git+https://github.com/bjodah/pyodesys@bdf2#egg=pyodesys"
             ;;
         chempy)
             pypkg_fqn="git+https://github.com/bjodah/chempy@nov20#egg=chempy"
             ;;
         *)
             pypkg_fqn=$pypkg
             ;;
    esac
    if [[ $pypkg == "pycvodes" ]]; then
        env \
            CFLAGS="-isystem $SUNDBASE/include $CFLAGS" \
            LDFLAGS="-Wl,--disable-new-dtags -Wl,-rpath,$SUNDBASE/lib -L$SUNDBASE/lib $LDFLAGS" \
            python -m pip install $INSTALL_PIP_FLAGS $pypkg_fqn
    else
        python -m pip install $INSTALL_PIP_FLAGS $pypkg_fqn
    fi
    python -m pytest -k "not pool_discontinuity_approx" --pyargs $pypkg
done

