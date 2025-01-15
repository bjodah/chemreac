#!/bin/bash
set -euxo pipefail
source /opt-3/cpython-v3.*-apt-deb/bin/activate
INSTALL_PIP_FLAGS="--cache-dir ./ci-cache/pip-cache --upgrade"  # --user
for pypkg in pyodeint pygslodeiv2 pycompilation pycodeexport batemaneq finitediff block_diag_ilu pycvodes pyodesys chempy; do
    case $pypkg in
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
             pypkg_fqn="git+https://github.com/bjodah/pyodesys@bdf2#egg=pycvodes"
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
            py3 -m pip install $INSTALL_PIP_FLAGS $pypkg_fqn
    else
        py3 -m pip install $INSTALL_PIP_FLAGS $pypkg_fqn
    fi
    py3 -m pytest -k "not pool_discontinuity_approx" --pyargs $pypkg
done

