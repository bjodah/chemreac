#!/bin/bash
set -euxo pipefail
source /opt-3/cpython-v3.*-apt-deb/bin/activate
python3 -m pip install --user --cache-dir ./ci-cache/pip-cache --upgrade finitediff "block_diag_ilu>=0.4.3" "pyodeint>=0.10.4" "pygslodeiv2>=0.9.3"  "batemaneq>=0.2.2"
python3 -c "import block_diag_ilu"
env \
    CFLAGS="-isystem $SUNDBASE/include $CFLAGS" \
    LDFLAGS="-Wl,--disable-new-dtags -Wl,-rpath,$SUNDBASE/lib -L$SUNDBASE/lib $LDFLAGS" \
python3 -m pip install --user --cache-dir ./ci-cache/pip-cache --upgrade "pycvodes>=0.14.1"
python3 -m pip install --user --cache-dir ./ci-cache/pip-cache --upgrade "pyodesys>=0.13.1"
python3 -m pip install --user --cache-dir ./ci-cache/pip-cache --upgrade "chempy>=0.7.10"
ls external/anyode
