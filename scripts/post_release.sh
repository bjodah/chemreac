#!/bin/bash
# Usage:
#
#    $ ANFILTE_CHANNELS="defaults conda-forge" ./scripts/post_release.sh v1.2.3 myserver githubuser upstream
#
VERSION=${1#v}
SERVER=$2
GITHUBUSER=$3
REMOTE=$4
PKG=$(find . -maxdepth 2 -name __init__.py -print0 | xargs -0 -n1 dirname | xargs basename)
PKG_UPPER=$(echo $PKG | tr '[:lower:]' '[:upper:]')
SDIST_FILE=dist/${PKG}-$VERSION.tar.gz
if [[ ! -f "$SDIST_FILE" ]]; then
    >&2 echo "Nonexistent file $SDIST_FILE"
    exit 1
fi
SHA256=$(openssl sha256 "$SDIST_FILE" | cut -f2 -d' ')
if [[ -d "dist/conda-recipe-$VERSION" ]]; then
    rm -r "dist/conda-recipe-$VERSION"
fi
cp -r conda-recipe/ dist/conda-recipe-$VERSION
sed -i -E \
    -e "s/git_url:(.+)/fn: \{\{ name \}\}-\{\{ version \}\}.tar.gz\n  url: https:\/\/pypi.io\/packages\/source\/\{\{ name\[0\] \}\}\/\{\{ name \}\}\/\{\{ name \}\}-\{\{ version \}\}.tar.gz\n  sha256: \"$SHA256\"/" \
    -e "/set version/d" \
    -e "/set number/d" \
    -e "/if number/d" \
    -e "s/.*endif*./\{% set version = \"$VERSION\" /" \
    -e "/cython/d" \
    dist/conda-recipe-$VERSION/meta.yaml

ssh $PKG@$SERVER 'mkdir -p ~/public_html/conda-recipes'
scp -r dist/conda-recipe-$VERSION/ $PKG@$SERVER:~/public_html/conda-recipes/
scp "$SDIST_FILE" "$PKG@$SERVER:~/public_html/releases/"
./scripts/update-gh-pages.sh v$VERSION $REMOTE

# ssh $PKG@$SERVER 'mkdir -p ~/public_html/conda-packages'
# for CONDA_PY in 27 35 36; do
#     anfilte-build . dist/conda-recipe-$VERSION dist/ --python ${CONDA_PY}
#     scp dist/linux-64/${PKG}-${VERSION}-py${CONDA_PY}*.bz2 $PKG@$SERVER:~/public_html/conda-packages/
# done
