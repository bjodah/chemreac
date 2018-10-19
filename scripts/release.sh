#!/bin/bash -xeu
# Usage:
#
#    $ ./scripts/release.sh v1.2.3 GITHUB_USER upstream
#

if [[ $1 != v* ]]; then
    echo "Argument does not start with 'v'"
    exit 1
fi
VERSION=${1#v}
GITHUB_USER=$2
REMOTE=$3
find . -type f -iname "*.pyc" -exec rm {} +
find . -type f -iname "*.o" -exec rm {} +
find . -type f -iname "*.so" -exec rm {} +
find . -type d -name "__pycache__" -exec rmdir {} +
find . -type f -iname ".coverage.*" -exec rm {} +
./scripts/check_clean_repo_on_master.sh
cd $(dirname $0)/..
# PKG will be name of the directory one level up containing "__init__.py" 
PKG=$(find . -maxdepth 2 -name __init__.py -print0 | xargs -0 -n1 dirname | xargs basename)
PKG_UPPER=$(echo $PKG | tr '[:lower:]' '[:upper:]')
${PYTHON:-python3} setup.py build_ext -i
./scripts/run_tests.sh
env ${PKG_UPPER}_RELEASE_VERSION=v$VERSION ${PYTHON:-python3} setup.py sdist
if [[ -e ./scripts/generate_docs.sh ]]; then
    env ${PKG_UPPER}_RELEASE_VERSION=v$VERSION ./scripts/generate_docs.sh
fi
# All went well, add a tag and push it.
git tag -a v$VERSION -m v$VERSION
git push $REMOTE
git push $REMOTE --tags
twine upload dist/${PKG}-$VERSION.tar.gz

set +x
echo ""
echo "    You may now create a new github release at with the tag \"v$VERSION\" and name"
echo "    it \"${PKG}-${VERSION}\". Here is a link:"
echo "        https://github.com/${GITHUB_USER}/${PKG}/releases/new "
echo "    name the release \"${PKG}-${VERSION}\", and don't foreget to manually attach the file:"
echo "        $(openssl sha256 $(pwd)/dist/${PKG}-${VERSION}.tar.gz)"
echo "    Then run:"
echo ""
echo "        $ ./scripts/post_release.sh v${VERSION} <myserver.examples.com> ${GITHUB_USER} ${REMOTE}"
echo ""
