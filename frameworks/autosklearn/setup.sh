#!/usr/bin/env bash
HERE=$(dirname "$0")
AMLB_DIR="$1"
VERSION=${2:-"latest"}
REPO=${3:-"https://github.com/automl/auto-sklearn.git"}
PKG=${4:-"auto-sklearn"}
if [[ "$VERSION" == "latest" ]]; then
    VERSION="master"
fi

# creating local venv
. $HERE/../shared/setup.sh $HERE

if [[ -x "$(command -v apt-get)" ]]; then
    SUDO apt-get install -y build-essential swig
fi

PIP install --no-cache-dir liac-arff packaging numpy
PIP install --no-cache-dir setuptools
PIP install --no-cache-dir numpy>=1.9.0
PIP install --no-cache-dir scipy>=0.14.1
PIP install --no-cache-dir scikit-learn>=0.22.0,<0.23
PIP install --no-cache-dir lockfile
PIP install --no-cache-dir joblib
PIP install --no-cache-dir pyyaml
PIP install --no-cache-dir pandas<1.0
PIP install --no-cache-dir liac-arff
PIP install --no-cache-dir ConfigSpace>=0.4.14,<0.5
PIP install --no-cache-dir pynisher>=0.4.2
PIP install --no-cache-dir pyrfr>=0.7,<0.9
PIP install --no-cache-dir git+https://github.com/automl/smac3@add_config_id
if [[ "$VERSION" =~ ^[0-9] ]]; then
    PIP install --no-cache-dir ${PKG}==${VERSION}
else
#    PIP install --no-cache-dir -e git+${REPO}@${VERSION}#egg=${PKG}
    TARGET_DIR="${HERE}/lib/${PKG}"
    rm -Rf ${TARGET_DIR}
    git clone --depth 1 --single-branch --branch ${VERSION} --recurse-submodules ${REPO} ${TARGET_DIR}
    PIP install -e ${TARGET_DIR}
fi

