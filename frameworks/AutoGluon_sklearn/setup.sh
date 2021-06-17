#!/usr/bin/env bash
HERE=$(dirname "$0")
AMLB_DIR="$1"
VERSION=${2:-"latest"}
REPO=${3:-"https://github.com/franchuterivera/autogluon.git"}
PKG=${4:-"autogluon"}
if [[ "$VERSION" == "latest" ]]; then
    VERSION="master"
fi

# creating local venv
. ${HERE}/../shared/setup.sh ${HERE}
#if [[ -x "$(command -v apt-get)" ]]; then
#    SUDO apt-get install -y libomp-dev
if [[ -x "$(command -v brew)" ]]; then
    brew install libomp
fi

cat ${HERE}/requirements.txt | sed '/^$/d' | while read -r i; do PIP install "$i"; done

#if [[ "$VERSION" =~ ^[0-9] ]]; then
#    PIP install --no-cache-dir ${PKG}==${VERSION}
#else
##    PIP install --no-cache-dir -e git+${REPO}@${VERSION}#egg={PKG}
#    TARGET_DIR="${HERE}/lib/${PKG}"
#    rm -Rf ${TARGET_DIR}
#    git clone --depth 1 --single-branch --branch ${VERSION} --recurse-submodules ${REPO} ${TARGET_DIR}
#    PIP install -e ${TARGET_DIR}
#fi

PIP install --upgrade setuptools
TARGET_DIR="${HERE}/lib/${PKG}"
rm -Rf ${TARGET_DIR}
git clone ${REPO} ${TARGET_DIR}
cd ${TARGET_DIR}
git checkout psutil
PIP install -e ${TARGET_DIR}/core
PIP install -e ${TARGET_DIR}/tabular
PIP install -e ${TARGET_DIR}/mxnet
PIP install -e ${TARGET_DIR}/extra
PIP install -e ${TARGET_DIR}/text
PIP install -e ${TARGET_DIR}/vision
PIP install -e ${TARGET_DIR}/autogluon
cd ${HERE}

