#!/usr/bin/env bash
HERE=$(dirname "$0")
AMLB_DIR="$1"
VERSION=${2:-"latest"}
REPO=${3:-"https://github.com/automl/Auto-PyTorch.git"}
PKG=${4:-"autoPyTorch"}
if [[ "$VERSION" == "latest" ]]; then
    VERSION="master"
fi

# creating local venv
. $HERE/../shared/setup.sh $HERE

if [[ -x "$(command -v apt-get)" ]]; then
    SUDO apt-get install -y build-essential swig
    SUDO apt-get install ffmpeg libsm6 libxext6  -y
fi

PIP install --no-cache-dir liac-arff packaging numpy
TARGET_DIR="${HERE}/lib/${PKG}"
rm -Rf ${TARGET_DIR}
git clone ${REPO} ${TARGET_DIR}
cd ${TARGET_DIR}
git checkout refactor_development_regularization_cocktails
cd ${HERE}
PIP install -r ${TARGET_DIR}/requirements.txt
PIP install -e ${TARGET_DIR}

PIP uninstall --yes smac
git clone https://github.com/franchuterivera/SMAC3.git
cd SMAC3
git checkout  debug_msg
cd ${HERE}
PIP install -e SMAC3

PIP uninstall --yes pynisher
git clone https://github.com/franchuterivera/pynisher.git
cd pynisher
git checkout  debug_msg
cd ${HERE}
PIP install -e pynisher
