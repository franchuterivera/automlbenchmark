#!/usr/bin/env bash
HERE=$(dirname "$0")
AMLB_DIR="$1"
VERSION=${2:-"latest"}
REPO=${3:-"https://github.com/franchuterivera/auto-sklearn.git"}
PKG=${4:-"auto-sklearn"}
if [[ "$VERSION" == "latest" ]]; then
    VERSION="master"
fi

# creating local venv
. $HERE/../shared/setup.sh $HERE

if [[ -x "$(command -v apt-get)" ]]; then
    SUDO apt-get install -y build-essential swig
fi

PIP install 'scipy>=0.14.1,<1.7.0'
PIP install --no-cache-dir liac-arff packaging numpy
TARGET_DIR="${HERE}/lib/${PKG}"
rm -Rf ${TARGET_DIR}
git clone ${REPO} ${TARGET_DIR}
cd ${TARGET_DIR}
git checkout cvavgintensifier_levels_meta_ES_rebase3_towering_metalearning
cd ${HERE}
PIP install -r ${TARGET_DIR}/requirements.txt
PIP install -e ${TARGET_DIR}
PIP uninstall --yes smac
git clone https://github.com/franchuterivera/SMAC3.git
cd SMAC3
git checkout use_max_budget_seen_ensembleintensifier_ES_dynamic_medianl2fastfix_towering
cd ${HERE}
PIP install -e SMAC3
