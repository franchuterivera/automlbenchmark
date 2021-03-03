#!/usr/bin/env bash
HERE=$(dirname "$0")
AMLB_DIR="$1"
VERSION=${2:-"stable"}
REPO=${3:-"https://github.com/franchuterivera/Auto-PyTorch.git"}
PKG=${4:-"Auto-PyTorch"}
if [[ "$VERSION" == "latest" ]]; then
    VERSION="master"
fi
# Use a version that fixes reproducibility issues
VERSION='reproducibility'

# creating local venv
. ${HERE}/../shared/setup.sh ${HERE}



# creating local venv
. $HERE/../shared/setup.sh $HERE

TARGET_DIR="$HERE/lib/Auto-PyTorch"

git clone https://github.com/franchuterivera/Auto-PyTorch.git $TARGET_DIR
cd $TARGET_DIR
echo before
git checkout reproducibility
echo checkout
cd $HERE

PIP install --no-cache-dir -U -r $HERE/requirements.txt
#PIP uninstall autopytorch
echo requirements
PIP install --no-cache-dir -e $TARGET_DIR
echo FINISHED
PIP freeze
