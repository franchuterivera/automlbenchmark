#!/usr/bin/env bash
HERE=$(dirname "$0")
. $HERE/../shared/setup.sh
TARGET_DIR="$HERE/lib/Auto-PyTorch"
if [[ ! -e "$TARGET_DIR" ]]; then
    git clone https://github.com/automl/Auto-PyTorch $TARGET_DIR
fi
PIP install --no-cache-dir -r $TARGET_DIR/requirements.txt
PIP install --no-cache-dir -e $TARGET_DIR
