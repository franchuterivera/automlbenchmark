#!/bin/bash
# Write to a temporal directory which we will copy over for debug
export TMPDIR=/tmp/agl_porto-seguro-safe-driver-prediction
mkdir $TMPDIR
export VIRTUAL_MEMORY_AVAILABLE=180000000000
source /home/riverav/work/venv_autogluon/bin/activate
python --version
pip freeze
cd /home/riverav/AUTOML_BENCHMARK/automlbenchmark_fork
python run_kaggle.py  -c 4 --runtime 3600 -m 180G -t porto-seguro-safe-driver-prediction -f AutoGluon
cd $TMPDIR
find . -name '*pkl' -ls -delete
find . -name '*model' -ls -delete
find . -name '*npy' -ls -delete
find . -name '*ensemble' -ls -delete
cp -r $TMPDIR /home/riverav/AUTOML_BENCHMARK/automlbenchmark_fork/debug_autogluon_porto-seguro-safe-driver-prediction