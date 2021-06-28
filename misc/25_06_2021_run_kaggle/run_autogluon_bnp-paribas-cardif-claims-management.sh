#!/bin/bash
# Write to a temporal directory which we will copy over for debug
export TMPDIR=$TMPDIR/debug_autogluon_bnp-paribas-cardif-claims-management
mkdir $TMPDIR
export VIRTUAL_MEMORY_AVAILABLE=34000000000
source /home/riverav/work/venv_autogluon/bin/activate
python --version
pip freeze
cd /home/riverav/AUTOML_BENCHMARK/automlbenchmark_fork
python run_kaggle.py  -c 8 --runtime 3600 -m 32G -t bnp-paribas-cardif-claims-management -f AutoGluon
cd $TMPDIR
find . -name '*pkl' -ls -delete
find . -name '*model' -ls -delete
find . -name '*npy' -ls -delete
find . -name '*ensemble' -ls -delete
cp -r $TMPDIR /home/riverav/AUTOML_BENCHMARK/automlbenchmark_fork/debug_autogluon_bnp-paribas-cardif-claims-management
