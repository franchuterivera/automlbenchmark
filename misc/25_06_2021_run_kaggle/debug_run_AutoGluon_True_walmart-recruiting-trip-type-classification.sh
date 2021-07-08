#!/bin/bash
export TMPDIR=/tmp/ag_walmart-recruiting-trip-type-classification
mkdir $TMPDIR
export VIRTUAL_MEMORY_AVAILABLE=180000000000
source /home/riverav/work/venv_autogluon/bin/activate
python --version
pip freeze
cd /home/riverav/AUTOML_BENCHMARK/automlbenchmark_fork
python run_kaggle.py  --iterative True -c 4 --runtime 3600 --half_for_test true -m 180G -t walmart-recruiting-trip-type-classification -f AutoGluon
cd $TMPDIR
find . -name '*pkl' -ls -delete
find . -name '*model' -ls -delete
find . -name '*npy' -ls -delete
find . -name '*ensemble' -ls -delete
cp -r $TMPDIR /home/riverav/AUTOML_BENCHMARK/automlbenchmark_fork/
