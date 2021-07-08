#!/bin/bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TMPDIR=/tmp/s2as_santander-customer-satisfaction
mkdir $TMPDIR
export VIRTUAL_MEMORY_AVAILABLE=180000000000
source /home/riverav/work/venv_autosklearn/bin/activate
python --version
pip freeze
cd /home/riverav/AUTOML_BENCHMARK/automlbenchmark_fork
python run_kaggle.py  --iterative False -c 2 --runtime 14400 -m 180G -t santander-customer-satisfaction -f AutoSklearn
cd $TMPDIR
find . -name '*pkl' -ls -delete
find . -name '*model' -ls -delete
find . -name '*npy' -ls -delete
find . -name '*ensemble' -ls -delete
cp -r $TMPDIR /home/riverav/AUTOML_BENCHMARK/automlbenchmark_fork/
