#!/bin/bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TMPDIR=/tmp/sasei_walmart-recruiting-trip-type-classification
mkdir $TMPDIR
export VIRTUAL_MEMORY_AVAILABLE=180000000000
source /home/riverav/work/venv_autosklearn_EI/bin/activate
python --version
pip freeze
cd /home/riverav/AUTOML_BENCHMARK/automlbenchmark_fork
python run_kaggle.py  --iterative True -c 4 --runtime 28800 -m 180G -t walmart-recruiting-trip-type-classification -f AutoSklearnEnsembleIntensification
cd $TMPDIR
find . -name '*pkl' -ls -delete
find . -name '*model' -ls -delete
find . -name '*npy' -ls -delete
find . -name '*ensemble' -ls -delete
cp -r $TMPDIR /home/riverav/AUTOML_BENCHMARK/automlbenchmark_fork/
