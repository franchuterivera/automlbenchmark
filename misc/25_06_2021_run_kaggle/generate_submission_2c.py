mapshowrt = {
    'AutoGluon': 's2ag',
    'AutoSklearn': 's2as',
    'AutoSklearnAutoGluonStrategy': 's2asag',
    'AutoSklearnEnsembleIntensification': 's2asei',
}
mapenv = {
    'AutoGluon': 'venv_autogluon',
    'AutoSklearn': 'venv_autosklearn',
    'AutoSklearnAutoGluonStrategy': 'venv_autosklearn_EI',
    'AutoSklearnEnsembleIntensification': 'venv_autosklearn_EI',
}
scripts = []
for iterative in [False]:
    for framework in ['AutoGluon', 'AutoSklearn', 'AutoSklearnAutoGluonStrategy', 'AutoSklearnEnsembleIntensification']:
        for kaggle in ['porto-seguro-safe-driver-prediction',
                       'santander-customer-satisfaction',
                       'santander-customer-transaction-prediction',
                       'ieee-fraud-detection',
                       'otto-group-product-classification-challenge',
                       'bnp-paribas-cardif-claims-management',
                       'walmart-recruiting-trip-type-classification',
                       ]:
            name = f"submission2c_run_{framework}_{iterative}_{kaggle}.sh"
            scripts.append(name)
            with open(name, 'w') as f:
                f.write("#!/bin/bash\n")
                if 'sklea' in framework.lower():
                    f.write("export OMP_NUM_THREADS=1\n")
                    f.write("export OPENBLAS_NUM_THREADS=1\n")
                    f.write("export MKL_NUM_THREADS=1\n")

                f.write(f"export TMPDIR=/tmp/{mapshowrt[framework]}_{kaggle}\n")
                f.write(f"mkdir $TMPDIR\n")
                f.write(f"export VIRTUAL_MEMORY_AVAILABLE=180000000000\n")
                f.write(f"source /home/riverav/work/{mapenv[framework]}/bin/activate\n")
                f.write(f"python --version\n")
                f.write(f"pip freeze\n")
                f.write(f"cd /home/riverav/AUTOML_BENCHMARK/automlbenchmark_fork\n")
                f.write(f"python run_kaggle.py  --iterative {iterative} -c 2 --runtime 14400 -m 180G -t {kaggle} -f {framework}\n")
                f.write(f"cd $TMPDIR\n")
                f.write(f"find . -name '*pkl' -ls -delete\n")
                f.write(f"find . -name '*model' -ls -delete\n")
                f.write(f"find . -name '*npy' -ls -delete\n")
                f.write(f"find . -name '*ensemble' -ls -delete\n")
                f.write(f"cp -r $TMPDIR /home/riverav/AUTOML_BENCHMARK/automlbenchmark_fork/\n")

with open('launch.submission2c.csv', 'w') as launch:
    for item in scripts:
        launch.write(f"sbatch --bosch -p bosch_cpu-cascadelake -t 10:00:00 --mem 180G -c 2 --job-name {item.replace('.sh', '')} -o /home/riverav/AUTOML_BENCHMARK/automlbenchmark_fork/{item.replace('.sh', '')} /home/riverav/AUTOML_BENCHMARK/automlbenchmark_fork/misc/25_06_2021_run_kaggle/{item}\n")
