# Combatting overfitting in Auto-sklearnand other AutoML systems

This repository list the steps to reproduce the results from the thesis titled: "Combatting overfitting in Auto-sklearnand other AutoML systems".

The main repository is a modified version of the [AutoMLBenchmark](https://github.com/openml/automlbenchmark). We structure this document based on the thesis chapter and sections, mainly to highlight how the tables/plots were created, when possible. At the end we provide more details on relevant topics like directory structure and complementary software.


## Setup and environment used
I employed conda in my laptop, and stored the environment used via: `conda env export > environment.yml`.

This environment was originally created by:
1- Downloading and installing [Anaconda](https://docs.anaconda.com/anaconda/install/index.html).
2- Creating a python environment for at least 3.8 version `conda create --name automlbenchmarkpy38 python=3.*`.
3- Installing the automlbenchmark requirements `pip install -r requirements.txt`.
4- Installing the autogluon utils requirements `pip install -r autogluonbenchmarking/autogluon_utils/requirements.txt`

For all the experiments I used a cluster managed by SLURM (many commands listed below will highlight this). Our hardware consisted of an Intel(R) Xeon(R) Gold 6242 CPU cluster running at 2.80GHz, on a 64-bit architecture. For more details, please check the thesis available [here](TODO).

In the cluster, one also must have an environment. This environment was created as follows:
```
python3 -m venv ./venv
source venv/bin/activate
# Install automlbenchmark requirements
git clone https://github.com/openml/automlbenchmark.git
pip install -r automlbenchmark/requirements.txt

# In my case, the environment is located in
/home/riverav/work/venv/bin/activate
```

## Reproducing figures/tables 

### Approach

For the Approach chapter, we wanted to do an isolated study of cross validation and stacking using scikit learn. All the results regarding this part of the thesis are available in the folder [13_02_2021_isolatedrepetitions](misc/13_02_2021_isolatedrepetitions/) and used as [this](misc/13_02_2021_isolatedrepetitions/sklearn/feedback_frank_5_levels/repeated_stacking_over_time_sklearn_seed_ES_moreresources.py) main script. 

We ran the script using 5 different seeds per dataset and collected the results in a csv format, every time under the name of `all_data.csv`. Each of the below subsections will highlight the path to the relevant data file as well as the plotting command used. We exemplify how the command was run in the cluster:

```
sbatch  --bosch -p bosch_cpu-cascadelake -t 96:00:00 --mem 12G -c 1 --job-name RepeatedStackingSklearnfold10101 -o /home/riverav/AUTOML_BENCHMARK/repeats_local_jobs/sklearn_feedback_from_frank_folds/10101.log /home/riverav/AUTOML_BENCHMARK/repeats_local_jobs/sklearn_feedback_from_frank_folds/10101.sh
```

Where the actual shell command is:
```
#!/bin/bash
source /home/riverav/work/venv/bin/activate
cd /home/riverav/AUTOML_BENCHMARK/automlbenchmark_fork/misc/13_02_2021_isolatedrepetitions/sklearn/feedback_frank_folds
python repeated_stacking_over_time_sklearn_seed_ES_folds.py --openml_id 10101
```

Notice that we gave plenty of time and memory for our jobs in this section. Every subsection points to a `launch.sh` used to make a run, as well as the plotting command used.

#### Over-fit of AutoML frameworks
We extracted the [leaderboard](https://github.com/openml/automlbenchmark/blob/9e5b66d11a804b19595d602f3f3e7e54e544e201/frameworks/AutoGluon/exec.py#L87) from AutoGluon by using the command `leaderboard = predictor.leaderboard(**leaderboard_kwargs)` and stored it in [misc/07_03_2021_test_train_frameworks/leaderboard_autogluon.csv](misc/07_03_2021_test_train_frameworks/leaderboard_autogluon.csv). Similarly, we took the runhistory from Auto-Sklearn available in the `<tmp>/smac3/run_<seed>/runhistory.json` of every baseline run and stored in a modified format here [misc/07_03_2021_test_train_frameworks/leaderboard_autosklearn.csv](misc/07_03_2021_test_train_frameworks/leaderboard_autosklearn.csv). Afterwards we generated the plot as follows:
```
cd misc/07_03_2021_test_train_frameworks
python compile.py
```
#### Diminishing returns of stacking
The data available for this experiment is compiled in [misc/13_02_2021_isolatedrepetitions/sklearn/feedback_frank_5_levels/all_data.csv](misc/13_02_2021_isolatedrepetitions/sklearn/feedback_frank_5_levels/all_data.csv). To make the plot from this data:
```
cd misc/13_02_2021_isolatedrepetitions/sklearn/feedback_frank_5_levels/
python compile.py
```
#### Impact of the number of repetitions
The data available for this experiment is compiled in [misc/13_02_2021_isolatedrepetitions/sklearn/5_folds/all_data.csv](misc/13_02_2021_isolatedrepetitions/sklearn/5_folds/all_data.csv). To make the plot from this data:
```
cd misc/13_02_2021_isolatedrepetitions/sklearn/5_folds/
python compile_plot.py
```
#### Impact of the number of folds
The data available for this experiment is compiled in [misc/13_02_2021_isolatedrepetitions/sklearn/feedback_frank_folds/all_data.csv](misc/13_02_2021_isolatedrepetitions/sklearn/feedback_frank_folds/all_data.csv). To make the plot from this data:
```
cd misc/13_02_2021_isolatedrepetitions/sklearn/feedback_frank_folds
python compile_plot.py
```
#### Not all configurations benefit from stacking
The data available for this experiment is compiled in [misc/13_02_2021_isolatedrepetitions/sklearn/rfonly/all_data.csv](misc/13_02_2021_isolatedrepetitions/sklearn/rfonly/all_data.csv). To make the plot from this data:
```
cd misc/13_02_2021_isolatedrepetitions/sklearn/rfonly
python compile_plot.py
```
### Experiments
We use singularity as the SLURM cluster does not support docker. To interact with the cluster we use the script [manager.py](manager.py) which creates a singularity image for a given framework, run it, and collects the results from the cluster. We will exemplify the workflow we follow using autosklearn.

The framework is defined in [resources/frameworks.yaml](resources/frameworks.yaml), where the params keyword is of special importance as it customizes the frameworks. 

To create a singularity image for autosklearn and run the first fold of the small benchmark we ran:

```
python manager.py --tag seed0 -f autosklearn_memefficient -b small --partition bosch_cpu-cascadelake --total_fold 0 --cores 8 --memory 32G --runtime 3600 --python_version 3.8 --run_mode single
```

This generates the sif image `frameworks/autosklearn_memefficient_metalearning/autosklearn_memefficient_metalearning_large_memefficient_meta_stable.sif` and start runs in the cluster that last around 1 hour. The script tells you where the runs are being exercised, for examples `/home/riverav/AUTOML_BENCHMARK/autosklearn/memefficient/2021.06.22-19.19.17`. Then we can collect the results as follows:

```
python manager.py --tag seed0 -f autosklearn_memefficient -b small --partition bosch_cpu-cascadelake --total_fold 0 --cores 8 --memory 32G --runtime 3600 --python_version 3.8 --run_dir /home/riverav/AUTOML_BENCHMARK/autosklearn/memefficient/2021.06.22-19.19.17 --collect_overfit true
```

If you are familiar with the automlbenchmark, you can check an intermediate file called `generate_sif.sh` which is the script responsible of generating the image.

The next subsections provide the path to where the results are located as well as the protocol file used to achieve these results.

Our best method is defined within [frameworks/autosklearnEnsembleIntensification_metaon/](frameworks/autosklearnEnsembleIntensification_metaon/). The [setup.sh](frameworks/autosklearnEnsembleIntensification_metaon/setup.sh) indicates how to create an environment to make autosklearn with ensemble intensification work. The main repository for our approach is `https://github.com/franchuterivera/auto-sklearn/tree/cvavgintensifier_individual_rebase3_towering_largedatasets_singlerepo`, as can be seen in the aforementioned `setup.sh`.

#### AutoMLBenchmark and ablation results
All the results for the automlbenchmark comparison were created using the protocol [misc/18_06_2021_finalbaselines/18_06_2021_finalbaselines.protocol](misc/18_06_2021_finalbaselines/18_06_2021_finalbaselines.protocol). The results are grouped by seed in `misc/18_06_2021_finalbaselines/seed<>`.

#### Kaggle

AutoGluon provided a mechanism to repeat their results [here](https://github.com/Innixma/autogluon-benchmarking). Nevertheless, the API of the framework has changed making it no longer usable for the newer versions of AutoGluon. Apart from that, it assumes one has access to AWS.

I cloned the repository and made updates as needed. The new repository is locally available under [autogluonbenchmarking/](autogluonbenchmarking/). My objective was to use the new API available in the AutoMLBenchmark under [frameworks/AutoGluon_020/exec.py](frameworks/AutoGluon_020/exec.py) with stacking enabled. Same for Auto-sklearn. To this end, I created a wrapper script under the name of [run_kaggle.py](run_kaggle.py) which fundamentally:

* Interacts with kaggle using the autogluon_utils package. This means data downloading and AutoGluon's preprocessing.
* Then it the autogluon-benchmarking task is translated to a Dataset (pandas for AutoGluon, numpy/feat_type for AutoSklearn) and a config file that states the run metric, the cores, memory, etc.
* The `run()` method of the `exec.py` of each framework returns a result object with probabilities and predicitons.
* The probabilities go through the post-processing from AutoGluon and then a submission is made.

To obtain a sumision file I used [misc/25_06_2021_run_kaggle/launch.submission.csv](misc/25_06_2021_run_kaggle/launch.submission.csv) in the cluster. This creates a csv file which can be then used with `python run_kaggle --submission` on your laptop (The meta cluster does not have internet connection!), or manually uploaded to the repository.

The good thing about this approach is the singularity image corresponding to the automlbenchmark can be reused for this task.
#### Auto-sklearnperformance with and without the final ensemble
All the results for the automlbenchmark comparison were created using the protocol [misc/23_01_2021_defaultmetricisolatedensemble/23_01_2021_defaultmetricisolatedensemble.seed1.ES.protocol](misc/23_01_2021_defaultmetricisolatedensemble/23_01_2021_defaultmetricisolatedensemble.seed1.ES.protocol). The results are grouped by seed in `misc/(23_01_2021_defaultmetricisolatedensemble/seed<>`.

## Custom Code and justification

### AutoGluon with memory overwrite
For AutoGluon, I had to create a fork of the official repository [here](https://github.com/franchuterivera/autogluon/tree/psutil_0.2.0) as the frameworks relies on psutil to asses the available virtual memory. SLURM does not control this, and autogluon ended up assuming that the run has 180G rather than 32G and kept crashing. I overwrote any call to psutil virtual memory check to return a fixed memory limit of 32G when needed, via an environment variable.
