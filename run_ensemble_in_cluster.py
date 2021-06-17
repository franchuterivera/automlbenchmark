import argparse
from random import randrange
import collections
import mmap
import os
import re
import time
import subprocess
import socket
import logging
import glob
import json
import typing
import sys
import yaml

import paramiko
from scp import SCPClient


from shutil import copyfile

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# Logger Setup
logger = logging.getLogger('manager')
ch = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

def generate_run_file(
    framework: str,
    benchmark: str,
    constraint: str,
    task: str,
    fold: int,
    run_dir: str
) -> str:
    """Generates a bash script for the sbatch command

    Args:
        framework (str): the framework to run
        benchmark (str): in which benchmark to run it
        constraint (str): under which constrains to run the framework
        task (str): in which dataset to run it and associated tasks
        fold (int): which of the 10 folds to run
        run_dir (str): in which directory to run the job

    Returns:
        str: the path to the bash file to run
    """

    run_file = f"{run_dir}/scripts/{framework}_{benchmark}_{constraint}_{task}_{fold}.sh"

    command = f"""#!/bin/bash
#Setup the run
echo "Running on HOSTNAME=$HOSTNAME with name $SLURM_JOB_NAME"
export PATH=/usr/local/kislurm/singularity-3.5/bin/:$PATH
source {ENVIRONMENT_PATH}
cd {AUTOMLBENCHMARK}
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then export TMPDIR=/tmp/{framework}_{benchmark}_{constraint}_{task}_{fold}_$SLURM_JOB_ID; else export TMPDIR=/tmp/{framework}_{benchmark}_{constraint}_{task}_{fold}$SLURM_ARRAY_JOB_ID'_'$SLURM_ARRAY_TASK_ID; fi
echo TMPDIR=$TMPDIR
export XDG_CACHE_HOME=$TMPDIR
echo XDG_CACHE_HOME=$XDG_CACHE_HOME
mkdir -p $TMPDIR
export SINGULARITY_BINDPATH="$TMPDIR:/tmp"
export VIRTUAL_MEMORY_AVAILABLE=4469755084

# Config the run
export framework={framework}
export benchmark={benchmark}
export constraint={constraint}
export task={task}
export fold={fold}
# Sleep a random number of seconds after init to be safe of run
sleep {np.random.randint(low=0,high=10)}
echo 'python runbenchmark.py {framework} {benchmark} {constraint} --task {task} --fold {fold} -m singularity --session {framework}_{benchmark}_{constraint}_{task}_{fold} -o {run_dir}/{framework}_{benchmark}_{constraint}_{task}_{fold} -u {run_dir}'
python runbenchmark.py {framework} {benchmark} {constraint} --task {task} --fold {fold} -m singularity --session {framework}_{benchmark}_{constraint}_{task}_{fold} -o {run_dir}/{framework}_{benchmark}_{constraint}_{task}_{fold} -u {run_dir}
echo "Deleting temporal folder $TMPDIR"
rm -rf $TMPDIR
echo 'Finished the run'
"""

    with open(os.path.join('/tmp', os.path.basename(run_file)), 'w') as f:
        f.write(command)
    remote_put(os.path.join('/tmp', os.path.basename(run_file)), run_file)
    return run_file

def launch_run(
    run_files: typing.List[str],
    args: typing.Any,
    run_dir: str,
):
    """Sends a job to sbatcht

    Args:
        run_files (typing.List[str]): the batch shell file
        args (typing.Any): namespace with the input script arguments
        run_dir (str): in which directory to run
    """

    # make sure we work with lists
    not_launched_runs = []
    for task in run_files:
        if not check_if_running(task):
            not_launched_runs.append(task)
    if not not_launched_runs:
        return
    run_files = not_launched_runs

    # Run options
    extra = ''
    if args.partition == 'bosch_cpu-cascadelake':
        extra += ' --bosch'

    # For array
    max_hours=8
    if 'array' in args.run_mode:
        raise NotImplementedError(f"Have to fix this for remote running ")
        job_list_file = to_array_run(run_files, args.memory, args.cores, run_dir)
        name, ext = os.path.splitext(os.path.basename(job_list_file))
        max_run = min(int(args.max_active_runs), len(run_files))
        extra += f" -p {args.partition} --array=0-{len(run_files)-1}%{max_run} --job-name {name}"
        _launch_sbatch_run(extra, job_list_file)

    elif args.run_mode == 'single':
        timestamp = time.strftime("%Y.%m.%d-%H.%M.%S")
        for task in run_files:
            if are_resource_available_to_run(partition=args.partition, min_cpu_free = 10) or True:
                name, ext = os.path.splitext(os.path.basename(task))
                this_extra = extra + f" -p {args.partition} -t 0{max_hours}:00:00 --mem {args.memory} -c {args.cores} --job-name {name} -o {os.path.join(run_dir, 'logs', name + '_'+ timestamp + '.out')}"
                _launch_sbatch_run(this_extra, task)
            else:
                logger.warn(f"Skip {task} as there are no more free resources... try again later!")
            # Wait 2 sec to update running job
            time.sleep(2)
def _launch_sbatch_run(options: str, script: str) -> int:
    """
    Launches a subprocess with sbatch command
    """
    command = "sbatch {} {}".format(
        options,
        script
    )
    logger.info(f"-I-: Running command={command}")
    #returned_value = subprocess.run(
    #    command,
    #    shell=True,
    #    stdout=subprocess.PIPE
    #).stdout.decode('utf-8')
    returned_value = remote_run(command)[0]

    success = re.compile('Submitted batch job (\d+)').match(returned_value)  # noqa: W605
    if success:
        return int(success[1])
    else:
        raise Exception("Could not launch job for script={script} returned_value={returned_value}")

