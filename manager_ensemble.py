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


############################################################################
#                                GLOBALS
############################################################################
# The enviornment to run to benchmark. It is basically following the steps
# git clone https://github.com/openml/automlbenchmark.git
# cd automlbenchmark
# python3 -m venv ./venv
# source venv/bin/activate
# pip3 install -r requirements.txt
ENVIRONMENT_PATH = '/home/riverav/work/venv/bin/activate'

# The base path where we expect all runs to reside
BASE_PATH = '/home/riverav/AUTOML_BENCHMARK/'

# The remote location of the benchmark
AUTOMLBENCHMARK = '/home/riverav/AUTOML_BENCHMARK/automlbenchmark_fork'
#assert os.access(BASE_PATH, os.W_OK), f"Cannot access the base path {BASE_PATH}"

# A memory mapping from SLURM to automlbenchmakr
MEMORY = {'12G': 12288, '32G': 32768, '8G': 4096}

# Stablish a nested connection to kisba1
USER = 'riverav'
VM = paramiko.SSHClient()
# this connection is using the private key of the system
# if this fails to you, set this up properly
VM.load_system_host_keys()
VM.set_missing_host_key_policy(paramiko.AutoAddPolicy())
VM.connect(hostname='132.230.166.39', username=USER, look_for_keys=False)
vmtransport = VM.get_transport()
INTERNAL = '10.5.166.221'
dest_addr = (INTERNAL, 22)  # kisbat
local_addr = ('132.230.166.39', 22)  # aadlogin
vmchannel = vmtransport.open_channel("direct-tcpip", dest_addr, local_addr)

# ssh is effectively the nested global object to talk to
SSH = paramiko.SSHClient()
SSH.load_system_host_keys()
SSH.set_missing_host_key_policy(paramiko.AutoAddPolicy())
SSH.connect(INTERNAL, username=USER, sock=vmchannel)

############################################################################
#                               FUNCTIONS
############################################################################
def remote_exists(directory: str) -> bool:
    """Checks if a directory exist on remote host

    Args:
        directory: The remote repository

    Returns:
        bool: whether or not the directory exists
    """
    logger.debug(f"[remote_exists] {directory}")
    stdin, stdout, stderr = SSH.exec_command(f"ls -d {directory}")
    stdout.channel.recv_exit_status()
    stderr.channel.recv_exit_status()
    if any(['cannot access' in a for a in stderr.readlines()]):
        return False
    else:
        return True


def remote_glob(pattern: str) -> typing.List[str]:
    logger.debug(f"[remote_glob] {pattern}")
    stdin, stdout, stderr = SSH.exec_command(f"ls -d {pattern}")
    stdout.channel.recv_exit_status()
    stderr.channel.recv_exit_status()
    errors = stderr.readlines()
    files = stdout.readlines()
    if any(['cannot access' in a for a in errors]):
        return []
    else:
        return [f.rstrip() for f in files]


def remote_makedirs(directory: str) -> bool:
    """Creates a nested directory

    Args:
        directory (str): the directory to create

    Returns:
        bool: True if the directory was created

    """
    logger.debug(f"[remote_makedirs] {directory}")
    stdin, stdout, stderr = SSH.exec_command(f"mkdir -p {directory}") #edited#
    return remote_exists(directory)


def remote_put(source: str, destination: str) -> bool:
    """
    Takes a local file from source and copies it to
    destination

    Args:
        source (str): the file to copy over to destination
        destination (str): The location of the file in remote

    Returns:
        bool: True if the new file exists in remote
    """
    logger.debug(f"[remote_put] {source}->{destination}")
    # First make the file available in addlogin
    scp = SCPClient(SSH.get_transport())
    scp.put(
        source,
        recursive=True,
        remote_path=destination,
    )
    return remote_exists(destination)


def remote_get(source: str) -> str:
    """
    Copies a file from remote to the /tmp directory for
    local processing

    Args:
        source (str): the file to copy from the remote

    Returns:
        str: the local full path of the file
    """
    logger.debug(f"[remote_get] {source}")
    destination = os.path.join('/tmp', os.path.basename(source))

    # First make the file available in addlogin
    scp = SCPClient(SSH.get_transport())
    scp.get(
        source,
        local_path=destination,
    )
    return destination


def remote_run(cmd: str) -> typing.List[str]:
    """
    Runs a command in the remote server and if
    successful, return the stdout line by line

    Args:
        cmd (str): string with the command to run

    Returs:
        List[str]: the stdout output

    Raises:
        Exception: if any error occurred
    """
    logger.debug(f"[remote_run] {cmd}")
    stdin, stdout, stderr = SSH.exec_command(cmd) #edited#
    stdout.channel.recv_exit_status()
    stderr.channel.recv_exit_status()
    errors = stderr.readlines()
    if any(errors):
        raise Exception(errors)
    returned_lines = stdout.readlines()
    logger.debug(f"[remote_run] {cmd}->{returned_lines}")
    return returned_lines


def create_run_dir_area(run_dir: typing.Optional[str], args: typing.Any
                       ) -> str:
    """Creates a run are if it doesn't previously exists
    The run area is special because it must contain:
        + config file to define which metric to optimize with
        + the constraint if using a special one
        + the framework when using a different version/tool

    Args:
        run_dir (str): where to create the run area
        args (typing.Any): the namespace with the arguments to the python script

    Returns:
        run_dir (str): the area in which to run
    """
    if run_dir is not None:
        if not remote_exists(run_dir):
            raise ValueError(f"The provided directory {run_dir} does not exists or is"
                             "not reachable by ssh"
                             )
        logger.info(f"Provided area {run_dir} exists. We will then not create a new one")
        return run_dir

    # TODO: make sure constraint given matches the constraint in which the run area was ran

    # No run dir provided, create one!
    version = f"{args.framework}_{args.seed}_es{args.ensemble_size}_B{args.bbc_cv_n_bootstrap}_N{args.bbc_cv_sample_size}"
    timestamp = time.strftime("%Y.%m.%d-%H.%M.%S")
    base_framework = "ENSEMBLE_ISOLATED"
    run_dir = os.path.join(
        BASE_PATH,
        base_framework,
        version,
        timestamp,
    )

    remote_makedirs(run_dir)

    remote_makedirs(os.path.join(run_dir, 'scripts'))
    remote_makedirs(os.path.join(run_dir, 'logs'))
    logger.info(f"Creating run directory: {run_dir}")

    return run_dir


def generate_run_file(
    framework: str,
    benchmark: str,
    constraint: str,
    task: str,
    fold: int,
    run_dir: str,
    args,
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

    run_file = f"{run_dir}/scripts/{framework}_{args.seed}_{args.bbc_cv_n_bootstrap}_{args.bbc_cv_sample_size}_{benchmark}_{constraint}_{task}_{fold}.sh"
    cmd = f"python test_strategies.py --strategy {framework} --task {task} --fold {fold} --output '{run_dir}/{framework}_{benchmark}_{constraint}_{task}_{fold}' --input_dir '{args.input_dir}' --ensemble_size {args.ensemble_size} --bbc_cv_sample_size {args.bbc_cv_sample_size} --bbc_cv_n_bootstrap {args.bbc_cv_n_bootstrap} --seed {args.seed}"

    command = f"""#!/bin/bash
#Setup the run
echo "Running on HOSTNAME=$HOSTNAME with name $SLURM_JOB_NAME"
export PATH=/usr/local/kislurm/singularity-3.5/bin/:$PATH
source {ENVIRONMENT_PATH}
cd {AUTOMLBENCHMARK}/auto-sklearn
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
echo "Running {cmd}"
{cmd}
echo "Deleting temporal folder $TMPDIR"
rm -rf $TMPDIR
echo 'Finished the run'
"""

    with open(os.path.join('/tmp', os.path.basename(run_file)), 'w') as f:
        f.write(command)
    remote_put(os.path.join('/tmp', os.path.basename(run_file)), run_file)
    return run_file


def get_task_from_benchmark(benchmarks: typing.List[str]) -> typing.Dict[str, typing.List[str]]:
    """Returns a dict with benchmark to task mapping

    Args:
        benchmakrs (typing.List[str]): List of benchmakrs to run

    Returns:
        typing.Dict[str, str]: mapping from benchmark to task
    """
    task_from_benchmark = {}  # type: typing.Dict[str, typing.List]

    for benchmark in benchmarks:
        task_from_benchmark[benchmark] = []
        filename = os.path.join('resources', 'benchmarks', f"{benchmark}.yaml")
        if not os.path.exists(filename):
            raise Exception(f"File {filename} not found!")
        with open(filename) as file:
            data = yaml.load(file, Loader=yaml.FullLoader)

        for task in data:
            if task['name'] in ['__dummy-task', '__defaults__']:
                continue
            task_from_benchmark[benchmark].append(task['name'])

    return task_from_benchmark


def score(df: pd.DataFrame, res_col: str = 'result') -> float:
    """
    Get the results as the automlbenchmark team through
    https://github.com/openml/automlbenchmark/blob/master/reports/report/results.py
    return row[res_col] if row[res_col] in [row.auc, row.acc] else -row[res_col]

    Args:
        df (pd.DataFrame): A frame containing the result of executing a job
        res_col (str): The column in the dataframe that contains the actual result
    """
    result = df['result'].iloc[0]
    auc = df['auc'].iloc[0] if 'auc' in df else None
    balac = df['balac'].iloc[0] if 'balac' in df else None
    acc = df['acc'].iloc[0] if 'acc' in df else None
    if result in [auc, acc, balac]:
        return result
    else:
        return -result


def norm_score(
    framework: str,
    benchmark: str,
    constraint: str,
    task: str,
    fold: int,
    score: float,
    run_dir: str,
) -> float:
    """
    Normalizes the result score between a constant predictor and a random forest

    Args:
        framework (str): the framework to run
        benchmark (str): in which benchmark to run it
        constraint (str): under which constrains to run the framework
        task (str): in which dataset to run it and associated tasks
        fold (int): which of the 10 folds to run
        score (float): the result of the framework/bechmakr/task/fold
        run_dir (str): from where to parse the results

    Returns:
        float: the normalized score
    """
    if score is None or not is_number(score):
        return score
    zero = get_results('constantpredictor', benchmark, constraint, task, fold, run_dir)
    if zero is None or not is_number(zero):
        logger.debug(f"No constantpredictor result for for benchmark={benchmark}")
        return score
    one = get_results('RandomForest', benchmark, constraint, task, fold, run_dir)
    if one is None or not is_number(one):
        logger.warn(f"No RandomForest result for for benchmark={benchmark}")
        return score
    return (score - zero) / (one - zero)


def get_results(
    framework: str,
    benchmark: str,
    constraint: str,
    task: str,
    fold: int,
    run_dir: str
) -> typing.Optional[float]:
    """
    Get the result of executing framework under a given set of constraints

    We rely on the existance of a result file, which contains a CSV with
    the results of the run

    Args:
        framework (str): the framework to run
        benchmark (str): in which benchmark to run it
        constraint (str): under which constrains to run the framework
        task (str): in which dataset to run it and associated tasks
        fold (int): which of the 10 folds to run
        score (float): the result of the framework/bechmakr/task/fold

    Returns:
        float: the un-normalized score
    """
    result_file = f"{run_dir}/{framework}_{benchmark}_{constraint}_{task}_{fold}/result.csv"

    if not remote_exists(result_file):
        return None

    result_file = remote_get(result_file)

    if os.path.getsize(result_file) < 5:
        logger.error(f"result_file={result_file} is empty")
        return None

    try:
        df = pd.read_csv(result_file)
    except Exception as e:
        print(f"result_file={result_file}")
        raise e
    df["fold"] = pd.to_numeric(df["fold"])
    df = df[(df['framework'] == framework) & (df['task'] == task) & (df['fold'] == fold)]

    # If no run return
    if df.empty:
        return None

    if df.shape[0] != 1:
        if not df[df.result.notnull()].empty:
            df = df[df.result.notnull()]
            df = df.iloc[[-1]]

    result = df['result'].iloc[-1]

    if result is None or pd.isnull(result):
        return df['info'].iloc[-1]

    return score(df)


def query_yes_no(question, default='no'):
    if default is None:
        prompt = " [y/n] "
    elif default == 'yes':
        prompt = " [Y/n] "
    elif default == 'no':
        prompt = " [y/N] "
    else:
        raise ValueError(f"Unknown setting '{default}' for default.")

    while True:
        try:
            resp = input(question + prompt).strip().lower()
            if default is not None and resp == '':
                return default == 'yes'
            else:
                return str2bool(resp)
        except ValueError:
            logger.critical("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")


def check_if_crashed(run_file) -> typing.Union[bool, str]:
    """Checks if a run crashed from the batch log file

    Args:
        run_file (str): the sbatch run file

    Return:
        str: Returns the cause of failure or false if no failure
    """
    name, ext = os.path.splitext(os.path.basename(run_file))
    logfile = glob.glob(os.path.join('logs', name + '*' + '.out'))
    if len(logfile) == 0:
        return False
    logfile.sort()
    logfile = logfile[0]
    if not os.path.exists(logfile) or os.path.getsize(logfile) < 10:
        return False

    causes = [
        'error: Exceeded job memory limit',
        'DUE TO TIME LIMIT',
        'MemoryError',
        'OutOfMemoryError',
        'ValueError: Cannot compare configs that were run on different instances-seeds-budgets',
    ]
    with open(logfile, 'rb', 0) as file, mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as s:
        for cause in causes:
            if s.find(cause.encode()) != -1:
                return cause
    return 'UNDEFINED CAUSE'


def check_if_running(run_file: str) -> bool:
    """
    Queries SLURM to check if a job is running or not

    Args:
        run_file (str): the script that might have been used to launch the run

    Returns:
        bool: Whether the job is running or not
    """
    name, ext = os.path.splitext(os.path.basename(run_file))

    # First check if there is a job with this name
    cmd = f"squeue --format=\"%.150j\" --noheader -u {USER}"
    #result = subprocess.run([
    #    'squeue',
    #    f"--format=\"%.50j\" --noheader -u {os.environ['USER']}"
    #], stdout=subprocess.PIPE).stdout.decode('utf-8')
    for i, line in enumerate(remote_run(cmd)):
        if name in line:
            return True

    # The check in the user job arrays
    #cmd = f"squeue --format=\"%.50i\" --noheader -u {user}"
    ##result = subprocess.run([
    ##    f"squeue --format=\"%.50i\" --noheader -u {os.environ['USER']}"
    ##], shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8')

    ##jobids = [jobid.split('_')[0] for jobid in result.splitlines() if "_" in jobid]
    #jobids = [jobid.split('_')[0] for jobid in remote_run(cmd) if "_" in jobid]
    #if jobids:
    #    jobids = list(set(jobids))
    #for i, array_job in enumerate(jobids):
    #    command = subprocess.run([
    #        f"scontrol show jobid -dd {array_job} | grep Command |  sort --unique",
    #    ], shell=True, stdout=subprocess.PIPE)
    #    command_out, filename = command.stdout.decode('utf-8').split('=')

    #    filename = filename.lstrip().rstrip()
    #    if not os.path.exists(filename):
    #        raise Exception(f"Could not find file {filename} from command={command_out} array_job={array_job }")
    #    with open(filename, 'r') as data_file:
    #        if name in data_file.read():
    #            return True

    return False


def get_node(partition):
    """
    Gets a free node to be able to run the run without HTTP errors
    PARTITION                AVAIL  TIMELIMIT  NODES  STATE NODELIST
meta_gpu-black              up   infinite      4  down* metagpu[2-4,8]
meta_gpu-black              up   infinite      1   drng metagpu9
meta_gpu-black              up   infinite      1  drain metagpu1
meta_gpu-black              up   infinite      1  alloc metagpu5
test_cpu-ivy                up    1:05:00      1    mix metaex01
ml_cpu-ivy                  up 4-00:00:00      1    mix metaex16
ml_cpu-ivy                  up 4-00:00:00      2   idle metaex[15,17]
cpu_ivy                     up    2:05:00      1    mix metaex16
cpu_ivy                     up    2:05:00     10  alloc metaex[18-27]
cpu_ivy                     up    2:05:00      7   idle metaex[15,17,28-32]
bosch_cpu-cascadelake       up   infinite     36  alloc kisexe[09-44]
allbosch_cpu-cascadelake    up    2:05:00      3    mix kisexe[03,06,08]
allbosch_cpu-cascadelake    up    2:05:00     41  alloc kisexe[01-02,04-05,07,09-44]
    """

    if partition == 'test_cpu-ivy':
        return 'metaex01'
    elif partition == 'ml_cpu-ivy':
        # Sleep for 5 seconds to make sure slurm updated
        time.sleep(5)

        result = subprocess.run(
            f"squeue --format=\"%.50R\" --noheader -u {os.environ['USER']}",
            shell=True,
            stdout=subprocess.PIPE
        )
        result = result.stdout.decode('utf-8')
        machines = {
            'metaex15': 0,
            'metaex16': 0,
            'metaex17': 0,
        }
        for i, line in enumerate(result.splitlines()):
            line = line.lstrip().strip()
            if line in machines:
                machines[line] = machines[line]+1
        return min(machines, key=machines.get)
    else:
        raise Exception(f"Unsupported partition={partition} provided")


def to_array_run(run_file, memory, cores, run_dir):

    name = f"{run_dir}/scripts/arrayjob_{len(run_file)}_{os.getpid()}_{randrange(100)}.sh"

    command = f"""#!/bin/bash
#SBATCH -o {run_dir}/logs/%x.%A.%a.out
#SBATCH -c {cores}
#SBATCH --mem {memory}

echo "Here's what we know from the SLURM environment"
echo SHELL=$SHELL
echo HOME=$HOME
echo CWD=$(pwd)
echo USER=$USER
echo JOB_ID=$JOB_ID
echo JOB_NAME=$JOB_NAME
echo HOSTNAME=$HOSTNAME
echo SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID
echo "Job started at: `date`"

"""
    for i, task in enumerate(run_file):
        command += f"\nif [ $SLURM_ARRAY_TASK_ID -eq {i} ]"
        command += "\n\tthen"
        command += f"\n\t\techo \"Chrashed {run_dir}/logs/$SLURM_JOB_NAME.$SLURM_ARRAY_JOB_ID.$SLURM_ARRAY_TASK_ID.out\" > {run_dir}/logs/{os.path.splitext(os.path.basename(task))[0]}.out"
        command += f"\n\t\tbash -x {task}"
        command += f"\n\t\t cp {run_dir}/logs/$SLURM_JOB_NAME.$SLURM_ARRAY_JOB_ID.$SLURM_ARRAY_TASK_ID.out {run_dir}/logs/{os.path.splitext(os.path.basename(task))[0]}.out"
        command += "\n\tfi\n"
    command += "\necho DONE at: `date`"

    with open(name, 'w') as f:
        f.write(command)
    return name


def are_resource_available_to_run(partition: str, min_cpu_free=128, max_total_runs=5):
    """
    Query slurm to make sure we can launch jobs

    Args:
        partition (str): which partition to launch and check
        min_cpu_free: only launch if cpu free
    """
    #result = subprocess.run(
    #    f"sfree",
    #    shell=True,
    #    stdout=subprocess.PIPE
    #)
    #result = result.stdout.decode('utf-8')

    # Also, account for a max total active runs
    cmd = f"squeue --format=\"%.50j\" --noheader -u {USER}"
    total_runs = remote_run(cmd)
    if len(total_runs) > max_total_runs:
        return False

    for i, line in enumerate(remote_run('sfree')):
        if partition in line and 'test' not in line:
            status  = re.split('\s+',
                               line.lstrip().rstrip())
            return int(status[3]) > min_cpu_free

    return False


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
            #time.sleep(2)
    elif args.run_mode == 'interactive':
        timestamp = time.strftime("%Y.%m.%d-%H.%M.%S")
        while len(run_files) > 0:
            if are_resource_available_to_run(partition=args.partition, min_cpu_free = 128):
                task = run_files.pop()
                name, ext = os.path.splitext(os.path.basename(task))
                this_extra = extra + f" -p {args.partition} -t 0{max_hours}:00:00 --mem {args.memory} -c {args.cores} --job-name {name} -o {os.path.join(run_dir, 'logs', name + '_'+ timestamp + '.out')}"
                _launch_sbatch_run(this_extra, task)
                # Wait 2 sec for the benchmark to be empty
                time.sleep(2)
            else:
                print(".", end="", flush=True)
                time.sleep(60)
    else:
        raise ValueError(f"Unsupported run_mode {args.run_mode}")


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


def str2bool(v):
    """
    Process any bool
    Taken from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def is_number(s: typing.Union[str, int, float]) -> bool:
    """
    Returns True if the provided string is a number. In any other case,
    it return False

    Args:
        s (str): the string to check

    Returns:
        bool: true if the string is number
    """
    if s is None:
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False


def get_job_status(
    framework: str,
    benchmarks: typing.List[str],
    tasks: typing.Dict[str, typing.List],
    folds: typing.List[int],
    constraint: str,
    run_dir: str,
    args,
) -> typing.Dict[str, typing.Any]:
    """
    Buils a dictionary of dictionaries with the status
    of tasks for a given framework

    Args:
        framework (str): framework that ran
        benchmarks (typingList[str]):  in what benchmakrs they ran
        tasks (typing.Dict[str, typing.List]): List of task ran
        folds (typing.List[int]): in which folds for CV
        constraint (str): under what contraint it was ran
        run_dir (str): the directory where the runs where launched

    Returns:
        Dict: dictionary containing the framework/task/benchmakr/fold info

    """

    # Get the task
    jobs = collections.defaultdict(dict)  # type: typing.Dict[str, typing.Dict[str, typing.Any]]
    total = 0

    jobs[framework] = dict()
    logger.debug('_'*40)
    logger.debug(f"\t\t{framework}")
    logger.debug('_'*40)

    for benchmark in benchmarks:
        jobs[framework][benchmark] = dict()
        logger.info('_'*40)
        logger.info(f"\tbenchmark={benchmark}")
        for task in tasks[benchmark]:
            jobs[framework][benchmark][task] = dict()
            logger.info('_'*40)
            logger.info(f"\t\t task={task}")
            for fold in folds:
                jobs[framework][benchmark][task][fold] = dict()

                # Check if the run files for this task exist
                # The run file is needed because it help us find out if the
                # job is running or not
                jobs[framework][benchmark][task][fold]['run_file'] = generate_run_file(
                    framework=framework,
                    benchmark=benchmark,
                    constraint=constraint,
                    task=task,
                    fold=fold,
                    run_dir=run_dir,
                    args=args,
                )
                if args.fast:
                    continue

                # Check if there are results already
                jobs[framework][benchmark][task][fold]['results'] = get_results(
                    framework=framework,
                    benchmark=benchmark,
                    constraint=constraint,
                    task=task,
                    fold=fold,
                    run_dir=run_dir,
                )

                # Normalize the score as in the paper
                jobs[framework][benchmark][task][fold]['norm_score'] = norm_score(
                    framework=framework,
                    benchmark=benchmark,
                    constraint=constraint,
                    task=task,
                    fold=fold,
                    score=jobs[framework][benchmark][task][fold]['results'],
                    run_dir=run_dir,
                )

                # Show status to see what is going on
                valid_result = is_number(jobs[framework][benchmark][task][fold]['results'])
                if valid_result:
                    status = 'Completed'
                elif check_if_running(jobs[framework][benchmark][task][fold]['run_file']):
                    status = 'Running'
                else:
                    status = 'N/A'
                    #crashed = check_if_crashed(jobs[framework][benchmark][task][fold]['run_file'])
                    #if crashed:
                    #    jobs[framework][benchmark][task][fold]['results'] = crashed
                    #    status = 'Chrashed'

                jobs[framework][benchmark][task][fold]['status'] = status

                logger.info(
                    f"\t\t\tFold:{fold} Status = {status} "
                    f"({jobs[framework][benchmark][task][fold]['results']})"
                )
                total = total + 1

    logger.debug('_'*40)
    logger.debug(f" A total of {total} runs checked")
    logger.debug('_'*40)

    return jobs


def launch(
    jobs: typing.Dict,
    args: typing.Any,
    run_dir: str
) -> None:
    """
    Takes a jobs dictionary and launches the remaining runs that have not yet been completed

    Args:
        jobs (typing.Dict): A handy dictionary with frameworks/benchmarck/task/fold info
        args (typing.Any): Namespace with the input argument of this script
        run_dir (str): Are on where to launch the jobs

    """

    # get a list of jobs to run
    run_files = []
    for framework, framework_dict in jobs.items():
        for benchmark, benchmark_dict in framework_dict.items():
            for task, task_dict in benchmark_dict.items():
                for fold, data in task_dict.items():
                    run_file = jobs[framework][benchmark][task][fold]['run_file']
                    if args.fast:
                        run_files.append(jobs[framework][benchmark][task][fold]['run_file'])
                        continue
                    results = jobs[framework][benchmark][task][fold]['results']
                    valid_result = is_number(results)

                    # Launch the run if it was not yet launched
                    if results is None:
                        if check_if_running(run_file):
                            status = 'Running'
                        else:
                            run_files.append(run_file)
                            status = 'Launched'
                    else:
                        # If the run failed, then ask the user to relaunch
                        if not valid_result:
                            if check_if_running(jobs[framework][benchmark][task][fold]['run_file']):
                                status = 'Running'
                            else:
                                status = 'Failed'
                                if query_yes_no(f"For framework={framework} benchmark={benchmark} constraint={constraint} task={task} fold={fold} obtained: {jobs[framework][benchmark][task][fold]['results']}. Do you want to relaunch this run?"):
                                    run_files.append(jobs[framework][benchmark][task][fold]['run_file'])
                                    status = 'Relaunched'

    # Launch the runs in sbatch
    if run_files:
        launch_run(
            run_files,
            args=args,
            run_dir=run_dir,
        )


def get_metric_from_run_area(run_dir):
    """
    Extracts the metric from the metadata in a run_dir
    Args:
        run_dir (str): Are on where to launch the jobs
    """
    config_file = remote_get(os.path.join(run_dir, 'config.yaml'))
    with open(config_file) as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    return data['benchmarks']['metrics']


def collect_overfit3(
    jobs: typing.Dict,
    args: typing.Any,
    run_dir: str
) -> pd.DataFrame:
    """
    Creates overfit dataset and print the total average. Just for autosklearn

    Args:
        jobs (typing.Dict): A handy dictionary with frameworks/benchmarck/task/fold info
        args (typing.Any): Namespace with the input argument of this script
        run_dir (str): Are on where to launch the jobs

    Returns:

    """

    # Collect a dataframe that will contain columns
    # tool Experiment metric train val test
    constraint = f"{args.runtime}s{args.cores}c{args.memory}"
    dataframe = []
    for framework, framework_dict in jobs.items():
        for benchmark, benchmark_dict in framework_dict.items():
            for task, task_dict in benchmark_dict.items():
                for fold, data in task_dict.items():
                    for overfit_file in remote_glob(os.path.join(
                        run_dir,
                        f"{framework}_{benchmark}_{constraint}_{task}_{fold}",
                        'debug',
                        'overfit.csv'
                    )):
                        overfit_file = remote_get(overfit_file)
                        frame = pd.read_csv(overfit_file, index_col=0)
                        for index, row in frame.iterrows():
                            row_dict = row.to_dict()

                            # We care about the best individual model and ensemble
                            if row_dict['model'] not in ['best_individual_model', 'best_ensemble_model']:
                                continue
                            model = row_dict['model']

                            train = row_dict['train']
                            val = row_dict['val']
                            test = row_dict['test']
                            if row_dict['model'] == 'best_individual_model':
                                key = 'best_ever_test_score_individual_model'
                            elif row_dict['model'] == 'best_ensemble_model':
                                key = 'best_ever_test_score_ensemble_model'
                            else:
                                raise NotImplementedError(row_dict['model'])
                            overfit = float(frame[frame['model'] == key]['test'].iloc[0] - row_dict['test'])
                            dataframe.append({
                                'tool': f"{framework}_es{args.ensemble_size}_B{args.bbc_cv_n_bootstrap}_N{args.bbc_cv_sample_size}",
                                'task': task,
                                'model': model,
                                'fold': fold,
                                'train':  train,
                                'val':  val,
                                'test':  test,
                                'overfit':  overfit,
                                'seed': args.seed,
                            })
    return pd.DataFrame(dataframe)


def collect_ensemble_history(
    jobs: typing.Dict,
    args: typing.Any,
    run_dir: str
) -> pd.DataFrame:
    """
    Collects the overfit files, integrating them per fold into a single framework file

    Args:
        jobs (typing.Dict): A handy dictionary with frameworks/benchmarck/task/fold info
        args (typing.Any): Namespace with the input argument of this script
        run_dir (str): Are on where to launch the jobs

    Returns:

    """

    # Collect a dataframe that will contain columns
    # tool Experiment metric train val test
    constraint = f"{args.runtime}s{args.cores}c{args.memory}"
    dataframe = []
    for framework, framework_dict in jobs.items():
        for benchmark, benchmark_dict in framework_dict.items():
            for task, task_dict in benchmark_dict.items():
                for fold, data in task_dict.items():
                    for overfit_file in remote_glob(os.path.join(
                        run_dir,
                        f"{framework}_{benchmark}_{constraint}_{task}_{fold}",
                        '*',  # The directory name the benchmark created
                        'debug',
                        task,
                        str(fold),
                        'ensemble_history.csv'
                    )):
                        overfit_file = remote_get(overfit_file)
                        frame = pd.read_csv(overfit_file, index_col=0)
                        # Tag with identifiers
                        frame['fold'] = fold
                        frame['tool'] = framework
                        frame['task'] = task

                        # Convert to relative time
                        dataframe.append(frame)
    return pd.concat(dataframe).reset_index(drop=True)


def collect_overhead(
    jobs: typing.Dict,
    args: typing.Any,
    run_dir: str
) -> pd.DataFrame:
    """
    Creates overfit dataset and print the total average. Just for autosklearn

    Args:
        jobs (typing.Dict): A handy dictionary with frameworks/benchmarck/task/fold info
        args (typing.Any): Namespace with the input argument of this script
        run_dir (str): Are on where to launch the jobs

    Returns:

    """

    # Collect a dataframe that will contain columns
    # tool Experiment metric train val test
    constraint = f"{args.runtime}s{args.cores}c{args.memory}"

    # Get first the list of jobs names to query
    job_names = []
    for framework, framework_dict in jobs.items():
        for benchmark, benchmark_dict in framework_dict.items():
            for task, task_dict in benchmark_dict.items():
                for fold, data in task_dict.items():
                    run_file = jobs[framework][benchmark][task][fold]['run_file']
                    name, ext = os.path.splitext(os.path.basename(run_file))
                    job_names.append(name)

    # Then get the overhead form slurm
    cmd = f"sacct -u riverav --format=JobID,JobName%100,MaxRSS,Elapsed,MaxDiskRead,MaxDiskWrite,MaxRSS,MaxVMSize,MinCPU,TotalCPU --parsable2 --name={','.join(job_names)} -S 2020-11-01"
    lines = remote_run(cmd)
    header = lines.pop(0).split('|')
    # We have to process 2 lines at the same time
    it = iter(lines)
    JobName2overhead = {}
    for line in iter(it):
        # JobID|JobName|MaxRSS|Elapsed|MaxDiskRead|MaxDiskWrite|MaxRSS|MaxVMSize|MinCPU|TotalCPU
        # 5432402|autosklearnBBCScoreEnsembleLatest_B_10_Nb_100_master_thesis_3600s1c8G_adult_4||01:00:31||||||12:33.749
        # 5432402.batch|batch|3052644K|01:00:31|180.41M|26.34M|3052644K|7083644K|00:03:56|12:33.749
        # 5432406|autosklearnBBCScoreEnsembleLatest_B_10_Nb_100_master_thesis_3600s1c8G_Albert_3||01:01:35||||||52:43.492
        # 5432406.batch|batch|5653216K|01:01:35|8938.85M|576.34M|5653216K|9714760K|00:10:02|52:43.492
        first_line = line.split('|')
        second_line = next(it).split('|')
        data = {}
        for i, (first, second) in enumerate(zip(first_line, second_line)):
            # It is super annoying that what we want might be in the first line or in the second
            # line.
            if 'batch' in first or first == '':
                data[header[i].rstrip()] = second.rstrip()
            else:
                data[header[i].rstrip()] = first.rstrip()

        JobName2overhead[data['JobName']] = data

    dataframe = []
    for framework, framework_dict in jobs.items():
        for benchmark, benchmark_dict in framework_dict.items():
            for task, task_dict in benchmark_dict.items():
                for fold, data in task_dict.items():
                    run_file = jobs[framework][benchmark][task][fold]['run_file']
                    name, ext = os.path.splitext(os.path.basename(run_file))
                    row = {
                        'tool': framework,
                        'task': task,
                        'fold': fold,
                    }
                    row.update(JobName2overhead[name])
                    # Throw in there the number of smac models fitted
                    try:
                        cmd = f"grep StatusType.SUCCESS {run_dir}/{framework}_{benchmark}_{constraint}_{task}_{fold}/*/debug/{task}/{fold}/smac3-output/run_*/runhistory.json"
                        row['SMACModelsSUCCESS'] = len(remote_run(cmd))
                    except Exception:
                        row['SMACModelsSUCCESS'] = 0
                    try:
                        cmd = f"grep StatusType. {run_dir}/{framework}_{benchmark}_{constraint}_{task}_{fold}/*/debug/{task}/{fold}/smac3-output/run_*/runhistory.json"
                        row['SMACModelsALL'] = len(remote_run(cmd))
                    except Exception:
                        row['SMACModelsALL'] = 0
                    dataframe.append(row)
    dataframe = pd.DataFrame(dataframe)
    return dataframe


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Manages the run of the benchmark')
    parser.add_argument(
        '-f',
        '--framework',
        help='What framework to manage',
        required=True
    )
    parser.add_argument(
        '-b',
        '--benchmarks',
        action='append',
        help='What benchmark to run',
        required=True,
        choices=['test', 'small', 'medium', 'large', 'master_thesis'],
    )
    parser.add_argument(
        '--input_dir',
        help='pre-ran smac directory area this is a patter',
        required=True,
    )
    parser.add_argument(
        '-t',
        '--task',
        required=False,
        help='What specific task to run'
    )
    parser.add_argument(
        '-c',
        '--cores',
        default=8,
        type=int,
        help='The number of cores to use'
    )
    parser.add_argument(
        '--runtime',
        default=3600,
        type=int,
        help='The number of seconds a run should take'
    )
    parser.add_argument(
        '-m',
        '--memory',
        default='32G',
        choices=['12G', '32G', '8G'],
        help='the ammount of memory to allocate to a job'
    )
    parser.add_argument(
        '-p',
        '--partition',
        default='bosch_cpu-cascadelake',
        choices=['ml_cpu-ivy', 'test_cpu-ivy', 'bosch_cpu-cascadelake'],
        help='In what partition to launch'
    )
    parser.add_argument(
        '--binary_metric',
        default="['balacc', 'auc', 'acc']",
        help='What metric set to use. Notice that it has to be a string'
    )
    parser.add_argument(
        '--multiclass_metric',
        default="['balacc', 'logloss', 'acc']",
        help='What metric set to use. Notice that it has to be a string'
    )
    parser.add_argument(
        '--regression_metric',
        default="['rmse', 'r2']",
        help='What metric set to use. Notice that it has to be a string'
    )
    parser.add_argument(
        '--total_folds',
        type=int,
        required=False,
        default=5,
        help='Total amount of folds to run'
    )
    parser.add_argument(
        '--folds',
        type=int,
        choices=list(range(5)),
        action='append',
        required=False,
        default=[],
        help='What fold out of 10 to launch'
    )
    parser.add_argument(
        '--run_mode',
        type=str,
        choices=['single', 'array', 'None', 'interactive'],
        default=None,
        help='Launches the run to sbatch'
    )
    parser.add_argument(
        '--max_active_runs',
        type=int,
        default=5,
        help='maximum number of active runs for a batch array'
    )
    parser.add_argument(
        '--verbose',
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help='Prints more debug info'
    )
    parser.add_argument(
        '--collect_overfit',
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help='generates a problems dataframe'
    )
    parser.add_argument(
        '--collect_overhead',
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help='generates a overhead'
    )
    parser.add_argument(
        '--collect_ensemble_history',
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help='generates a problems dataframe'
    )
    parser.add_argument(
        '--models',
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help='generates a models used dataframe'
    )
    parser.add_argument(
        '--run_dir',
        help='The area from where to run'
    )
    parser.add_argument(
        '--bbc_cv_sample_size',
        help='patter of wher the debug file originally is',
        required=False,
        default=0.50,
    )
    parser.add_argument(
        '--bbc_cv_n_bootstrap',
        help='patter of wher the debug file originally is',
        required=False,
        default=100,
    )
    parser.add_argument(
        '--ensemble_size',
        help='patter of wher the debug file originally is',
        required=False,
        default=50,
    )
    parser.add_argument(
        '--seed',
        help='patter of wher the debug file originally is',
        required=False,
        default=392,
    )
    parser.add_argument(
        '--fast',
        help='patter of wher the debug file originally is',
        required=False,
        default=True,
        type=str2bool,
        nargs='?',
        const=True,
    )

    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # Make sure singularity is properly set
    if 'kisbat' in socket.gethostname():
        if 'kislurm/singularity-3.5' not in os.environ['PATH']:
            raise ValueError(
                f"Singularity version to be used must be 3.5"
                "did you run export PATH=/usr/local/kislurm/singularity-3.5/bin/:$PATH ?"
            )

    # Get what framework the user wants to run
    if not args.run_dir:
        if args.framework not in [
            'autosklearnBBCScoreEnsemble',
            'autosklearnBBCScoreEnsembleMAX',
            'autosklearnBBCScoreEnsembleMAXWinner',
            'autosklearnBBCScoreEnsembleAVGMDEV',
            'autosklearnBBCEnsembleSelection',
            'autosklearnBBCEnsembleSelectionFULL',
            'autosklearnBBCEnsembleSelectionNoPreSelect',
            'autosklearnBBCEnsembleSelectionPreSelectInES',
            'autosklearnBBCEnsembleSelectionPreSelectInESRegularizedEnd',
            'bagging',
            'None',
        ]:
            raise ValueError(f"Unsupported framework {args.framework}!!")

    # Make sure the run_dir has the desired information
    run_dir = create_run_dir_area(
        run_dir=args.run_dir,
        args=args,
    )

    # Update the remote version of the benchmakr with the local version
    print(os.getcwd())
    updated_files = subprocess.run(
        f"rsync --update -avzhP -e \"ssh -p 22 -A {USER}@132.230.166.39 ssh\" {os.getcwd()}/* {USER}@{INTERNAL}:{AUTOMLBENCHMARK}",
        shell=True,
        stdout=subprocess.PIPE
    ).stdout.decode('utf-8')
    print(f"updated_files = {updated_files}")

    # Tasks are benchmark dependent
    if args.task:
        tasks = {}
        count = 0
        tasks_from_benchmark = get_task_from_benchmark(args.benchmarks)
        for benchmark in args.benchmarks:
            if args.task in tasks_from_benchmark[benchmark]:
                tasks[benchmark] = [args.task]
                count += 1
            else:
                tasks[benchmark] = []
        if count != 1:
            raise ValueError(
                f"For task={args.task} it was not uniquely defined for "
                f"benchmark={args.benchmarks}({count})"
            )
    else:
        tasks = get_task_from_benchmark(args.benchmarks)

    # If no folds provided, then use all of them
    if len(args.folds) == 0:
        args.folds = list(range(args.total_folds))

    # Get the job status
    constraint = f"{args.runtime}s{args.cores}c{args.memory}"
    jobs = get_job_status(
        framework=args.framework,
        benchmarks=args.benchmarks,
        tasks=tasks,
        folds=args.folds,
        constraint=constraint,
        run_dir=run_dir,
        args=args,
    )

    # Can only run on array or normal mode
    if args.run_mode:
        launch(
            jobs=jobs,
            args=args,
            run_dir=run_dir
        )

    if args.collect_overfit:
        overfit = collect_overfit3(
            jobs=jobs,
            args=args,
            run_dir=run_dir
        )
        filename = f"{args.framework}_{args.seed}_{args.ensemble_size}_{args.bbc_cv_n_bootstrap}_{args.bbc_cv_sample_size}_overfit.csv"
        logger.info(f"Please check {filename}")
        overfit.to_csv(filename)

    if args.collect_ensemble_history:
        overfit = collect_ensemble_history(
            jobs=jobs,
            args=args,
            run_dir=run_dir
        )
        filename = f"{args.framework}_{args.seed}_{args.ensemble_size}_{args.bbc_cv_n_bootstrap}_{args.bbc_cv_sample_size}_ensemble_history.csv"
        logger.info(f"Please check {filename}")
        overfit.to_csv(filename)

    if args.collect_overhead:
        overhead = collect_overhead(
            jobs=jobs,
            args=args,
            run_dir=run_dir
        )
        filename = f"{args.framework}_{args.seed}_{args.ensemble_size}_{args.bbc_cv_n_bootstrap}_{args.bbc_cv_sample_size}_overhead.csv"
        logger.info(f"Please check {filename}")
        overhead.to_csv(filename)

    # Close the connection to the server
    SSH.close()
    VM.close()
