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
import openml
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
def get_framework_information() -> typing.Dict[str, typing.Dict]:
    """
    Returns a dictionary with information from all the frameworks
    Returns:
        typing.Dict[str, Dict]: framework as key pointing to a dict with information
    """
    # Read in the framework file
    framework_file = 'resources/frameworks.yaml'
    if not os.path.exists(framework_file):
        raise ValueError("This script is expected to be run inside of the automlbenchmark dir")

    with open(framework_file) as file:
        # Valid frameworks is a dict that looks like:
        # 'TPOT': {'version': '0.11.5', 'project': 'https://github.com/EpistasisLab/tpot'},
        valid_frameworks = yaml.load(file, Loader=yaml.FullLoader)
    return valid_frameworks


def get_author_from_benchmark() -> typing.Dict[str, typing.Dict]:
    """
    Gets the author from the configuration file
    Returns:
        typing.Dict[str, Dict]: framework as key pointing to a dict with information
    """
    # read the config file
    config_file = 'resources/config.yaml'
    if not os.path.exists(config_file):
        raise ValueError("This script is expected to be run inside of the automlbenchmark dir")

    with open(config_file) as file:
        author = yaml.load(file, Loader=yaml.FullLoader)['container']['image_defaults']['author']
    return author


def create_singularity_image(framework: str) -> None:
    """
    Creates a singularity image via a docker image
    Args:
        framework: The framework from which to generate the image
    """
    command = subprocess.run([
        f"git rev-parse --abbrev-ref HEAD",
    ], shell=True, stdout=subprocess.PIPE)
    current_branch = command.stdout.decode('utf-8').strip()
    if 'master' in current_branch:
        raise ValueError('Using the master branch is not yet supported!')

    author = get_author_from_benchmark()
    valid_frameworks = get_framework_information()
    version = valid_frameworks[framework]['version'].replace('-', '_').lower()

    run_file = 'generate_sif.sh'
    command = f"""
#!/bin/bash
#source {ENVIRONMENT_PATH}
if [[ "$(docker images -q {author}/{framework.lower()}:{version}-stable 2> /dev/null)" == "" ]]; then
    echo "Please prese yes/enter to create the docker image"
    python runbenchmark.py {framework} -m docker -s only
    docker tag {author}/{framework.lower()}:{version}-{current_branch}  {author}/{framework.lower()}:{version}-stable
    docker push  {author}/{framework.lower()}:{version}-stable
fi
singularity pull frameworks/{framework}/{framework.lower()}_{version}-stable.sif docker://{author}/{framework.lower()}:{version}-stable
    """
    with open(run_file, 'w') as f:
        f.write(command)
    command = subprocess.run([
        f"bash {run_file}",
    ], shell=True, stdout=subprocess.PIPE)

    # Check if things went ok
    sif_file = f"frameworks/{framework}/{framework.lower()}_{version}-stable.sif"
    if not os.path.exists(sif_file):
        raise Exception(f"Failed to generate the sif file {sif_file}")
    return True


def validate_framework(framework: str) -> None:
    """
    Makes sure we can run a given framework. This implies:
        + Checking if the framework is in the frameworks folder
        + Check if the framework is in the resource file
        + Check if there is a valid singularity image for this job
    Args:
        framework (str): framework to validate
    """
    valid_frameworks = get_framework_information()

    # We only allow running a framework with a singularity image already downloaded
    # Also, we require the framework to be in the resource file
    if framework not in valid_frameworks.keys():
        raise ValueError(f"We expect that the framework={framework} will be in the "
                         "{framework_file} file. This is required by automlbenchmark")
    version = valid_frameworks[framework]['version'].replace('-', '_')
    sif_file = f"frameworks/{framework}/{framework.lower()}_{version.lower()}-stable.sif"

    if not os.path.exists(sif_file):
        logger.warn("Trying to generate the singularity image. Please enter 'y' for yes"
                    "When the benchmark ask you about if you are sure you want to generate"
                    "the image.")
        create_singularity_image(framework)

        # Try to create the file
        if not os.path.exists(sif_file):
            raise ValueError(f"Because we run in singularity mode, we require the sif {sif_file} "
                              "file to be available. Please take a look into "
                              "https://aadwiki.informatik.uni-freiburg.de/automlbenchmark "
                              "to see how to generate this file. ")


def generate_metadata(run_dir: str, args: typing.Any, timestamp: str,
                      experiment_description:str) -> None:
    """Generates a metadata file with run information for debug purposes
    Args:
        run_dir (str): where to generate this file
        args (typing.Any): Namespace with args for this python script
        timestamp (str): When the run was launched
    """
    constraint = f"{args.runtime}s{args.cores}c{args.memory}"

    # we want to output a json file with all info
    metadata = {}  # type: typing.Dict[str, typing.Union[str, int]]

    # Constraint info
    metadata['constraint_name'] = constraint
    metadata['cores'] = args.cores
    metadata['slurm_memory'] = args.memory
    metadata['job_memory'] = MEMORY[args.memory]
    metadata['runtime'] = args.runtime
    metadata['framework'] = args.framework
    valid_frameworks = get_framework_information()
    metadata['version'] = valid_frameworks[args.framework]['version']
    metadata['run_date'] = timestamp
    metadata['description'] = experiment_description
    metadata['automlbenchmark_commit'] = subprocess.run(
        ['git rev-parse HEAD'], shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8').strip()
    metadata['automlbenchmark_commit_date'] = subprocess.run(
        ['git log -1 --format=%cd '], shell=True, stdout=subprocess.PIPE
    ).stdout.decode('utf-8').strip()
    with open(f"/tmp/metadata.json", 'w') as fp:
        json.dump(metadata, fp)
    remote_put('/tmp/metadata.json', f"{run_dir}/metadata.json")


def remote_exists(directory: str) -> bool:
    """Checks if a directory exist on remote host
    Args:
        directory: The remote repository
    Returns:
        bool: whether or not the directory exists
    """
    logger.debug(f"[remote_exists] {directory}")
    stdin, stdout, stderr = SSH.exec_command(f"ls -d {directory}") #edited#
    stdout.channel.recv_exit_status()
    stderr.channel.recv_exit_status()
    if any(['cannot access' in a for a in stderr.readlines()]):
        return False
    else:
        return True


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
    return stdout.readlines()


def get_user_description() -> str:
    """
    Ask the user for a description of the experiment to be run.
    Returns:
        str: A description to annotate the experiment. This is to remind the
             purpose of the experiment
    """
    correct = False
    while not correct:
        inp = input("Please provide a brief description of the experiment."
                    "\nThis will be stored in the metadata to remind yourself the "
                    "\npurpose of this experiment: ")

        logger.info("We are gonna annotate this experiment with the description: "
                    f"\n{inp}\n ")
        correct = query_yes_no('Are you ok with this?')
    return inp


def create_run_dir_area(run_dir: typing.Optional[str], args: typing.Any,
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

    # Get user description and store it in the metadata
    experiment_description = get_user_description()

    # No run dir provided, create one!
    valid_frameworks = get_framework_information()
    version = valid_frameworks[args.framework]['version']
    timestamp = time.strftime("%Y.%m.%d-%H.%M.%S")
    base_framework = args.framework.split('_', 1)[0]
    run_dir = os.path.join(
        BASE_PATH,
        base_framework,
        version,
        timestamp,
    )

    remote_makedirs(run_dir)

    # make also a run dir for scripts / logs
    remote_makedirs(os.path.join(run_dir, 'scripts'))
    remote_makedirs(os.path.join(run_dir, 'logs'))
    logger.info(f"Creating run directory: {run_dir}")

    # Copy over the frameworks dir to this area. We do so, because
    # we want to have a completely isolated and reproducible run.
    # So we copy over the resources to this directory
    remote_put('resources/frameworks.yaml', f"{run_dir}/frameworks.yaml")

    # Copy the SIF file
    sif_file = f"frameworks/{args.framework}/{args.framework.lower()}_{version.lower()}-stable.sif"
    src = f"{AUTOMLBENCHMARK}/frameworks/{args.framework}/{os.path.basename(sif_file)}"
    dst = f"{run_dir}/{os.path.basename(sif_file)}"
    remote_run(f"cp {src} {dst}")
    if not remote_exists(dst):
        raise ValueError("We expect the sif file for singularity to be both in the "
                         f"run dir {run_dir} and {AUTOMLBENCHMARK}, yet copying failed! "
                         f"Command: cp {src} {dst}")

    # We create the constrain we are gonna use
    constraint = f"{args.runtime}s{args.cores}c{args.memory}"
    with open(f"/tmp/constraints.yaml", 'w') as file_c:
        file_c.write(f"{constraint}:\n")
        file_c.write("  folds: 10\n")
        file_c.write(f"  max_runtime_seconds: {args.runtime}\n")
        file_c.write(f"  cores: {args.cores}\n")
        file_c.write(f"  max_mem_size_mb: {MEMORY[args.memory]}\n")
    remote_put('/tmp/constraints.yaml', f"{run_dir}//constraints.yaml")

    # Create a config file, mostly to pass a metric
    # And then also, make the run reproducible/isolated via local info
    with open(f"/tmp/config.yaml", 'w') as file_c:
        file_c.write("benchmarks:\n")
        file_c.write("  metrics:\n")
        file_c.write(f"    binary: {args.binary_metric}\n")
        file_c.write(f"    multiclass: {args.multiclass_metric}\n")
        file_c.write(f"    regression: {args.regression_metric}\n")
        file_c.write("  constraints_file:\n")
        file_c.write("    " + '- \'{root}/resources/constraints.yaml\'' + "\n")
        file_c.write("    " + '- \'{user}/constraints.yaml\'' + "\n")
        file_c.write("\n")
        file_c.write("frameworks:\n")
        file_c.write("  definition_file:\n")
        file_c.write("    " + '- \'{user}/frameworks.yaml\'' + "\n")
    remote_put('/tmp/config.yaml', f"{run_dir}//config.yaml")

    # Lastly we create a metatada file
    generate_metadata(run_dir, args, timestamp, experiment_description)

    return run_dir


def slugify(value):
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.
    """
    value = str(re.sub('[^\w\s-]', '_', value))
    return value


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

    run_file = f"{run_dir}/scripts/{framework}_{slugify(benchmark)}_{constraint}_{slugify(task)}_{fold}.sh"
    virtual_memory_available = 4469755084 if '8G' in constraint else 34000000000

    if 'openml' in task:
        cmd = f"python runbenchmark.py {framework} {task} {constraint} --fold {fold} -m singularity --session {framework}_{slugify(benchmark)}_{constraint}_{slugify(task)}_{fold} -o {run_dir}/{framework}_{slugify(benchmark)}_{constraint}_{slugify(task)}_{fold} -u {run_dir}"
    else:
        cmd = f"python runbenchmark.py {framework} {benchmark} {constraint} --task {task} --fold {fold} -m singularity --session {framework}_{benchmark}_{constraint}_{task}_{fold} -o {run_dir}/{framework}_{benchmark}_{constraint}_{task}_{fold} -u {run_dir}"

    command = f"""#!/bin/bash
#Setup the run
echo "Running on HOSTNAME=$HOSTNAME with name $SLURM_JOB_NAME"
export PATH=/usr/local/kislurm/singularity-3.5/bin/:$PATH
source {ENVIRONMENT_PATH}
cd {AUTOMLBENCHMARK}
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then export TMPDIR=/tmp/{framework}_{slugify(benchmark)}_{constraint}_{task}_{fold}_$SLURM_JOB_ID; else export TMPDIR=/tmp/{framework}_{slugify(benchmark)}_{constraint}_{task}_{fold}$SLURM_ARRAY_JOB_ID'_'$SLURM_ARRAY_TASK_ID; fi
echo TMPDIR=$TMPDIR
export XDG_CACHE_HOME=$TMPDIR
echo XDG_CACHE_HOME=$XDG_CACHE_HOME
mkdir -p $TMPDIR
export SINGULARITY_BINDPATH="$TMPDIR:/tmp"
export VIRTUAL_MEMORY_AVAILABLE={virtual_memory_available}

# Config the run
export framework={framework}
export benchmark={benchmark}
export constraint={constraint}
export task={task}
export fold={fold}
# Sleep a random number of seconds after init to be safe of run
sleep {np.random.randint(low=0,high=10)}
echo {cmd}
{cmd}
echo "Deleting temporal folder $TMPDIR"
rm -rf $TMPDIR
echo 'Finished the run'
"""

    with open(os.path.join('/tmp', os.path.basename(run_file)), 'w') as f:
        f.write(command)
    remote_put(os.path.join('/tmp', os.path.basename(run_file)), run_file)
    return run_file


def is_openml_suite(benchmark: str) -> bool:
    """
    Utility to check that the provided argument is a benchmark suite
    Args:
        benchmark (str): a string that could be a benchmark suite
    Returns
        (bool): whether or not the provided string is a benchmark suite
    """
    openml_type_id = benchmark.split('/')
    if len(openml_type_id) == 3 and openml_type_id[0] == 'openml' and openml_type_id[1] == 's':
        try:
            openml.study.get_suite(openml_type_id[2])
        except Exception:
            logger.error(f"Could not interpret {benchmark} as suite. It will be ignored...")
            return False
        return True
    return False


def is_openml_task(task: str) -> bool:
    """
    Utility to check that the provided argument is an openml task
    Args:
        task (str): a string that could be a task
    Returns
        (bool): whether or not the provided string is a task
    """
    openml_type_id = task.split('/')
    if len(openml_type_id) == 3 and openml_type_id[0] == 'openml' and openml_type_id[1] == 't':
        try:
            openml.tasks.get_task(openml_type_id[2], download_data=False)
        except Exception:
            logger.error(f"Could not interpret {task} as task. It will be ignored...")
            return False
        return True
    return False


def convert_openml_suite_to_openml_task(benchmark: str) -> typing.Dict[str, str]:
    """
    Convert an openml suite to tasks
    Args:
        benchmark (str): a string that could be a benchmark suite
    Returns:
        Dict[str, str]: A mapping from suite to tasks
    """
    suite = openml.study.get_suite(218)
    return {benchmark: [f"openml/t/{task}" for task in suite.tasks]}


def get_task_from_benchmark(benchmarks: typing.List[str]) -> typing.Dict[str, typing.List[str]]:
    """Returns a dict with benchmark to task mapping
    Args:
        benchmakrs (typing.List[str]): List of benchmakrs to run
    Returns:
        typing.Dict[str, str]: mapping from benchmark to task
    """
    task_from_benchmark = {}  # type: typing.Dict[str, typing.List]

    for benchmark in benchmarks:

        # In the case of openmlsuite
        if is_openml_suite(benchmark):
            task_from_benchmark.update(convert_openml_suite_to_openml_task(benchmark))
        elif is_openml_task(benchmark):
            # openml/t/59 is actually used as a benchmark and task at the same time
            task_from_benchmark[benchmark] = [benchmark]
        else:
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
        logger.warning(f"No RandomForest result for for benchmark={benchmark}")
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
    result_file = f"{run_dir}/{framework}_{slugify(benchmark)}_{constraint}_{slugify(task)}_{fold}/results.csv"

    if not remote_exists(result_file):
        return None

    result_file = remote_get(result_file)

    df = pd.read_csv(result_file)
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


def check_if_running(run_file: str, tag: typing.Optional[str] = None) -> bool:
    """
    Queries SLURM to check if a job is running or not
    Args:
        run_file (str): the script that might have been used to launch the run
    Returns:
        bool: Whether the job is running or not
    """
    name, ext = os.path.splitext(os.path.basename(run_file))

    if tag is not None:
        name = name + tag

    # First check if there is a job with this name
    cmd = f"squeue --format=\"%.190j\" --noheader -u {USER}"
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


def are_resource_available_to_run(partition: str, min_cpu_free=128, max_total_runs=50):
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
        if not check_if_running(task, args.tag):
            not_launched_runs.append(task)
    if not not_launched_runs:
        return
    run_files = not_launched_runs

    # Run options
    extra = ''
    if args.partition == 'bosch_cpu-cascadelake':
        extra += ' --bosch'

    # For array
    max_hours=4
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
                job_name = name
                if args.tag is not None:
                    job_name = name + args.tag
                # try WA for autopytorch 1h1c 8G
                memory = args.memory
                this_extra = extra + f" -p {args.partition} -t 0{max_hours}:00:00 --mem {args.memory} -c {args.cores} --job-name {job_name} -o {os.path.join(run_dir, 'logs', name + '_'+ timestamp + '.out')}"
                _launch_sbatch_run(this_extra, task)
            else:
                logger.warning(f"Skip {task} as there are no more free resources... try again later!")
            # Wait 2 sec to update running job
            time.sleep(2)
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
    logger.info('_'*40)
    logger.info(f"\t\t{framework}")
    logger.info('_'*40)

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
                )

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
                elif check_if_running(jobs[framework][benchmark][task][fold]['run_file'], tag=args.tag):
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

    logger.info('_'*40)
    logger.info(f" A total of {total} runs checked")
    logger.info('_'*40)

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
                    results = jobs[framework][benchmark][task][fold]['results']
                    valid_result = is_number(results)

                    # Launch the run if it was not yet launched
                    if results is None:
                        if check_if_running(run_file, args.tag):
                            status = 'Running'
                        else:
                            run_files.append(run_file)
                            status = 'Launched'
                    else:
                        # If the run failed, then ask the user to relaunch
                        if not valid_result:
                            if check_if_running(jobs[framework][benchmark][task][fold]['run_file'], args.tag):
                                status = 'Running'
                            else:
                                status = 'Failed'
                                if args.force or query_yes_no(f"For framework={framework} benchmark={benchmark} constraint={constraint} task={task} fold={fold} obtained: {jobs[framework][benchmark][task][fold]['results']}. Do you want to relaunch this run?"):
                                    run_files.append(jobs[framework][benchmark][task][fold]['run_file'])
                                    status = 'Relaunched'

    # Launch the runs in sbatch
    if run_files:
        launch_run(
            run_files,
            args=args,
            run_dir=run_dir,
        )


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
                    result_file = f"{run_dir}/{framework}_{slugify(benchmark)}_{constraint}_{slugify(task)}_{fold}/results.csv"
                    if remote_exists(result_file):
                        result_file = remote_get(result_file)
                        dataframe.append(pd.read_csv(result_file))

    return pd.concat(dataframe).reset_index(drop=True)

def get_normalized_score(
    jobs: typing.Dict[str, typing.Dict],
    framework: str,
    benchmarks: typing.List[str],
    tasks: typing.Dict[str, typing.Any],
    folds: typing.List[int]
) -> pd.DataFrame:
    """
    Creates a pandas dataframe with the normalized score for the runs
    Args:
        jobs (typing.Dict): A handy dictionary with frameworks/benchmarck/task/fold info
        framework (str): framework that ran
        benchmarks (typingList[str]):  in what benchmakrs they ran
        tasks (typing.List[str]): what task ids where ran
        folds (typing.List[int]): in which folds for CV
    Returns
        pd.DataFrame: A Frame with the CV average of the normalized scores of each run
    """

    # Create a Dataframe
    dataframe = []
    for benchmark in benchmarks:
        for task in tasks[benchmark]:
            row = {
                'benchmark': benchmark,
                'Task': task,
            }
            average = []
            for fold in folds:
                score = jobs[framework][benchmark][task][fold]['norm_score']
                if is_number(score):
                    average.append(score)
            if len(average) < 1:
                average_result = 'N/A'
                row[framework + '_mean'] = average_result
                row[framework + '_std'] = average_result
            else:
                row[framework + '_mean'] = np.nanmean(average) if np.any(average) else 0
                row[framework + '_std'] = np.nanstd(average) if np.any(average) else 0
            row[framework + '_num_folds'] = len(average)
            dataframe.append(row)
    dataframe = pd.DataFrame(dataframe)
    return dataframe


def get_problems(
    jobs: typing.Dict[str, typing.Dict],
    framework: str,
    benchmarks: typing.List[str],
    tasks: typing.Dict[str, typing.Any],
    folds: typing.List[int]
) -> pd.DataFrame:
    """
    Creates a pandas dataframe with the reason for crash of a run
    Args:
        jobs (typing.Dict): A handy dictionary with frameworks/benchmarck/task/fold info
        framework (str): framework that ran
        benchmarks (typingList[str]):  in what benchmakrs they ran
        tasks (typing.List[str]): what task ids where ran
        folds (typing.List[int]): in which folds for CV
    Returns
        pd.DataFrame: A Frame with crash information per run
    """
    # Create a Dataframe
    dataframe = []
    for benchmark in benchmarks:
        for task in tasks[benchmark]:
            row = {
                'benchmark': benchmark,
                'Task': task,
                'Issues': [],
            }
            row[framework] = 0
            for fold in folds:
                # If this run crashed,  register why
                if 'Chrashed' in jobs[framework][benchmark][task][fold]['status']:
                    if jobs[framework][benchmark][task][fold]['results'] not in row['Issues']:
                        row['Issues'].append(jobs[framework][benchmark][task][fold]['results'])
                    row[framework] += 1
            dataframe.append(row)
    dataframe = pd.DataFrame(dataframe)
    return dataframe


def complete_missing_csv():
    results_file = 'results.csv'
    results = pd.read_csv(os.path.join('results', results_file))

    for local_path in glob.glob('results/*/scores/results.csv'):
        local_result = pd.read_csv(local_path)
        if not local_result.empty:
            # Just take rsults that make sense
            local_result = local_result[local_result.applymap(np.isreal)['result']]
            local_result = local_result[~local_result['fold'].isnull()]
            if not local_result.empty:
                results = pd.concat([results, local_result], sort=False).drop_duplicates()

    results['fold'] = results['fold'].astype('int32')
    results.to_csv("final.csv", index=False)


def get_autosklearn_model_list(model_file):

    if not os.path.exists(model_file):
        logger.error(f"Could not find model file: {model_file}")
        return []

    pattern = re.compile(".*'classifier:__choice__': '(\w+)'.*")  # noqa: W605
    model_list = []
    for i, line in enumerate(open(model_file)):
        match = re.match(pattern, line)
        if match:
            model_list.append(match.group(1))
    return model_list


def get_h2o_model_list(model_file):

    # Read the json file
    model_file = model_file.replace('full_models', 'models').replace('.zip', '.json')
    if not os.path.exists(model_file):
        logger.error(f"Could not find model file: {model_file}")
        return []

    if 'StackedEnsemble' not in model_file:
        # Sometimes a model is better than the stack
        model_file = os.path.basename(model_file).split('_')[0]
        return [model_file]

    model_list = []
    parsed = json.load(open(model_file))
    for i, params in enumerate(parsed['parameters']):
        if not parsed['parameters'][i]['name'] == 'base_models':
            continue
        for model in parsed['parameters'][i]['actual_value']:
            model_list.append(model['name'].split('_')[0])
        break
    return model_list


def get_model_list(run_file):
    "Tries to extract the list of models from a runfile"
    name, ext = os.path.splitext(os.path.basename(run_file))
    logfile = glob.glob(os.path.join('logs', name + '*' + '.out'))
    if len(logfile) == 0:
        return False
    logfile.sort()
    logfile = logfile[0]
    match = re.match(
        "([\w-]+)_(small|medium|large|test)_([\dA-Za-z]+)_([.\w-]+)_(\d)", name)  # noqa: W605
    if match is None:
        logger.error(f"Could not match name={name}")
        return []
    framework, benchmark, constraint, task, fold = match.groups()

    # If we cant file the model file, bail out
    if not os.path.exists(logfile):
        logger.error(f"Could not find log file: {logfile}")
        return []

    # Get the path where models where saved
    pattern = re.compile(
        ".*Scores\s*saved\s*to\s*`\/output\/(.*)\/scores\/results.csv.*")  # noqa: W605
    result_file = None
    for i, line in enumerate(open(logfile)):
        match = re.match(pattern, line)
        if match:
            result_file = match.group(1)
            break
    if result_file is None:
        logger.error(f"Could not extract result file from: {logfile}")
        return []

    # Extract per framework models
    if 'autosklearn' in framework:
        model_file = os.path.join('results', result_file, 'models', task, fold, 'models.txt')
        return get_autosklearn_model_list(model_file)
    elif 'H2O' in framework:
        model_file = glob.glob(
            os.path.join('results', result_file, 'full_models', task, fold, '*.zip')
        )
        if len(model_file) < 1:
            logger.critical(f"Error while parsing {run_file}.. No model file found!")
            return []
        return get_h2o_model_list(model_file[-1])
    else:
        raise ValueError(f"No support for framework={framework}")


def get_used_models_per_framework(jobs):
    """
    Returns a per framework dict with a dataframe of models used.
    If more than 1 fold is provided, the average is taken
    """
    framework_models = {}
    for framework, framework_dict in jobs.items():
        rows = []
        for benchmark, benchmark_dict in framework_dict.items():
            for task, task_dict in benchmark_dict.items():
                model_usage = {'benchmark': benchmark, 'task': task}
                for fold, data in task_dict.items():
                    run_file = jobs[framework][benchmark][task][fold]['run_file']
                    for model in get_model_list(run_file):
                        if model not in model_usage:
                            model_usage[model] = 1
                        else:
                            model_usage[model] += 1

                # Calculate average
                for key, value in model_usage.items():
                    if key == 'task':
                        continue
                    if key == 'benchmark':
                        continue
                    model_usage[key] = value/len(task_dict.keys())

                rows.append(model_usage)
        framework_models[framework] = pd.DataFrame(rows)
        framework_models[framework].to_csv(f"{framework}_models.csv", index=False)
    return framework_models


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Manages the run of the benchmark')
    parser.add_argument(
        '-f',
        '--framework',
        help='What framework to manage',
        choices=list(get_framework_information().keys()),
        required=True
    )
    parser.add_argument(
        '-b',
        '--benchmarks',
        action='append',
        help='What benchmark to run',
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
        choices=['12G', '32G'],
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
        #default="['balacc', 'auc', 'acc']",
        default="['auc', 'logloss', 'acc']",
        help='What metric set to use. Notice that it has to be a string'
    )
    parser.add_argument(
        '--multiclass_metric',
        #default="['balacc', 'logloss', 'acc']",
        default="['logloss', 'acc']",
        help='What metric set to use. Notice that it has to be a string'
    )
    parser.add_argument(
        '--regression_metric',
        default="['rmse', 'r2']",
        help='What metric set to use. Notice that it has to be a string'
    )
    parser.add_argument(
        '--folds',
        type=int,
        choices=list(range(10)),
        action='append',
        required=False,
        default=[],
        help='What fold out of 10 to launch. If none is provided, it will be assumed all 10 folds need to be ran'
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
        '--problems',
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help='generates a problems dataframe'
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
        '--tag',
        default=None,
        required=False,
        help='to tag a run so name does not collide'
    )
    parser.add_argument(
        '--force',
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help='Forces to rerun a failed run'
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
        validate_framework(args.framework)

    # Update the remote version of the benchmakr with the local version
    print(os.getcwd())
    updated_files = subprocess.run(
        f"rsync --update -avzhP -e \"ssh -p 22 -A {USER}@132.230.166.39 ssh\" {os.getcwd()}/* {USER}@{INTERNAL}:{AUTOMLBENCHMARK}",
        shell=True,
        stdout=subprocess.PIPE
    ).stdout.decode('utf-8')
    logger.info(f"updated_files = {updated_files}")

    # Make sure the run_dir has the desired information
    run_dir = create_run_dir_area(
        run_dir=args.run_dir,
        args=args,
    )

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
        args.folds = list(range(10))

    # Get the job status
    constraint = f"{args.runtime}s{args.cores}c{args.memory}"
    jobs = get_job_status(
        framework=args.framework,
        benchmarks=args.benchmarks,
        tasks=tasks,
        folds=args.folds,
        constraint=constraint,
        run_dir=run_dir
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
        filename = f"{args.framework}_overfit.csv"
        logger.info(f"Please check {filename}")
        overfit.to_csv(filename)
    # Close the connection to the server
    SSH.close()
    VM.close()
