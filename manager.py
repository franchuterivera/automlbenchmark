import tempfile
from functools import partial
import uuid
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

    author = get_author_from_benchmark()
    valid_frameworks = get_framework_information()
    version = valid_frameworks[framework]['version'].replace('-', '_').lower()

    run_file = 'generate_sif.sh'
    # There are two ways to generate a local sif image from a docker image
    # https://github.com/hpcng/singularity/issues/1537
    # https://www.nas.nasa.gov/hecc/support/kb/converting-docker-images-to-singularity-for-use-on-pleiades_643.html
    temp_sif_name = os.path.join(
        tempfile.gettempdir(),
        str(uuid.uuid1(clock_seq=os.getpid())),
    )
    command = f"""
#!/bin/bash
#source {ENVIRONMENT_PATH}
if [[ "$(docker images -q {author}/{framework.lower()}:{version}-stable 2> /dev/null)" == "" ]]; then
    echo "Please prese yes/enter to create the docker image"
    python runbenchmark.py {framework} -m docker -s only
fi
docker save {author}/{framework.lower()}:{version}-{current_branch} -o {temp_sif_name}
singularity build frameworks/{framework}/{framework.lower()}_{version}-stable.sif docker-archive://{temp_sif_name}
cd frameworks/{framework}/
# Compatibility with development images
ln -sf {framework.lower()}_{version}-stable.sif {framework.lower()}_{version}-dev.sif
# Compatibility with stable images
ln -sf {framework.lower()}_{version}-stable.sif {framework.lower()}_{version}_stable.sif
cd ../..
rm {temp_sif_name}
    """
    with open(run_file, 'w') as f:
        f.write(command)
    raise NotImplementedError(sif_file)
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
        logger.warning("Trying to generate the singularity image. Please enter 'y' for yes"
                    "When the benchmark ask you about if you are sure you want to generate"
                    "the image.")
        create_singularity_image(framework)

        # Try to create the file
        if not os.path.exists(sif_file):
            raise ValueError(f"Because we run in singularity mode, we require the sif {sif_file} "
                              "file to be available. Please take a look into "
                              "https://aadwiki.informatik.uni-freiburg.de/automlbenchmark "
                              "to see how to generate this file. ")


def generate_metadata(run_dir: str, args: typing.Any, timestamp: str) -> None:
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
    destination = os.path.join('/tmp', str(uuid.uuid1(clock_seq=os.getpid())) + os.path.basename(source))

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

    # Copy also the benchmarks
    remote_put('resources/benchmarks', f"{run_dir}/benchmarks")

    # Copy the SIF file
    sif_file = f"frameworks/{args.framework}/{args.framework.lower()}_{version.lower()}-stable.sif"
    src = f"{AUTOMLBENCHMARK}/frameworks/{args.framework}/{os.path.basename(sif_file)}"
    dst = f"{run_dir}/{os.path.basename(sif_file)}"
    remote_run(f"cp {src} {dst}")
    if not remote_exists(dst):
        raise ValueError("We expect the sif file for singularity to be both in the "
                         f"run dir {run_dir} and {AUTOMLBENCHMARK}, yet copying failed! "
                         f"Command: cp {src} {dst}")
    #copyfile(sif_file, f"{run_dir}/{os.path.basename(sif_file)}")
    #remote_put(sif_file, f"{run_dir}/{os.path.basename(sif_file)}")

    # Extract the version of smac/autosklearn
    #cwd = os.getcwd();
    #subprocess.run(
    #    [f"cd {run_dir}; singularity exec  {os.path.basename(sif_file)} cp  -r /bench .; cd {cwd}"],
    #    shell=True, stdout=subprocess.PIPE
    #).stdout.decode('utf-8').strip()

    # We create the constrain we are gonna use
    constraint = f"{args.runtime}s{args.cores}c{args.memory}"
    with open(f"/tmp/constraints.yaml", 'w') as file_c:
        file_c.write(f"{constraint}:\n")
        file_c.write(f"  folds: {args.total_folds}\n")
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
        file_c.write("  definition_dir:\n")
        file_c.write("    " + '- \'{user}/benchmarks\'' + "\n")
        file_c.write("\n")
        file_c.write("frameworks:\n")
        file_c.write("  definition_file:\n")
        #file_c.write("    " + '- \'{root}/resources/frameworks.yaml\'' + "\n")
        file_c.write("    " + '- \'{user}/frameworks.yaml\'' + "\n")
    remote_put('/tmp/config.yaml', f"{run_dir}//config.yaml")

    # Lastly we create a metatada file
    generate_metadata(run_dir, args, timestamp)

    return run_dir


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

    virtual_memory_available = 4469755084 if '8G' in constraint else 34000000000

    query_for_tmp = '${TMPDIR+x}'
    command = f"""#!/bin/bash
# Setup the run
echo "Running on HOSTNAME=$HOSTNAME with name $SLURM_JOB_NAME"
export PATH=/usr/local/kislurm/singularity-3.5/bin/:$PATH
source {ENVIRONMENT_PATH}
cd {AUTOMLBENCHMARK}
# If the temporary directory is set, honor it
if [ -z {query_for_tmp} ]; then export TMPDIR='/tmp'; else echo "TMPDIR is set to '$TMPDIR'"; fi
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then export TMPDIR=$TMPDIR/{framework}_{benchmark}_{constraint}_{task}_{fold}_$SLURM_JOB_ID; else export TMPDIR=$TMPDIR/{framework}_{benchmark}_{constraint}_{task}_{fold}$SLURM_ARRAY_JOB_ID'_'$SLURM_ARRAY_TASK_ID; fi
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
    result_file = f"{run_dir}/{framework}_{benchmark}_{constraint}_{task}_{fold}/results.csv"

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
    return False

    # The check in the user job arrays
    cmd = f"squeue --format=\"%.150i\" --noheader -u {USER}"
    #result = subprocess.run([
    #    f"squeue --format=\"%.50i\" --noheader -u {os.environ['USER']}"
    #], shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8')

    #jobids = [jobid.split('_')[0] for jobid in result.splitlines() if "_" in jobid]
    jobids = [jobid.split('_')[0] for jobid in remote_run(cmd) if "_" in jobid]
    if jobids:
        jobids = list(set(jobids))
    for i, array_job in enumerate(jobids):
        #command = subprocess.run([
        #], shell=True, stdout=subprocess.PIPE)
        #command_out, filename = command.stdout.decode('utf-8').split('=')
        cmd = f"scontrol show jobid -dd {array_job} | grep Command |  sort --unique"
        filename = remote_run(cmd)[0]

        filename = filename.lstrip().rstrip()
        filename = filename.replace('Command=', '')
        if not remote_exists(filename):
            raise Exception(f"Could not find file {filename} from command={cmd} array_job={array_job }")
        filename = remote_get(filename)
        with open(filename, 'r') as data_file:
            if name in data_file.read():
                return True

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

    filename = f"arrayjob_{len(run_file)}_{os.getpid()}_{randrange(100)}.sh"

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

    with open(filename, 'w') as f:
        f.write(command)
    remote_name = f"{run_dir}/scripts/{filename}"
    remote_put(filename, remote_name)
    return remote_name


def are_resource_available_to_run(partition: str, min_cpu_free=128, max_total_runs=5):
    """
    Query slurm to make sure we can launch jobs

    Args:
        partition (str): which partition to launch and check
        min_cpu_free: only launch if cpu free
    """
    return True
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
    max_hours = (args.runtime + 3600) // 3600
    if 'array' in args.run_mode:
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
                                if query_yes_no(f"For framework={framework} benchmark={benchmark} constraint={constraint} task={task} fold={fold} obtained: {jobs[framework][benchmark][task][fold]['results']}. Do you want to relaunch this run?") or args.force:
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


def collect_overfit(
    jobs: typing.Dict,
    args: typing.Any,
    run_dir: str
) -> pd.DataFrame:
    """
    Collects overfit data into a dataframe

    Args:
        jobs (typing.Dict): A handy dictionary with frameworks/benchmarck/task/fold info
        args (typing.Any): Namespace with the input argument of this script
        run_dir (str): Are on where to launch the jobs

    Returns:

    """

    # Collect a dataframe that will contain columns
    # tool Experiment metric train val test
    constraint = f"{args.runtime}s{args.cores}c{args.memory}"
    metric = get_metric_from_run_area(run_dir)
    dataframe = []
    for framework, framework_dict in jobs.items():
        for benchmark, benchmark_dict in framework_dict.items():
            for task, task_dict in benchmark_dict.items():
                train, test, val = {}, {}, {}
                for fold, data in task_dict.items():
                    for overfit_file in remote_glob(os.path.join(
                        run_dir,
                        f"{framework}_{benchmark}_{constraint}_{task}_{fold}",
                        '*',  # The directory name the benchmark created
                        'overfit',
                        task,
                        str(fold),
                        'overfit.csv'
                    )):
                        overfit_file = remote_get(overfit_file)
                        frame = pd.read_csv(overfit_file, index_col=0)
                        for index, row in frame.iterrows():
                            row_dict = row.to_dict()
                            # Make best_individual_modelSOMETHING -> best_individual_model
                            if 'best_individual_model' in row_dict['model']:
                                row_dict['model'] = 'best_individual_model'
                            if 'best_ensemble_model' in row_dict['model']:
                                row_dict['model'] = 'best_ensemble_model'
                            if row_dict['model'] not in train:
                                train[row_dict['model']] = []
                            train[row_dict['model']].append(row_dict['train'])
                            if row_dict['model'] not in val:
                                val[row_dict['model']] = []
                            val[row_dict['model']].append(row_dict['val'])
                            if row_dict['model'] not in test:
                                test[row_dict['model']] = []
                            test[row_dict['model']].append(row_dict['test'])
                for model in train.keys():
                    row_dict = {
                        'tool': framework,
                        'task': task,
                        'metric': metric,
                        'model': model,
                        'train_mean':  np.nanmean(train[model]) if np.any(train[model]) else 0,
                        'val_mean':  np.nanmean(val[model]) if np.any(val[model]) else 0,
                        'test_mean':  np.nanmean(test[model]) if np.any(test[model]) else 0,
                        'train_std':  np.nanstd(train[model]) if np.any(train[model]) else 0,
                        'val_std':  np.nanstd(val[model]) if np.any(val[model]) else 0,
                        'test_std':  np.nanstd(test[model]) if np.any(test[model]) else 0,
                    }
                    dataframe.append(row_dict)

    return pd.DataFrame(dataframe)


def collect_overfit2(
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

    all_test_scores = []
    all_overfits = []

    # Collect a dataframe that will contain columns
    # tool Experiment metric train val test
    constraint = f"{args.runtime}s{args.cores}c{args.memory}"
    metric = get_metric_from_run_area(run_dir)
    dataframe = []
    for framework, framework_dict in jobs.items():
        for benchmark, benchmark_dict in framework_dict.items():
            for task, task_dict in benchmark_dict.items():
                train, test, val, overfit = {}, {}, {}, {}
                for fold, data in task_dict.items():
                    for overfit_file in remote_glob(os.path.join(
                        run_dir,
                        f"{framework}_{benchmark}_{constraint}_{task}_{fold}",
                        '*',  # The directory name the benchmark created
                        'overfit',
                        task,
                        str(fold),
                        'overfit.csv'
                    )):
                        overfit_file = remote_get(overfit_file)
                        frame = pd.read_csv(overfit_file, index_col=0)
                        for index, row in frame.iterrows():
                            row_dict = row.to_dict()

                            # We care about the best individual model and ensemble
                            if row_dict['model'] not in ['best_individual_model', 'best_ensemble_model']:
                                continue

                            # Collect data
                            if row_dict['model'] not in train:
                                train[row_dict['model']] = []
                            train[row_dict['model']].append(row_dict['train'])
                            if row_dict['model'] not in val:
                                val[row_dict['model']] = []
                            val[row_dict['model']].append(row_dict['val'])
                            if row_dict['model'] not in test:
                                test[row_dict['model']] = []
                            test[row_dict['model']].append(row_dict['test'])
                            if row_dict['model'] not in overfit:
                                overfit[row_dict['model']] = []

                            if row_dict['model'] == 'best_individual_model':
                                key = 'best_ever_test_score_individual_model'
                            elif row_dict['model'] == 'best_ensemble_model':
                                key = 'best_ever_test_score_ensemble_model'
                            else:
                                raise NotImplementedError(row_dict['model'])
                            overfit[row_dict['model']].append(
                                frame[frame['model'] == key]['test'] - row_dict['test'])
                for model in train.keys():
                    row_dict = {
                        'tool': framework,
                        'task': task,
                        'metric': metric,
                        'model': model,
                        'train_mean':  np.nanmean(train[model]) if np.any(train[model]) else 0,
                        'val_mean':  np.nanmean(val[model]) if np.any(val[model]) else 0,
                        'test_mean':  np.nanmean(test[model]) if np.any(test[model]) else 0,
                        'overfit_mean':  np.nanmean(overfit[model]) if np.any(overfit[model]) else 0,
                        'train_std':  np.nanstd(train[model]) if np.any(train[model]) else 0,
                        'val_std':  np.nanstd(val[model]) if np.any(val[model]) else 0,
                        'test_std':  np.nanstd(test[model]) if np.any(test[model]) else 0,
                        'overfit_std':  np.nanstd(overfit[model]) if np.any(overfit[model]) else 0,
                    }
                    dataframe.append(row_dict)
                    if model == 'best_ensemble_model':
                        all_test_scores.extend(test[model])
                        all_overfits.extend(overfit[model])
    print(f"Average Test Score = {np.mean(all_test_scores)} +- {np.std(all_test_scores)}")
    print(f"Average Overfit = {np.mean(all_overfits)} +- {np.std(all_overfits)}")

    return pd.DataFrame(dataframe)


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
    metric = get_metric_from_run_area(run_dir)
    dataframe = []
    for framework, framework_dict in jobs.items():
        for benchmark, benchmark_dict in framework_dict.items():
            for task, task_dict in benchmark_dict.items():
                for fold, data in task_dict.items():

                    if 'autosklearn' not in framework.lower() or True:
                        result = jobs[framework][benchmark][task][fold]['results']
                        if is_number(result):
                            if not args.multiclass_metric == "['logloss', 'acc']":
                                result = abs(result)
                        else:
                            result = np.nan
                        dataframe.append({
                            'tool': framework if not args.tag_with_constraint else f"{framework}_{constraint}",
                            'task': task,
                            'model': 'best_ensemble_model',
                            'fold': fold,
                            'train':  np.nan,
                            'val':  np.nan,
                            # assuming balacc in metric
                            'test':  result,
                            'overfit':  np.nan,
                            'constraint': constraint,
                        })
                        continue

                    for overfit_file in remote_glob(os.path.join(
                        run_dir,
                        f"{framework}_{benchmark}_{constraint}_{task}_{fold}",
                        '*',  # The directory name the benchmark created
                        'overfit',
                        task,
                        str(fold),
                        'overfit.csv'
                    )):
                        overfit_file = remote_get(overfit_file)
                        frame = pd.read_csv(overfit_file, index_col=0)
                        for index, row in frame.iterrows():
                            row_dict = row.to_dict()

                            # We care about the best individual model and ensemble
                            if row_dict['model'] not in ['best_individual_model', 'best_ensemble_model'] and 'best_ensemble_model' not in row_dict['model']:
                                continue
                            model = row_dict['model']

                            train = row_dict['train']
                            val = row_dict['val']
                            test = row_dict['test']
                            if row_dict['model'] == 'best_individual_model':
                                key = 'best_ever_test_score_individual_model'
                            elif 'best_ensemble_model' in row_dict['model']:
                                key = row_dict['model']
                            elif row_dict['model'] == 'best_ensemble_model':
                                key = 'best_ever_test_score_ensemble_model'
                            else:
                                raise NotImplementedError(row_dict['model'])
                            overfit = float(frame[frame['model'] == key]['test'].iloc[0] - row_dict['test'])
                            dataframe.append({
                                'tool': framework,
                                'task': task,
                                'model': model,
                                'fold': fold,
                                'train':  train,
                                'val':  val,
                                'test':  test,
                                'overfit':  overfit,
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


def collect_performance_history(
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
                    rh_dataframe = []
                    for run_history_file in remote_glob(os.path.join(
                        run_dir,
                        f"{framework}_{benchmark}_{constraint}_{task}_{fold}",
                        '*',  # The directory name the benchmark created
                        'debug',
                        task,
                        str(fold),
                        'smac3-output',
                        '*', #_run_SEED
                        'runhistory.json'
                    )):
                        run_history_file = remote_get(run_history_file)
                        with open(run_history_file) as f:
                            runhistory = json.load(f)

                        for run in runhistory['data']:
                            if 'SUCCESS' not in str(run[1][2]):
                                continue

                            # Alternatively, it is possible to also obtain the start time with ``run_value.starttime``
                            endtime = pd.Timestamp(time.strftime('%Y-%m-%d %H:%M:%S',
                                                                 time.localtime(run[1][4])))
                            val_score = 1.0 - run[1][0]
                            test_score = 1.0 - run[1][5]['test_loss']
                            train_score = 1.0 - run[1][5]['train_loss']
                            rh_dataframe.append({
                                'Timestamp': endtime,
                                'single_best_optimization_score': val_score,
                                'single_best_test_score': test_score,
                                'single_best_train_score': train_score,
                                'task': task,
                                'fold': fold,
                                'framework': framework,
                            })
                    rh_dataframe = pd.DataFrame(rh_dataframe)
                    ensemble_dataframe = []
                    for ensemble in remote_glob(os.path.join(
                        run_dir,
                        f"{framework}_{benchmark}_{constraint}_{task}_{fold}",
                        '*',  # The directory name the benchmark created
                        'debug',
                        task,
                        str(fold),
                        '.auto-sklearn',
                        'ensemble_history.json',
                    )):
                        ensemble_file = remote_get(ensemble)
                        with open(ensemble_file) as f:
                            ensemble = json.load(f)

                        timestamps = [t for t in ensemble['Timestamp'].values()]
                        ensemble_optimization_scores = [t for t in ensemble['ensemble_optimization_score'].values()]
                        ensemble_test_score = [t for t in ensemble['ensemble_test_score'].values()]
                        for t, opt, test in zip(timestamps, ensemble_optimization_scores, ensemble_test_score):
                            ensemble_dataframe.append({
                                'Timestamp': pd.Timestamp(t, unit='ms'),
                                'ensemble_optimization_score': opt,
                                'ensemble_test_score': test,
                                'task': task,
                                'fold': fold,
                                'framework': framework,
                            })
                    ensemble_dataframe = pd.DataFrame(ensemble_dataframe)
                    dataframe.append(
                        pd.merge(
                            rh_dataframe,
                            ensemble_dataframe,
                            on="Timestamp", how='outer'
                        ).sort_values('Timestamp').fillna(method='ffill')
                    )

    return pd.concat(dataframe).reset_index(drop=True)


def collect_stats(
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
                    rh_dataframe = []
                    for stats_file in remote_glob(os.path.join(
                        run_dir,
                        f"{framework}_{benchmark}_{constraint}_{task}_{fold}",
                        '*',  # The directory name the benchmark created
                        'debug',
                        task,
                        str(fold),
                        'smac3-output',
                        '*', #_run_SEED
                        'stats.json'
                    )):
                        stats_file = remote_get(stats_file)
                        with open(stats_file) as f:
                            stats = json.load(f)

                        dataframe.append({
                            'framework': framework,
                            'task': task,
                            'fold': fold,
                            "submitted_ta_runs": stats['submitted_ta_runs'],
                            "finished_ta_runs": stats['finished_ta_runs'],
                            "n_configs": stats['n_configs'],
                            "wallclock_time_used": stats['wallclock_time_used'],
                            "ta_time_used": stats['ta_time_used'],
                            "inc_changed": stats['inc_changed'],
                            "_n_configs_per_intensify": stats['_n_configs_per_intensify'],
                            "_n_calls_of_intensify": stats['_n_calls_of_intensify'],
                            "_ema_n_configs_per_intensifiy": stats['_ema_n_configs_per_intensifiy'],
                            "_EMA_ALPHA": stats['_EMA_ALPHA'],
                        })

    return pd.DataFrame(dataframe)


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
            print(f"Error while parsing {run_file}.. No model file found!")
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
        print(framework_models[framework])
        framework_models[framework].to_csv(f"{framework}_models.csv", index=False)
    return framework_models


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
        choices=list(get_framework_information().keys()),
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
        #default="['acc']",
        default="['balacc', 'auc', 'acc']",
        #default="['auc', 'logloss', 'acc']",
        help='What metric set to use. Notice that it has to be a string'
    )
    parser.add_argument(
        '--multiclass_metric',
        default="['balacc', 'logloss', 'acc']",
        #default="['acc']",
        #default="['logloss', 'acc']",
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
        '--collect_overhead',
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help='generates a overhead'
    )
    parser.add_argument(
        '--collect_history',
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help='generates a problems dataframe'
    )
    parser.add_argument(
        '--collect_stats',
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
        '--tag_with_constraint',
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help='generates a models used dataframe'
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
        f"rsync --update -avzhP --exclude '*/venv/*' --exclude '*/lib/*' -e \"ssh -p 22 -A {USER}@132.230.166.39 ssh\" {os.getcwd()}/* {USER}@{INTERNAL}:{AUTOMLBENCHMARK}",
        shell=True,
        stdout=subprocess.PIPE
    ).stdout.decode('utf-8')
    print(f"updated_files = {updated_files}")
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
        constraint = f"{args.runtime}s{args.cores}c{args.memory}"
        filename = f"{args.framework}_overfit.csv" if args.tag_with_constraint is None else f"{args.framework}_{constraint}_overfit.csv"
        logger.info(f"Please check {filename}")
        overfit.to_csv(filename)

    if args.collect_history:
        overfit = collect_performance_history(
            jobs=jobs,
            args=args,
            run_dir=run_dir
        )
        filename = f"{args.framework}_history.csv"
        logger.info(f"Please check {filename}")
        overfit.to_csv(filename)

    if args.collect_stats:
        stats = collect_stats(
            jobs=jobs,
            args=args,
            run_dir=run_dir
        )
        filename = f"{args.framework}_stats.csv"
        logger.info(f"Please check {filename}")
        stats.to_csv(filename)

    if args.collect_overhead:
        overhead = collect_overhead(
            jobs=jobs,
            args=args,
            run_dir=run_dir
        )
        filename = f"{args.framework}_overhead.csv"
        logger.info(f"Please check {filename}")
        overhead.to_csv(filename)

    # Close the connection to the server
    SSH.close()
    VM.close()
