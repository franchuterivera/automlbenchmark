import os
import pandas as pd

from argparse import ArgumentParser
import json
import re

parser = ArgumentParser()
parser.add_argument(
    '--baseline_dir',
    help='Where to look for baseline files',
    type=str,
    action='extend',
    nargs='+',
)
parser.add_argument(
    '--qa_dir',
    help='Where to look for baseline files',
    type=str,
    action='extend',
    nargs='+',
)
args = parser.parse_args()


def collect_row_for_run(directory, seed):
    """
    A row to collect looks like a dictionary with:
    {task, fold, seed, model, best_individual_val, best_individual_test, best_ensemble_val, best_ensemble_test}
    """
    with open(os.path.join(directory, 'metadata.json')) as json_file:
        metadata = json.load(json_file)
    regularex = re.compile(r'debug/([^/]+)/(\d)/')
    rows = []
    for run_history in glob.glob(directory, '*/*/debug/*/*/smac3-output/run_*/runhistory.json'):
        match = regularex.search(run_history)
        if match is None:
            raise ValueError(run_history)
        task = match.group(1)
        fold = match.group(2)
        framework = metadata['framework']

        with open(run_history) as json_file:
            rh = json.load(json_file)

        df = []
        for element in rh['data']:
            run_key = element[0]
            run_value = element[1]
            cost, duration, status, start, end, info = run_value
            if info is not None and 'num_run' in info:
                row = {'num_run': info['num_run'], 'Validation': 1 - cost, 'Test': 1-info['test_loss']}
                if 'repeats' in run_key[1]:
                    instance = eval(run_key[1])
                    row['repeats'] = instance['repeats']
                    row['level'] = instance['level']
                df.append(row)
        df = pd.DataFrame(df)
        best_individual_val = df.loc[df['Validation']==df['Validation'].max(), 'validation']
        best_individual_test = df.loc[df['Validation']==df['Validation'].max(), 'Test']

        ensemble_json = os.path.join(os.path.dirame(os.path.dirname(os.path.dirname(run_history))), '.auto-sklearn/ensemble_history.json')
        with open(ensemble_json) as json_file:
            eh = json.load(json_file)
        df = pd.DataFrame(eh)
        best_ensemble_val = df.iloc[[-1]['ensemble_optimization_score']
        best_ensemble_test = df.iloc[[-1]['ensemble_test_score']
        rows.append({
            'task': task,
            'fold': fold,
            'framework': framework,
            'seed': seed,
            'best_individual_val': best_individual_val,
            'best_individual_test': best_individual_test,
            'best_ensemble_val': best_ensemble_val,
            'best_ensemble_test': best_ensemble_test,
        })
    return rows

run_data = []
for i, (base_dir, qa_dir) in zip(args.baseline_dir, args.qa_dir):
    run_data.extend(collect_row_for_run(base_dir, seed=i))
    run_data.extend(collect_row_for_run(qa_dir, seed=i))
df = pd.DataFrame(run_data)
print(df)
