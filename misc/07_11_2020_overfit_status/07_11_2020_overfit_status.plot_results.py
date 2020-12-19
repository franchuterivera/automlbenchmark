import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# ====================================================================
#                               Functions
# ====================================================================
def parse_data(csv_location: str) -> pd.DataFrame:
    """
    Collects the data of a given experiment, annotated in a csv.
    we expect the csv to look something like:
    (index),tool,task,model,fold,train,val,test,overfit
    """
    data = []
    for data_file in glob.glob(os.path.join(csv_location, '*.csv')):
        data.append(
            pd.read_csv(
                data_file,
                index_col=0,
            )
        )

    data = pd.concat(data).reindex()

    # Only plot ensemble for now
    model = 'best_ensemble_model'
    data = data[data['model'] == model]

    # Make sure our desired columns are numeric
    data['test'] = pd.to_numeric(data['test'])
    data['overfit'] = pd.to_numeric(data['overfit'])

    # then we want to fill in the missing values
    all_tools = [t for t in data['tool'].unique().tolist()]
    num_rows = [len(df[df['tool'] == t].index) for t in all_tools]
    tool_with_more_rows = all_tools[np.argmax(num_rows)]
    required_columns = ['task', 'model', 'fold']

    # There is a function called isin pandas, but it gives
    # wrong results -- do this fill in manually
    # base df has all the task/fold/models in case one is missing, like for a crash
    base_df = data[data['tool'] == tool_with_more_rows][required_columns].reset_index(drop=True)
    for tool in list(set(all_tools) - {tool_with_more_rows}):
        fix_df = data[data['tool'] == tool][required_columns].reset_index(drop=True)

        # IsIn from pandas does it base on the index. We need to unstack/stack values
        # for real comparisson
        missing_rows = base_df.iloc[base_df[~base_df.stack(
        ).isin(fix_df.stack().values).unstack()].dropna(how='all').index]
        missing_rows['tool'] = tool
        data = pd.concat([data, missing_rows], sort=True).reindex()

    # A final sort
    data = data.sort_values(by=['tool']+required_columns).reset_index(drop=True)

    return data


def plot_relative_performance(df: pd.DataFrame, tools: typing.List[str],
                              metric: str = 'test') -> None:
    """
    Generates a relative performance plot, always compared to
    autosklearn.
    """

    if 'autosklearn' not in df['tool'].tolist():
        raise ValueError('We need autosklearn in the dataframe to compare')

    if tool not in df['tool'].tolist():
        raise ValueError(f"Experiment {tool} was not found in the dataframe {df['tool']}")

    # Get the desired frames
    autosklearn_df = df[df['tool'] == 'autosklearn'].reset_index(drop=True)
    desired_df = df[df['tool'] == tool].reset_index(drop=True)
    desired_df[metric] = desired_df[metric].subtract(autosklearn_df[metric])

    for tool in tools:
        # make sure autosklearn is in the data
        sns.set_style("whitegrid")
        ax = sns.lineplot(
            'task',
            metric,
            data=desired_df,
            ci='sd',
            palette=sns.color_palette("Set2"),
            err_style='band',
        )

    plt.show()


def printsomethind():
    experiment_results = {}
    for tool_task, test_value in data.groupby(['tool', 'task']).mean()['test'].to_dict().items():
        tool, task = tool_task
        if tool not in experiment_results:
            experiment_results[tool] = {}
        if task not in experiment_results[tool]:
            experiment_results[tool][task] = test_value

    summary = []
    for tool in experiment_results:
        row = experiment_results[tool]
        row['tool'] = tool
        summary.append(row)

    summary = pd.DataFrame(summary)
    print(summary)

    # The best per task:
    for task in [c for c in summary.columns if c != 'tool']:
        best = summary[task].argmax()
        print(f"{task}(best) = {summary['tool'].iloc[best]}")

    # How many times better than autosklearn
    summary_no_tool_column = summary.loc[:, summary.columns != 'tool']
    baseline_results = summary[summary['tool']=='autosklearn'].loc[:, summary[summary['tool']=='autosklearn'].columns != 'tool']
    for index, row in summary.iterrows():
        tool = row['tool']
        if tool == 'autosklearn': continue
        print(f"{tool} (better_than_baseline): {np.count_nonzero(summary_no_tool_column.iloc[index] > baseline_results)}")


 if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Utility to plot CSV results')
    parser.add_argument(
        '--csv_location',
        help='Where to look for csv files',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--experiment',
        action='append',
        help='Generates results/plots just for this experiment',
        default=None,
        type=str,
        required=False,
    )

    # First get the data
    df = parse_data(args.csv_location)
