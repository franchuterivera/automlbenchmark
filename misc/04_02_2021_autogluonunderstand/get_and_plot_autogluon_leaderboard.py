from scipy.stats import rankdata
import tqdm
import argparse
import glob
import logging
import typing
import os

import pandas as pd

import networkx as nx

import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import wilcoxon

import seaborn as sns

logger = logging.getLogger()

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
    for data_file in glob.glob(os.path.join(csv_location, '*out')):
        # File name is AutoGluon_master_thesis_28800s1c8G_cnae-9_0_2021.02.04-00.25.17.out
        auto, master, thesis, constraint, task, fold, time = os.path.basename(data_file).split('_')
        this_df = pd.read_csv(
            data_file,
            index_col=0,
        )
        this_df['task'] = task
        this_df['fold'] = fold
        data.append(this_df)

    if len(data) == 0:
        logger.error(f"No overfit data to parse on {csv_location}")
        return

    data = pd.concat(data).reindex()

    for num_col in ['score_test', 'score_val']:
        data[num_col] = pd.to_numeric(data[num_col])

    # Remove _L# from the model name
    data['model'] = data['model'].map(lambda x: str(x)[:-3])

    # Build a new dataframe which is the product of L1-L2
    level0 = data[data['stack_level'] == 0].set_index(['model', 'task', 'fold'])
    level1 = data[data['stack_level'] == 1].set_index(['model', 'task', 'fold'])
    result = level1 - level0

    # Set it in the expected format
    #  variable  performance
    # model(index)          variable  performance
    # CatBoost_BAG          score_test     0.027778
    # atBoost_BAG           score_val     0.016461

    return result.reset_index().melt(
        id_vars=['model', 'task', 'fold'],
        value_name='performance',
        value_vars=['score_test', 'score_val'])


def collapse_seed_fold(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapses a dataframe that has multiple runs per seed
    """
    # Collapse the seed
    return df.groupby(
        ['model', 'variable', 'task']
    ).mean().reset_index()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Utility to plot CSV results')
    parser.add_argument(
        '--csv_location',
        help='Where to look for csv files',
        type=str,
        action='extend',
        nargs='+',
    )
    args = parser.parse_args()

    # First get the data from the log files
    dfs = []
    for i, csv_location in enumerate(args.csv_location):
        logger.info(f"Working on {i}: {csv_location}")
        df = parse_data(csv_location)

        if 'seed' not in df.columns:
            df['seed'] = i
        elif 'block_Aseed_fold_dataset' in args.rank:
            # We are comparing a run that is somewhat paired.
            # From autosklearn we created 3 seeds from which we then
            # run 10 times each
            # Aseed is the Autosklearn father seed
            df['Aseed'] = i

        dfs.append(df)

    df = pd.concat(dfs).reset_index()
    df.to_csv('before_collapse.csv')
    df = collapse_seed_fold(df)
    df.to_csv('after_collapse.csv')

    # plot
    g = sns.FacetGrid(df, col="task", col_wrap=2)
    g.map_dataframe(sns.barplot, y="model", x="performance", hue='variable', palette="colorblind")
    g.set_axis_labels("Model", "Balanced Accuracy L1-L0")
    g.add_legend()
    #plt.savefig(f"{_repeats.pdf")
    plt.show()
