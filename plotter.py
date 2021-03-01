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
MAPPING = {
    "None": 'autosklearn',
    "autosklearnBBCEnsembleSelection": 'Alg. 5',
    "autosklearnBBCEnsembleSelectionNoPreSelect": 'Alg. 5 (no Line 12.)',
    "autosklearnBBCEnsembleSelectionPreSelectInES": 'Alg. 7',
    "autosklearnBBCEnsembleSelectionPreSelectInESFULL": 'Alg. 6',
    "autosklearnBBCSMBOAndEnsembleSelectionBISMAC": 'Alg. 11',
    "autosklearnBBCSMBOAndEnsembleSelection": 'Alg. 13',
    "autosklearnBBCScoreEnsemble": 'Alg. 9',
    "autosklearnBBCScoreEnsembleMAX": 'Alg. 10',
    "autosklearnBBCScoreEnsembleMAXWinner": 'Alg. 11',
    "autosklearnBBCEnsembleSelectionPreSelectInESRegularizedEnd": 'Alg. 12',
    "autosklearnBBCScoreEnsembleAVGMDEV": 'Alg. 13',
    "autosklearnBagging": 'bagging',
    "autosklearnStacking": 'Alg. 14',
    "autosklearnThresholdout": 'Alg. 17',
}


def parse_data(csv_location: str) -> pd.DataFrame:
    """
    Collects the data of a given experiment, annotated in a csv.
    we expect the csv to look something like:
    (index),tool,task,model,fold,train,val,test,overfit
    """
    data = []
    for data_file in glob.glob(os.path.join(csv_location, '*overfit.csv')):
        data.append(
            pd.read_csv(
                data_file,
                index_col=0,
            )
        )

    if len(data) == 0:
        logger.error(f"No overfit data to parse on {csv_location}")
        return

    data = pd.concat(data).reindex()

    # Only plot ensemble for now
    model = 'best_ensemble_model'
    data = data[data['model'] == model]

    # Make sure our desired columns are numeric
    data['test'] = pd.to_numeric(data['test'])
    data['overfit'] = pd.to_numeric(data['overfit'])

    # then we want to fill in the missing values
    all_tools = [t for t in data['tool'].unique().tolist()]
    num_rows = [len(data[data['tool'] == t].index) for t in all_tools]
    tool_with_more_rows = all_tools[np.argmax(num_rows)]
    required_columns = ['task', 'model', 'fold']

    # There is a function called isin pandas, but it gives
    # wrong results -- do this fill in manually
    # base df has all the task/fold/models in case one is missing, like for a crash
    #base_df = data[data['tool'] == tool_with_more_rows][required_columns].reset_index(drop=True)
    base_df = data.loc[data['tool'] == tool_with_more_rows, required_columns].reset_index(drop=True)
    for tool in list(set(all_tools) - {tool_with_more_rows}):
        fix_df = data.loc[data['tool'] == tool, required_columns].reset_index(drop=True)

        # IsIn from pandas does it base on the index. We need to unstack/stack values
        # for real comparisson
        #missing_rows = base_df.iloc[base_df[~base_df.stack(
        #).isin(fix_df.stack().values).unstack()].dropna(how='all').index]
        mask = base_df.stack().isin(fix_df.stack().values).unstack()
        missing_rows = base_df.iloc[base_df[~mask].dropna(how='all').index].copy()
        missing_rows['tool'] = tool
        data = pd.concat([data, missing_rows], sort=True).reindex()

    # A final sort
    data = data.sort_values(by=['tool']+required_columns).reset_index(drop=True)

    return data


def parse_overhead(csv_location, tools=None, keys=['MaxRSS',
                                                   'MaxVMSize',
                                                   'SMACModelsSUCCESS',
                                                   'SMACModelsALL']):
    """
    Collects the data of a given experiment, annotated in a csv.
    we expect the csv to look something like:
    Index(['tool', 'task', 'fold', 'JobID', 'JobName', 'MaxRSS', 'Elapsed',
       'MaxDiskRead', 'MaxDiskWrite', 'MaxVMSize', 'MinCPU', 'TotalCPU',
       'SMACModelsSUCCESS', 'SMACModelsALL'],
      dtype='object')
    """
    df = []
    for df_file in glob.glob(os.path.join(csv_location, '*overhead.csv')):
        df.append(
            pd.read_csv(
                df_file,
                index_col=0,
            )
        )

    if len(df) == 0:
        logger.error(f"No overhead data to parse on {csv_location}")
        return

    df = pd.concat(df).reset_index(drop=True)

    # Convert the K, M into a number so we can do stuff
    for column in list(set.intersection({'MaxRSS', 'MaxDiskRead', 'MaxDiskWrite', 'MaxVMSize'}, set(keys))):
        df[column] = df[column].replace(r'[KM]+$', '', regex=True).astype(float)

    # Convert time to a number of seconds
    for column in list(set.intersection({'Elapsed', 'MinCPU', 'TotalCPU'}, set(keys))):
        df[column] = pd.to_timedelta(['00:' + x if x.count(':') < 2 else x for x in df[column]]).total_seconds()

    # Collapse by fold
    data = []
    for key in keys:
        data.append(
            df[['tool', 'task', 'fold', key]].groupby(['task', 'tool']).mean().unstack().drop('fold', 1)
        )
    return pd.concat(data, axis='columns')


def plot_relative_performance(df: pd.DataFrame, tools: typing.List[str],
                              metric: str = 'test', output_dir: typing.Optional[str] = None,
                              ) -> None:
    """
    Generates a relative performance plot, always compared to
    autosklearn.
    """

    if 'autosklearn' not in df['tool'].tolist():
        raise ValueError('We need autosklearn in the dataframe to compare')

    if any([tool not in df['tool'].tolist() for tool in tools]):
        raise ValueError(f"Experiment {tools} was not found in the dataframe {df['tool']}")

    # Get the desired frames
    autosklearn_df = df[df['tool'] == 'autosklearn'].reset_index(drop=True)

    for tool in tools:
        desired_df = df[df['tool'] == tool].reset_index(drop=True)
        desired_df[metric] = desired_df[metric].subtract(autosklearn_df[metric])

        # make sure autosklearn is in the data
        sns.set_style("whitegrid")
        sns.lineplot(
            'task',
            metric,
            data=desired_df,
            ci='sd',
            palette=sns.color_palette("Set2"),
            err_style='band',
            label=tool,
        ).set_title(f"Relative {metric} metric against Autosklearn")

    plt.legend()
    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, '_'.join(tools) + '.pdf'))

    plt.show()


def plot_ranks(df: pd.DataFrame,
               metric: str = 'test', output_dir: typing.Optional[str] = None,
               ) -> None:
    """
    Generates a relative performance plot, always compared to
    autosklearn.
    """

    if 'autosklearn' not in df['tool'].tolist():
        raise ValueError('We need autosklearn in the dataframe to compare')

    # Step 1: Calculate the mean and std of each fold,
    # That is, collapse the folds
    df = df.groupby(['tool', 'model', 'task']).mean().add_suffix('_mean').reset_index()
    df.to_csv('debug.csv')

    # Sadly the rank method of group by gives weird result, so rank manually
    df['rank'] = 0
    df['seed'] = 0
    for task in df['task'].unique():
        df.loc[df['task'] == task, 'rank'] = df.loc[df['task'] == task, metric + '_mean'].rank()

    # make sure autosklearn is in the data
    sns.set_style("whitegrid")
    sns.pointplot(
        'seed',
        'rank',
        data=df,
        ci='sd',
        platte=sns.diverging_palette(250, 30, 65, center="tab10", as_cmap=True),
        dodge=True,
        join=False,
        hue='tool',
    ).set_title(f"Ranking Plot")

    plt.legend(ncol=4, loc='upper center')
    plt.xlim(-1, 1)

    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, 'ranking_plot.pdf'))

    plt.show()


def generate_pairwise_comparisson_matrix(df: pd.DataFrame,
                                         metric: str = 'test',
                                         tools: typing.List = [],
                                         ) -> pd.DataFrame:
    """
    Generates pairwise matrices as detailed in
    https://en.wikipedia.org/wiki/Condorcet_method

    "For each possible pair of candidates, one pairwise count indicates how many voters
    prefer one of the paired candidates over the other candidate, and another pairwise count
    indicates how many voters have the opposite preference.
    The counts for all possible pairs of candidates summarize
    all the pairwise preferences of all the voters."

    """
    # Step 1: Calculate the mean and std of each fold,
    # That is, collapse the folds
    df = df.groupby(['tool', 'model', 'task']).mean().add_suffix('_mean').reset_index()

    #df.to_csv('raw_data.csv')

    # Just care about the provided tools
    if len(tools) < 1:
        tools = sorted(df['tools'].unique())
    df = df[df['tool'].isin(tools)]

    pairwise_comparisson_matrix = pd.DataFrame(data=np.zeros((len(tools), len(tools))),
                                               columns=tools, index=tools)

    for task in df['task'].unique():
        # we use datasets as our voters
        # Then we add each ballot
        df_task = df[df['task'] == task]

        # So we create a ballot for this dataset
        this_pairwise_comparisson_matrix = pd.DataFrame(
            data=np.zeros((len(tools), len(tools))),
            columns=tools, index=tools
        )

        for candidate in tools:
            # Every row in the matrix dictates over who the current tool
            # wins in performance. Equal performance (likely diagonal)
            # should get zero
            candidate_performance = df_task.loc[df_task['tool'] == candidate,
                                                metric + '_mean'].to_numpy().item(0)

            for opponent in tools:

                # Do not update same performance
                if opponent == candidate:
                    continue

                opponent_performance = df_task.loc[df_task['tool'] == opponent,
                                                   metric + '_mean'].to_numpy().item(0)

                # We update the this_pairwise_comparissin_matrix cells of the current tool
                # that have a better performance than the current tool
                if not np.isnan(candidate_performance) and np.isnan(opponent_performance):
                    # If we have a NaN as opponent, we always win
                    this_pairwise_comparisson_matrix.loc[candidate, opponent] = 1
                elif metric == 'test' and candidate_performance > opponent_performance:
                    this_pairwise_comparisson_matrix.loc[candidate, opponent] = 1
                elif metric == 'overfit' and candidate_performance < opponent_performance:
                    this_pairwise_comparisson_matrix.loc[candidate, opponent] = 1
                elif metric not in ['test', 'overfit']:
                    raise NotImplementedError(metric)

        pairwise_comparisson_matrix = pairwise_comparisson_matrix.add(
            this_pairwise_comparisson_matrix,
            fill_value=0
        )
    return pairwise_comparisson_matrix


def beautify_node_name(name: str, separator=' ') -> str:
    """
    Just makes a name nicer to plot in graph viz
    """
    # Some pre checks
    if 'None_' in name: return 'autosklearn'
    if 'bagging_' in name: return 'bagging'
    # Mapping hold translations to make plotting better
    for key in sorted(MAPPING.keys(), key=len, reverse=True):
        if key in name:
            name = name.replace(key, MAPPING[key])
            break
    name = name.replace("_", separator)
    return name


def get_sorted_locked_winner_losser_tuples(pairwise_comparisson_matrix: pd.DataFrame
                                           ) -> typing.List[typing.Tuple[str, str]]:
    """
    Implements the sort and lock step of
    https://en.wikipedia.org/wiki/Ranked_pairs

    Sort:
    The pairs of winners, called the "majorities", are then sorted from the largest majority to the smallest majority. A majority for x over y precedes a majority for z over w if and only if one of the following conditions holds:

Vxy > Vzw. In other words, the majority having more support for its alternative is ranked first.
Vxy = Vzw and Vwz > Vyx. Where the majorities are equal, the majority with the smaller minority opposition is ranked first.[vs 1]

    Lock
    The next step is to examine each pair in turn to determine the pairs to "lock in".

    Lock in the first sorted pair with the greatest majority.
    Evaluate the next pair on whether a Condorcet cycle occurs when this pair is added to the locked pairs.
    If a cycle is detected, the evaluated pair is skipped.
    If a cycle is not detected, the evaluated pair is locked in with the other locked pairs.
    Loop back to Step #2 until all pairs have been exhausted.
    """

    pairs = []
    for candidate, row in pairwise_comparisson_matrix.iterrows():
        for opponent in pairwise_comparisson_matrix.columns:
            if opponent == candidate:
                next
            pair = (beautify_node_name(candidate, separator='\n'),
                    beautify_node_name(opponent, separator='\n'))
            # we count positive votes first
            number_of_votes = row[opponent]
            number_of_oposition = pairwise_comparisson_matrix.loc[opponent, candidate]

            # We ask for number_of_vote > 0 so that we do not double count pairs
            # that is, it is the same candidate, opponent than opponent, candidate
            winner = pairwise_comparisson_matrix.loc[
                candidate, opponent] > pairwise_comparisson_matrix.loc[
                    opponent, candidate]
            if number_of_votes > 0 and winner:
                pairs.append((pair, number_of_votes, number_of_oposition))

    # Sort the list
    sorted_pairs = sorted(pairs, key=lambda x: x[1])
    return [pair for pair, v, o, in sorted_pairs]


def plot_winner_losser_barplot(pairwise_comparisson_matrix: pd.DataFrame) -> None:
    """
    Make a plot to visualize winner and losers
    """
    # Plot also winners and losers
    plot_frame = pairwise_comparisson_matrix.copy().reset_index()

    # Beautify Name
    plot_frame['index'] = plot_frame['index'].apply(lambda x: beautify_node_name(x))

    winners = plot_frame.sum(axis=1)
    total = (len(df['task'].unique() ) - 1) * (pairwise_comparisson_matrix.shape[0])
    lossers = total - winners
    plot_frame['counts'] = winners
    plot_frame['type'] = 'winners'
    plot_frame_lossers = plot_frame.copy()
    plot_frame_lossers['counts'] = lossers
    plot_frame_lossers['type'] = 'losses'

    order = plot_frame.sort_values(by=['counts'])['index'].to_list()

    #sns.set_theme()
    sns.set_style("whitegrid")
    sns.set(font_scale=1.05)
    plot = sns.catplot(
        #x="index",
        #y="counts",
        x="counts",
        y="index",
        hue='type',
        kind='bar',
        order=order,
        data=plot_frame.append(plot_frame_lossers, ignore_index=True),
        palette=sns.color_palette("tab10"),
        legend=False,
    )

    plot.despine(left=True)
    plt.legend(loc='lower right')
    plt.show()


def save_pairwise_to_disk(pairwise_comparisson_matrix: pd.DataFrame) -> None:
    """
    Save to disk after nice formatting
    """
    df = pairwise_comparisson_matrix.copy().reset_index()
    df['index'] = df['index'].apply(lambda x: beautify_node_name(x))
    df.set_index('index')
    df.columns = [beautify_node_name(c) for c in df.columns]
    df.to_csv('pairwise_comparisson_matrix.csv')


def plot_ranked_pairs_winner(df: pd.DataFrame,
                             metric: str = 'test',
                             tools: typing.List = [],
                             output_dir: typing.Optional[str] = None) -> None:
    """
    Implements the voting mechanism from
    https://en.wikipedia.org/wiki/Ranked_pairs.
    """

    # Tally: To tally the votes, consider each voter's preferences.
    # For example, if a voter states "A > B > C" (A is better than B, and B is better than C),
    # the tally should add one for A in A vs. B, one for A in A vs. C, and one for B
    # in B vs. C. Voters may also express indifference (e.g., A = B), and unstated candidates
    # are assumed to be equal to the stated candidates.
    pairwise_comparisson_matrix = generate_pairwise_comparisson_matrix(
        df, metric, tools
    )
    save_pairwise_to_disk(pairwise_comparisson_matrix)

    plot_winner_losser_barplot(pairwise_comparisson_matrix)

    # Sort and lock
    # a sorted list of locked winners (winner, losser)
    pair_of_wrinners = get_sorted_locked_winner_losser_tuples(pairwise_comparisson_matrix)

    G = nx.DiGraph()
    G.add_edges_from(pair_of_wrinners)

    # Find the source of the DAG
    targets, all_nodes = set(), set()
    for e in G.edges():
        source, target = e
        targets.add(target)
        all_nodes.add(target)
        all_nodes.add(source)
    root = all_nodes - targets
    color_map = ['#FFFFFF' if node not in root else '#1f77b4' for node in G]

    # https://stackoverflow.com/questions/55859493/how-to-place-nodes-in-a-specific-position-networkx
    #for prog in ['neato', 'circo', 'dot', 'fdp']:
    for prog in ['dot']:
        nx.draw(
            G,
            node_size=10000,
            #node_color='#FFFFFF',
            node_color=color_map,
            #node_color='#c3acf2',
            linewidths=2,
            edge_color='black',
            arrowsize=50,
            with_labels=True,
            labels={n: n for n in G.nodes},
            #node_shape='d',
            font_color='#000000',
            font_size=15,
            pos=nx.drawing.nx_agraph.graphviz_layout(
                G,
                prog=prog,
                args='-Grankdir=LR -Gnodesep=305 -Granksep=305 -sep=305'
            )
        )
        ax = plt.gca()  # to get the current axis
        ax.collections[0].set_edgecolor("#000000")

        plt.tight_layout()
        plt.show()
        plt.close()


def plot_testsubtrain_history(csv_location: str, tools: typing.List[str],
                              output_dir: typing.Optional[str] = None) -> None:
    """
    Parses a list of ensemble history files and plot the difference between
    train and test
    """
    dfh = []
    for data_file in glob.glob(os.path.join(csv_location, '*ensemble_history.csv')):
        dfh.append(
            pd.read_csv(
                data_file,
                index_col=0,
            )
        )

    if len(dfh) == 0:
        logger.error(f"No ensemble history data to parse on {csv_location}")
        return

    dfh = pd.concat(dfh).reindex()

    # Data needs to be sorted so that we can subtract, add, etc as everything is ordered
    dfh = dfh.sort_values(by=['tool', 'task', 'fold', 'Timestamp']).reset_index(drop=True)

    if any([tool not in dfh['tool'].tolist() for tool in tools]):
        raise ValueError(f"Experiment {tools} was not found in the dataframe {dfh['tool']}")

    # amount of data and for that we re-build the dataframe because it is easier
    recompleted_desired_df = []
    for tool in tools:
        desired_df = dfh[dfh['tool'] == tool].reset_index(drop=True)
        desired_df['TestMinusTrain'] = desired_df['ensemble_test_score'].subtract(
            desired_df['ensemble_optimization_score'])

        # Make the timestamp the same, as the longest stamp so that
        # sns does everything for us
        # That is, we want to make sure every fold of every task has the same
        for task in desired_df['task'].unique():
            mask = desired_df['task'] == task
            all_folds = desired_df[mask]['fold'].unique().tolist()
            count_folds = [desired_df[mask][desired_df[mask]['fold'] == a].shape[
                0] for a in all_folds]
            argmax = np.argmax(count_folds)
            biggest_fold = all_folds[np.argmax(count_folds)]

            # Make timestamp a range
            time_mask = (desired_df['task'] == task) & (desired_df['fold'] == biggest_fold)
            desired_df.loc[time_mask, 'Timestamp'] = pd.Series(range(count_folds[argmax]), index = desired_df.loc[time_mask, 'Timestamp'].index)

            # So the strategy here is to copy over the biggest fold,
            # and re-place values of other folds into it. So the expectation
            # is that we have the same timestamt and biggest fold will have 1000
            # elements, it is easier to make a copy of this biggest fold data and
            # replace a the rows with that of other fold. This will collapse uncertainty
            # in the rows that only the biggest fold has

            # No need to to do anything for the biggest fold. For the remaining
            # folds, we copy the biggest fold data as a base and overwrite with desired data
            recompleted_desired_df.append(
                desired_df[mask][desired_df[mask]['fold'] == biggest_fold]
            )

            for fold in set(all_folds) - {biggest_fold}:
                base_frame = desired_df[mask][desired_df[mask]['fold'] == biggest_fold
                                              ].reset_index(drop=True)
                base_frame['fold'] = fold
                this_frame = desired_df[mask][desired_df[mask]['fold'] == fold
                                              ].reset_index(drop=True)
                # Copy values from original frame into the base frame that is gonna be
                # used to create a new frame with same number of timestamps
                base_frame.loc[:this_frame['TestMinusTrain'].shape[0], 'TestMinusTrain'
                               ] = this_frame['TestMinusTrain']
                recompleted_desired_df.append(base_frame)

    desired_df = pd.concat(recompleted_desired_df).reset_index(drop=True)

    # make sure autosklearn is in the data
    sns.set_style("whitegrid")
    ordered_tasks = desired_df.task.value_counts().index
    g = sns.FacetGrid(desired_df, col="task", col_wrap=2, sharex=False, sharey=False, hue="tool")
    # height=1.7, aspect=4,)
    g.map(sns.lineplot, 'Timestamp', 'TestMinusTrain', ci='sd',

        palette=sns.color_palette("Set2"),
        err_style='band',
        label=tool,
    )
    g.set(xticks=[])

    plt.subplots_adjust(top=0.9)
    g.fig.suptitle('Test-Train History')

    plt.legend()
    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, '_'.join(tools) + '.pdf'))

    plt.show()


def generate_ranking_per_seed_per_dataset(dfs: typing.List[pd.DataFrame], metric: str = 'test') -> pd.DataFrame:
    """
    Generates a ranking dataframe when seeds and dataset are of paired population.
    That is, seed and dataset is same for a group of configurations to try

    Args:
        dfs (List[pdDataFrame]): A list of dataframes each representing a seed

    Returns:
        pd.DataFrame: with the ranking
    """

    # Collapse each input frame on the fold dimension
    dfs = [df.groupby(['tool', 'model', 'task']
                      ).mean().add_suffix('_mean').reset_index() for df in dfs]

    # Tag each frame with a seed
    for i, df in enumerate(dfs):
        dfs[i]['seed'] = i
        dfs[i].set_index('seed', append=True, inplace=True)

    df = pd.concat(dfs).reset_index()
    df.to_csv('raw_data.csv')

    # Create a ranking for seed and dataset
    result = pd.DataFrame(index=df['tool'].unique())
    score = pd.DataFrame(index=df['tool'].unique())
    for seed in df['seed'].unique():
        for task in df['task'].unique():
            this_frame = df.loc[(df['seed']==seed) & (df['task']==task)].set_index('tool')
            this_frame[f"{seed}_{task}"] = this_frame[metric + '_mean'].rank(
                na_option='bottom',
                ascending=False,
                method='min',
            )
            result[f"{seed}_{task}"] = this_frame[f"{seed}_{task}"]
            score[f"{seed}_{task}"] = this_frame[metric + '_mean']

    result['Avg. Ranking'] = result.mean(axis=1)
    result = result.reset_index()
    result['index'] = result['index'].apply(lambda x: beautify_node_name(x))
    result.set_index('index')
    score = score.reset_index()
    score['index'] = score['index'].apply(lambda x: beautify_node_name(x))
    score.set_index('index')
    return result, score


def generate_ranking_per_fold_per_dataset(df: typing.List[pd.DataFrame], metric: str = 'test') -> pd.DataFrame:
    """
    also rank per fold because even folds can be statistically different

    Args:
        dfs (List[pdDataFrame]): A list of dataframes each representing a seed

    Returns:
        pd.DataFrame: with the ranking
    """
    # Create a ranking for seed and dataset
    scores = pd.pivot_table(df, values='test', index=['task', 'tool'], columns=['fold'])
    scores.to_csv('debug_score_rank.csv')
    result = scores.groupby(level=0).rank(ascending=False, method='average', na_option='bottom').mean(axis='columns').reset_index()

    # Change the format -- test name is changed to 0
    result = pd.pivot_table(result, values=0, index=['tool'], columns=['task'])
    return result, scores


def generate_ranking_per_dataset(dfs: typing.List[pd.DataFrame], metric: str = 'test') -> pd.DataFrame:
    """
    Generates a ranking dataframe when seeds and dataset are of paired population.
    That is, seed and dataset is same for a group of configurations to try

    Args:
        dfs (List[pdDataFrame]): A list of dataframes each representing a seed

    Returns:
        pd.DataFrame: with the ranking
    """
    # Tag each frame with a seed
    for i, df in enumerate(dfs):
        dfs[i]['seed'] = i
    df = pd.concat(dfs).reset_index()

    # Collapse each input frame on the fold dimension
    df = df.groupby(['tool', 'model', 'task']
                      ).mean().add_suffix('_mean').reset_index()

    df.to_csv('raw_data.csv')

    # Create a ranking for seed and dataset
    result = pd.DataFrame(index=df['tool'].unique())
    score = pd.DataFrame(index=df['tool'].unique())
    for task in df['task'].unique():
        this_frame = df.loc[(df['task']==task)].set_index('tool')
        this_frame[f"{task}"] = this_frame[metric + '_mean'].rank(
            na_option='bottom',
            ascending=False,
            method='min',
        )
        logger.debug(f"generate_ranking_per_dataset: task={task} this_frame={this_frame}")
        result[f"{task}"] = this_frame[f"{task}"]
        score[f"{task}"] = this_frame[metric + '_mean']

    result['Avg. Ranking'] = result.mean(axis=1)
    result = result.reset_index()
    result['index'] = result['index'].apply(lambda x: beautify_node_name(x))
    result.set_index('index')

    score = score.reset_index()
    score['index'] = score['index'].apply(lambda x: beautify_node_name(x))
    score.set_index('index')
    return result, score


def bootstrap_wins(list_of_blocks_challenger: typing.List, list_of_blocks_reference: typing.List, bootstrap: int = 300) -> pd.DataFrame:
    """

    """
    wins = 0
    assert len(list_of_blocks_challenger) == len(list_of_blocks_reference,)
    for i in range(bootstrap):
        # First select a block
        block_idx = np.random.choice(list(range(len(list_of_blocks_challenger))))

        # Fault tolerance
        if len(list_of_blocks_challenger[block_idx]) == 0:
            # This run failed
            continue
        challenger = np.random.choice(list_of_blocks_challenger[block_idx])

        # Fault tolerance
        if len(list_of_blocks_reference[block_idx]) == 0:
            # The challenger did not failed but reference did. It is a win!
            wins += 1
            continue
        reference = np.random.choice(list_of_blocks_reference[block_idx])
        if challenger > reference:
            wins += 1
    return wins


def generate_ranking_per_Aseed_fold_per_dataset(df: pd.DataFrame, bootstrap: int = 200
                                                ) -> pd.DataFrame:
    """
    Performs the rank calculation per bootstrap block where a block is a Aseed-fold-dataset pair

    Args:
        dfs (List[pdDataFrame]): A list of dataframes each representing a seed

    Returns:
        pd.DataFrame: with the ranking
    """

    # fix different seeds issue
    # Just to make sure seed are same FOR PANDAS INDEXING
    for seed in df['seed'].unique():
        if seed > 200:
            df.loc[df['seed'] == seed, 'seed'] = seed - 100

    # Form the desired array with raw scores, an average of everything
    result = pd.pivot_table(df, values='test', index=['tool'], columns=['task'], aggfunc=np.mean)
    df['wins'] = 0  # we rank based on wins
    ranking = pd.pivot_table(df, values='wins', index=['tool'], columns=['task'], aggfunc=np.mean)

    # fix the metric -- greater is better in certain cases. This can be done per task

    # Sort values for proper compare -- yet I don;t think it matters
    df.sort_values(by=['tool', 'task', 'fold', 'Aseed', 'seed'], inplace=True)

    test_score_per_task_fold_aseed_tool = pd.pivot_table(df, values='test', index=['task', 'fold', 'Aseed', 'tool'], columns='seed')

    folds = df['fold'].unique()
    aseeds = df['Aseed'].unique()

    for task in tqdm.tqdm(ranking.columns):

        # We divide by 2 because bootstrap is used in 2 placed
        # we have blocks and within each block we have repetitions.
        # we sample with replacemented bootstrap//2 times from a block
        # which in this context is task/fold/autosklearnseed. Within this block
        # are 10 repetitions and we take  bootstrap//2 comparissons
        for boot in range(bootstrap // 2):
            # Pick a random fold, seed
            fold = np.random.choice(folds)
            aseed = np.random.choice(aseeds)
            index = (task, fold, aseed)  # would yield a table with tool as row and repetitions as col

            # Get the values per tool(rows) and columns(10 seed repetitions)
            data = test_score_per_task_fold_aseed_tool.loc[index].to_numpy()
            data[np.isnan(data)] = 0
            # Do sampling with replacement to get tool as rows again, but different
            # permutations of the 10 seeds
            resample_index = np.random.randint(data.shape[1], size=(data.shape[0], bootstrap//2))
            data = np.take_along_axis(data, resample_index, 1)
            wins = pd.Series(
                (rankdata(data, axis=0, method='min')-1).sum(axis=1),
                index=test_score_per_task_fold_aseed_tool.loc[index].index
            )
            # Pretty cool using index: So the index are the tool we are evaluation.
            # if a whole tool failed, it won't even be there and this tool for this
            # task won't be incremented. We add the number of times that for this
            # bootstrap block of fold/seed/task, an experiment was better than other
            ranking[task] = ranking[task] + wins

    result = result.reset_index()
    result['tool'] = result['tool'].apply(lambda x: beautify_node_name(x))
    result.set_index('tool')

    ranking = ranking.rank(axis=0, method='average', ascending=False)
    ranking = ranking.reset_index()
    ranking['tool'] = ranking['tool'].apply(lambda x: beautify_node_name(x))
    ranking.set_index('tool')
    print(ranking)
    return ranking, result


def generate_ranking_per_dataset2(dfs: typing.List[pd.DataFrame], metric: str = 'test') -> pd.DataFrame:
    """
    Generates a ranking dataframe when seeds and dataset are of paired population.
    That is, seed and dataset is same for a group of configurations to try

    Args:
        dfs (List[pdDataFrame]): A list of dataframes each representing a seed

    Returns:
        pd.DataFrame: with the ranking
    """
    # Tag each frame with a seed
    df = dfs[0]

    # Collapse the seed
    df = df.groupby(['tool', 'model', 'task', 'fold']
                      ).mean().add_suffix('_seedmean').reset_index()

    # Collapse the fold
    df = df.groupby(['tool', 'model', 'task']
                      ).mean().add_suffix('_foldmean').reset_index()

    df.to_csv('raw_data.csv')

    # Create a ranking for seed and dataset
    result = pd.DataFrame(index=df['tool'].unique())
    score = pd.DataFrame(index=df['tool'].unique())
    for task in df['task'].unique():
        this_frame = df.loc[(df['task']==task)].set_index('tool')
        this_frame[f"{task}"] = this_frame[metric + '_seedmean_foldmean'].rank(
            na_option='bottom',
            ascending=False,
            method='average',
            #method='dense',
        )
        result[f"{task}"] = this_frame[f"{task}"]
        score[f"{task}"] = this_frame[metric + '_seedmean_foldmean']

    result['Avg. Ranking'] = result.mean(axis=1)
    result = result.reset_index()
    result['index'] = result['index'].apply(lambda x: beautify_node_name(x))
    result.set_index('index')

    score = score.reset_index()
    score['index'] = score['index'].apply(lambda x: beautify_node_name(x))
    score.set_index('index')
    return result, score


def collapse_seed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapses a dataframe that has multiple runs per seed
    """
    # Collapse the seed
    return df.groupby(
        ['tool', 'model', 'task', 'fold']
    ).mean().reset_index()


def get_best_tool_for_task(df_task: pd.DataFrame) -> typing.List[str]:
    """
    Simply gets which is the best tool for a given task by doing a bunch of
    wilcoxon test for greatness
    """
    tools = df_task['tool'].unique().tolist()
    df_task = df_task.sort_values(['seed', 'fold'])
    best = tools.pop()
    problematic = []
    while len(tools) > 0:
        challenger = tools.pop()
        try:
            data = pd.merge(
                df_task[(df_task['tool'] == challenger)],
                df_task[(df_task['tool'] == best)],
                suffixes=['_challenger', '_best'],
                how="inner", on=['seed', 'fold'])
            logger.debug(data[['seed', 'fold', 'tool_challenger', 'test_challenger', 'tool_best', 'test_best']])
            # wilcoxon([6, 7, 8, 9, 10], [1, 2, 3, 4, 5], alternative='greater')
            # WilcoxonResult(statistic=15.0, pvalue=0.03125)
            w_g, p_g = wilcoxon(
                # challenger,
                data['test_challenger'],
                # best,
                data['test_best'],
                alternative='greater'
            )
            w_l, p_l = wilcoxon(
                # challenger,
                data['test_challenger'],
                # best,
                data['test_best'],
                alternative='less'
            )
            w_e, p_e = wilcoxon(
                # challenger,
                data['test_challenger'],
                # best,
                data['test_best'],
                alternative='two-sided'
            )
            if p_g < 0.05:
                logger.debug(f"YES challenger={challenger}({np.mean(df_task[(df_task['tool'] == challenger)]['test'])}) better than {best}({np.mean(df_task[(df_task['tool'] == best)]['test'])}) as p_g={p_g}/p_l={p_l}/p_e={p_e} total={data.shape[0]}")
                best = challenger
                # If there is a new challenger
                # We reset the list of problematic
                # So a problematic tool would be one
                # such that the challenger is so similar to best
                # that we cannot decide which one is best
                problematic = []
            elif p_l < 0.05:
                logger.debug(f"NOT challenger={challenger}({np.mean(df_task[(df_task['tool'] == challenger)]['test'])}) better than {best}({np.mean(df_task[(df_task['tool'] == best)]['test'])}) as p_g={p_g}/p_l={p_l}/p_e={p_e} total={data.shape[0]}")
                # Best is still best
                pass
            else:
                logger.debug(f"PROBLEM challenger={challenger}({np.mean(df_task[(df_task['tool'] == challenger)]['test'])}) better than {best}({np.mean(df_task[(df_task['tool'] == best)]['test'])}) as p_g={p_g}/p_l={p_l}/p_e={p_e} -> {((data['test_challenger'] - data['test_best']) > 0).value_counts()} total={data.shape[0]}")
                # Any other case is problematic
                problematic.append(challenger)
        except ValueError as e:
            # ValueError: zero_method 'wilcox' and 'pratt' do not work if x - y is zero for all elements.
            # Fundamentally X == Y
            logger.debug(f"PROBLEM challenger={challenger}({np.mean(df_task[(df_task['tool'] == challenger)]['test'])}) better than {best}({np.mean(df_task[(df_task['tool'] == best)]['test'])}) as p={e}")
            problematic.append(challenger)

    if len(problematic) > 0:
        problematic.append(best)
        performances = [np.mean(df_task[df_task['tool']==tool]['test']) for tool in problematic]
        problematic = [x for _, x in sorted(zip(performances, problematic))]
        return problematic
    else:
        return [best]


def wilcoxon_averaging(df: pd.DataFrame, contains: typing.Optional[str] = None,
                       base_tool_name: str = 'None_es50_B100_N1.0') -> pd.DataFrame:
    """
    Collapses all seeds/folds to a single test performance result
    per dataset. It adds also 2 columns, which are p-values againts
    autosklearn for a greater-than-wilcoxon test and p-values for
    lesser than the best performing model.

    To determine the best performing model, and ALLvsALL approach is followed,
    in which a paired computation is exhaustively done to find a best model. In
    case of tie, WHAT TO DO??
    """
    # Just care right now for 100 bootstrap data
    if contains is not None:
        df = df[df["tool"].str.contains(contains)]

    # Collapse fold
    #df = df.groupby(
    #    ['tool', 'model', 'task', 'seed']
    #).mean().reset_index()

    dataframe = []
    for task in df['task'].unique():
        logger.debug(f"\n\n\n{task}")
        #if task != 'vehicle':
        #    continue
        best_tool_for_task = get_best_tool_for_task(df[df['task']==task])
        logger.debug(f"For task={task} best_tool_for_task={best_tool_for_task}")
        best = df.query(f"tool == '{best_tool_for_task[-1]}' & task == '{task}'").sort_values(['seed', 'fold'])
        for tool in df['tool'].unique():
            challenger = df.query(f"tool == '{tool}' & task == '{task}'").sort_values(['seed', 'fold'])
            data = pd.merge(best, challenger, how="inner", on=['seed', 'fold'])

            try:
                w, p = wilcoxon(data['test_y'], data['test_x'], alternative='two-sided'),
                if p < 0.05:
                    statistically_eq = False
                else:
                    # We do not have enough information to reject the null hypothesis
                    # But we are interested to highlight the equivalent ones
                    statistically_eq = True
            except ValueError:
                # X == Y
                statistically_eq = True

            dataframe.append({
                'task': task,
                'tool': tool,
                'statistically_eq': statistically_eq,
                'test_avg': data['test_y'].mean(),
                'test_std': data['test_y'].std(),
                'is_best': tool in best_tool_for_task,
            })
    df = pd.DataFrame(dataframe)
    df['tool'] = df['tool'].apply(lambda x: beautify_node_name(x))
    df['task_best'] = False
    for task in df['task']:
        maximum = df[df['task']==task]['test_avg'].max()
        df.loc[(df['task']==task) & (df['test_avg']==maximum), 'task_best'] = True
    df.to_csv('wilcoxon.csv')

    #rearrenge the format
    df_test = pd.DataFrame(index=df['tool'].unique(), columns=df['task'].unique())
    df_best = pd.DataFrame(index=df['tool'].unique(), columns=df['task'].unique())
    df_EQ = pd.DataFrame(index=df['tool'].unique(), columns=df['task'].unique())
    for tool in df['tool'].unique():
        for task in df['task'].unique():
            df_test.at[tool, task] = df[(df['tool']==tool) & (df['task']==task)]['test_avg'].values.item(0)
            print(f"df_test={df_test} with tool={tool} and task={task} and {df[(df['tool']==tool) & (df['task']==task)]}")
            df_EQ.at[tool, task] = df[(df['tool']==tool) & (df['task']==task)]['is_best'].values.item(0)
            df_best.at[tool, task] = df[(df['tool']==tool) & (df['task']==task)]['task_best'].values.item(0)
            print(f"df_best={df_best} ")
    df_test.to_csv('wilcoxon_test.csv')
    df_best.to_csv('wilcoxon_best.csv')
    df_EQ.to_csv('wilcoxon_EQ.csv')

    return df


def generate_ranking_per_fold_per_dataset_bags(df: pd.DataFrame, bootstrap: int = 200
                                                ) -> pd.DataFrame:
    """
    Performs the rank calculation per bootstrap block where a block is a Aseed-fold-dataset pair

    Args:
        dfs (List[pdDataFrame]): A list of dataframes each representing a seed

    Returns:
        pd.DataFrame: with the ranking
    """

    # Form the desired array with raw scores, an average of everything
    result = pd.pivot_table(df, values='test', index=['tool'], columns=['task'], aggfunc=np.mean)
    df['wins'] = 0  # we rank based on wins
    ranking = pd.pivot_table(df, values='wins', index=['tool'], columns=['task'], aggfunc=np.mean)

    # fix the metric -- greater is better in certain cases. This can be done per task

    # Sort values for proper compare -- yet I don;t think it matters
    df.sort_values(by=['tool', 'task', 'fold', 'seed'], inplace=True)

    test_score_per_task_fold_tool = pd.pivot_table(df, values='test', index=['task', 'fold', 'tool'], columns='seed')
    test_score_per_task_fold_tool.to_csv('test_score_per_task_fold_tool.csv')

    folds = df['fold'].unique()

    for task in tqdm.tqdm(ranking.columns):

        # We divide by 2 because bootstrap is used in 2 placed
        # we have blocks and within each block we have repetitions.
        # we sample with replacemented bootstrap//2 times from a block
        # which in this context is task/fold/autosklearnseed. Within this block
        # are 10 repetitions and we take  bootstrap//2 comparissons
        for boot in range(bootstrap // 2):
            # Pick a random fold, seed
            fold = np.random.choice(folds)
            index = (task, fold)  # would yield a table with tool as row and repetitions as col

            # Get the values per tool(rows) and columns(10 seed repetitions)
            data = test_score_per_task_fold_tool.loc[index].to_numpy()
            data[np.isnan(data)] = 0
            # Do sampling with replacement to get tool as rows again, but different
            # permutations of the 10 seeds
            resample_index = np.random.randint(data.shape[1], size=(data.shape[0], bootstrap//2))
            data = np.take_along_axis(data, resample_index, 1)
            wins = pd.Series(
                (rankdata(data, axis=0, method='min')-1).sum(axis=1),
                index=test_score_per_task_fold_tool.loc[index].index
            )
            # Pretty cool using index: So the index are the tool we are evaluation.
            # if a whole tool failed, it won't even be there and this tool for this
            # task won't be incremented. We add the number of times that for this
            # bootstrap block of fold/seed/task, an experiment was better than other
            ranking[task] = ranking[task] + wins

    ranking = ranking.rank(axis=0, method='average', ascending=False)
    return ranking, result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Utility to plot CSV results')
    parser.add_argument(
        '--csv_location',
        help='Where to look for csv files',
        type=str,
        action='extend',
        nargs='+',
    )
    parser.add_argument(
        '--metric',
        help='Test or overfit metric',
        default='test',
        type=str,
        required=False,
    )
    parser.add_argument(
        '--rank',
        help='Generates results/plots just for this experiment',
        required=True,
        type=str,
        # Aseed means here parent autosklearn seed
        choices=['fold_dataset', 'block_Aseed_fold_dataset', 'bag_fold_dataset']
    )
    parser.add_argument(
        '--tools',
        help='Limit the comparison to a set of tools',
        type=str,
        action='extend',
        nargs='+',
    )
    args = parser.parse_args()

    # First get the data
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
    if args.tools is not None and len(args.tools) > 0:
        # Only keep the desired tools
        df = df[df['tool'].isin(args.tools)]
    df.to_csv('debug.csv')

    # Average the folds across seed to remove noise
    if 'block_Aseed_fold_dataset' in args.rank:
        ranking, rawscores = generate_ranking_per_Aseed_fold_per_dataset(df)
    elif 'bag_fold_dataset' in args.rank:
        ranking, rawscores = generate_ranking_per_fold_per_dataset_bags(df)
    elif 'fold_dataset' in args.rank:
        df = collapse_seed(df)
        ranking, rawscores = generate_ranking_per_fold_per_dataset(df)
    else:
        raise NotImplementedError(args.rank)

    # Save to disk
    filename = f"ranking_{args.rank}"
    ranking.to_csv(f"{filename}.csv")
    rawscores.to_csv(f"{filename}_rawscores.csv")
    print(f"Check {filename}.csv")
