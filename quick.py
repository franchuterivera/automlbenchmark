import pandas as pd
df_rl = pd.read_csv('repeat_level_contribution.csv')

data = {'ALL': {}}

for level in df_rl['level'].unique():
    this_df = df_rl[df_rl['level']==level]
    for repeat in sorted(df_rl['repeats'].unique(), reverse=True):
        # We ignore repeat 0 as when doing repeat 1, it was already analyzed
        if repeat == 0:
            continue
        print(f"Working on level={level} repeat={repeat}")
        improvement = this_df.set_index(['task', 'fold', 'seed', 'num_run', 'repeats', 'level']).xs(repeat, level='repeats')['Test'] - this_df.set_index(['task', 'fold', 'seed', 'num_run', 'repeats', 'level']).xs(repeat-1, level='repeats')['Test']
        print(f"improvement={improvement}")
        improvement_df = improvement.to_frame().dropna().reset_index()
        for statistic in ['max', 'mean', 'min']:
            if statistic not in data['ALL']:
                data['ALL'][statistic] = {}
            if statistic == 'max':
                result = improvement.max()
            elif statistic == 'mean':
                result = improvement.mean()
            elif statistic == 'min':
                result = improvement.min()
            else:
                raise NotImplementedError(statistic)
            if f"L:{level} {repeat-1}->{repeat}" not in data['ALL'][statistic]:
                data['ALL'][statistic][f"L:{level} {repeat-1}->{repeat}"] = result
            else:
                # Only fill once
                raise ValueError(f"L:{level} {repeat-1}->{repeat}")
        for task in improvement_df['task'].unique():
            if task not in data:
                data[task] = {}
            for statistic in ['max', 'mean', 'min']:
                if statistic not in data[task]:
                    data[task][statistic] = {}
                if statistic == 'max':
                    result = improvement_df.loc[improvement_df['task']==task, 'Test'].max()
                elif statistic == 'mean':
                    result = improvement_df.loc[improvement_df['task']==task, 'Test'].mean()
                elif statistic == 'min':
                    result = improvement_df.loc[improvement_df['task']==task, 'Test'].min()
                else:
                    raise NotImplementedError(statistic)
                if f"L:{level} {repeat-1}->{repeat}" not in data[task][statistic]:
                    data[task][statistic][f"L:{level} {repeat-1}->{repeat}"] = result
                else:
                    # Only fill once
                    raise ValueError(f"L:{level} {repeat-1}->{repeat}")

# Also add Level transition goodness:
transition = f"L1R4->L2R0"
# Remove level to compare across level
test_level1 = df_rl[df_rl['level']==2].set_index(['task', 'fold', 'seed', 'num_run', 'repeats', 'level']).xs(0, level='repeats').reset_index(level=4, drop=True)['Test']
test_level0 = df_rl[df_rl['level']==1].set_index(['task', 'fold', 'seed', 'num_run', 'repeats', 'level']).xs(4, level='repeats').reset_index(level=4, drop=True)['Test']
improvement = test_level1 - test_level0
data['ALL'][statistic]
for statistic in ['max', 'mean', 'min']:
    if statistic not in data['ALL']:
        data['ALL'][statistic] = {}
    if statistic == 'max':
        result = improvement.max()
    elif statistic == 'mean':
        result = improvement.mean()
    elif statistic == 'min':
        result = improvement.min()
    else:
        raise NotImplementedError(statistic)
    if transition not in data['ALL'][statistic]:
        data['ALL'][statistic][transition] = result
    else:
        # Only fill once
        raise ValueError(transition)

improvement_df = improvement.to_frame().dropna().reset_index()
for task in improvement_df['task'].unique():
    if task not in data:
        data[task] = {}
    for statistic in ['max', 'mean', 'min']:
        if statistic not in data[task]:
            data[task][statistic] = {}
        if statistic == 'max':
            result = improvement_df.loc[improvement_df['task']==task, 'Test'].max()
        elif statistic == 'mean':
            result = improvement_df.loc[improvement_df['task']==task, 'Test'].mean()
        elif statistic == 'min':
            result = improvement_df.loc[improvement_df['task']==task, 'Test'].min()
        else:
            raise NotImplementedError(statistic)
        if transition not in data[task][statistic]:
            data[task][statistic][transition] = result
        else:
            # Only fill once
            raise ValueError(transition)

rows = []
for task, statistics in data.items():
    for statistic, repeat_levels in statistics.items():
        row = {'task': task, 'statistic': statistic}
        for repeat_level, result in repeat_levels.items():
            row[repeat_level] = result
        rows.append(row)
df = pd.DataFrame(rows)
print(df.to_markdown())
