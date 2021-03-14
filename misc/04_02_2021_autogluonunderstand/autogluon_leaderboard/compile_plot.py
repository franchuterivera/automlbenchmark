import pandas as pd
import math
from matplotlib import pyplot as plt
import glob
import re


filere = re.compile(r'seed(\d)/AutoGluon_master_thesis_28800s1c8G_(.+)_(\d)_')

# First read the data
dfs = []
for filename in glob.glob('seed*/*'):
    # filename=seed1/AutoGluon_master_thesis_28800s1c8G_segment_0_2021.02.04-00.25.17.out
    match = filere.search(filename)
    if match is None:
        raise ValueError(filename)

    df = pd.read_csv(filename)
    df['seed'] = match.group(1)
    df['task'] = match.group(2)
    df['fold'] = match.group(3)
    dfs.append(df)

df = pd.concat(dfs)

#,model,score_test,score_val,pred_time_test,pred_time_val,fit_time,pred_time_test_marginal,pred_time_val_marginal,fit_time_marginal,stack_level,can_infer,fit_order
df = df.groupby(['model', 'stack_level', 'task']).mean().reset_index()

total = len(df['task'].unique())

# Remove the level tag from model
df['model'] = df['model'].str[:-3]

# Plot configuration
fig = plt.figure(figsize=(18,12))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
fig.suptitle(f"AutoGluon Leaderboard for 8h 1c 8Gb")
for i, task in enumerate(df['task'].unique()):
    ax = fig.add_subplot(math.ceil(total/2), 2, i+1)
    level_dict = {
        0: ['r', 'r--'],
        1: ['b', 'b--'],
        2: ['g', 'g--'],
        3: ['y', 'y--'],
        4: ['c', 'c--'],
        5: ['m', 'm--'],
        6: ['m', 'm--'],
        7: ['m', 'm--'],
        8: ['m', 'm--'],
        9: ['m', 'm--'],
        10: ['m', 'm--'],
        11: ['m', 'm--'],
        12: ['m', 'm--'],
        12: ['m', 'm--'],
        6: ['darkorange', 'darkgorange--'],
        7: ['tab:purple', 'tab:purple--'],
        8: ['tab:brown', 'tab:brown--'],
        9: ['tab:gray', 'tab:gray--'],
    }
    for i, model in enumerate(df['model'].unique()):
        df[
            (df['task']==task) & (df['model']==model)
        #].plot(x='stack_level', y=['score_test', 'score_val'], ax=ax, # style=level_dict[i],
        ].plot(x='stack_level', y=['score_test'], ax=ax, # style=level_dict[i],
               #label=[f"L:{model} Test", f"L:{model} Val"])
               label=[f"L:{model}"], legend=False)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center', ncol=5)
    ax.set_title(f"Task={task}")
    ax.set(ylabel='Test Balanced Accuracy', xlabel='Stacking Level')
    ax.grid(True)
#plt.show()
plt.savefig('autogluonleaderboard.pdf')
plt.close()
