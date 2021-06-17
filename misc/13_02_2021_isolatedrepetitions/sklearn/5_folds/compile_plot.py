import pandas as pd
import math
from matplotlib import pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14


#,hue,performance,model,dataset_name,level,repeat,seed,val_score
#0,test_performance_singlemodelTrue,0.7315020347761746,RandomForestClassifier,53,0,1,94,0.766404681091421
df = pd.read_csv('/home/chico/master_thesis/automlbenchmark/misc/13_02_2021_isolatedrepetitions/sklearn/5_folds/all_data.csv')
data = df.groupby(['hue', 'dataset_name', 'level', 'repeat']).mean().reset_index()


# Print some statistics
df_statistics = df.groupby(['hue', 'dataset_name', 'level', 'repeat', 'model']).mean().reset_index()
print(f"hue, dataset_name, level, repeat")
for hue in data['hue'].unique():
    for dataset_name in data['dataset_name'].unique():
        for level in data['level'].unique():
            repeats = sorted(data['repeat'].unique())
            for i in range(1, len(repeats)):
                if repeats[i] not in df_statistics.set_index(['hue', 'dataset_name', 'level', 'model']).loc[(hue, dataset_name, level), 'repeat'].unique():
                    continue
                if repeats[i-1] not in df_statistics.set_index(['hue', 'dataset_name', 'level', 'model']).loc[(hue, dataset_name, level), 'repeat'].unique():
                    continue
                performance_r_a = df_statistics.set_index(['hue', 'dataset_name', 'level', 'repeat', 'model']).loc[(hue, dataset_name, level, repeats[i-1]), 'performance']
                performance_r_b = df_statistics.set_index(['hue', 'dataset_name', 'level', 'repeat', 'model']).loc[(hue, dataset_name, level, repeats[i]), 'performance']
                print(f"{hue}, {dataset_name}, {level}, {repeats[i-1]}->{repeats[i]}, {(performance_r_b - performance_r_a).max()}")

# Just care about single model false -- that is, that we can use any model available at that time
data = data[data['hue']=='test_performance_singlemodelFalse']

total = len(df['dataset_name'].unique())

# Plot configuration
fig = plt.figure(figsize=(18,12))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
fig.suptitle(f"Avg Model Performance of 5-k-fold splits")
for i, dataset_name in enumerate(data['dataset_name'].unique()):
    ax = fig.add_subplot(math.ceil(total/3), 3, i+1)
    level_dict = {
        0: ['r', 'r--'],
        1: ['b', 'b--'],
        2: ['g', 'g--'],
        3: ['y', 'y--'],
    }
    for level in data['level'].unique():
        data[
            (data['dataset_name']==dataset_name) & (data['level']==level)
        ].plot(x='repeat', y=['performance', 'val_score'], ax=ax, style=level_dict[level],
               label=[f"L:{level} Test", f"L:{level} Val"])
    ax.set_title(f"OpenMLID={dataset_name}")
    ax.set(ylabel='Balanced Accuracy', xlabel='Number of CV Repetitions')
    ax.grid(True)
plt.savefig(f"effectsofrepetitions.pdf")
plt.close()

fig = plt.figure(figsize=(18,12))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for col in ['level', 'repeat']:
    data[col] = data[col].astype(int)
#fig.suptitle(f"Avg Model Performance of 5-k-fold splits")
for i, dataset_name in enumerate(data['dataset_name'].unique()):
    ax = fig.add_subplot(math.ceil(total/2), 2, i+1)
    level_dict = {
        0: ['r--'],
        1: ['b--'],
        2: ['g--'],
        3: ['y--'],
    }
    for level in data['level'].unique():
        data[
            (data['dataset_name']==dataset_name) & (data['level']==level)
        ].plot(x='repeat', y=['performance'], ax=ax, style=level_dict[level], alpha=0.5, linewidth=1,
               label=[f"L:{level}"], legend=False, marker='o', markersize=2, fontsize=14)
    ax.set_title(f"OpenMLID={dataset_name}")
    ax.set(ylabel='Balanced Accuracy', xlabel='Number of CV Repetitions')
    ax.grid(True)
    xint = range(data.loc[data['dataset_name']==dataset_name, 'repeat'].min(), data.loc[data['dataset_name']==dataset_name, 'repeat'].max() + 1)
    ax.set_xticks(xint)

handles, labels = ax.get_legend_handles_labels()
lgd = fig.legend(handles, labels, loc='lower center', ncol=10)
plt.savefig(f"subsechowtorepeat.pdf", dpi=1000, bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.close()

