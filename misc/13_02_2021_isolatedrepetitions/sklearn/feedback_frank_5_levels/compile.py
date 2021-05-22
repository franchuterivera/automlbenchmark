import pandas as pd
import math
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure


# color dictionary
color = {
    0: 'r',
    1: 'b',
    2: 'g',
    3: 'y',
    4: 'm',
    5: 'c',
    6: 'k',
}

# ,hue,performance,model,dataset_name,level,repeat,seed,val_score
df = pd.read_csv('all_data.csv')
for col in ['level', 'repeat']:
    df[col] = df[col].astype(int)

# Create a test meand and std
df_mean = df.groupby(['hue', 'model', 'dataset_name', 'level', 'repeat']).mean().add_suffix('_mean')
df_std = df.groupby(['hue', 'model', 'dataset_name', 'level', 'repeat']).std()
df_mean['performance_std'] = df_std['performance']
df = df_mean.reset_index()

#######################################
#                1
# First point we want to make, is that
# stacking many levels has diminishing returns
#######################################
# We assume like 6 datasets
df_part1 = df[(df['repeat'] == 5) & (df['hue'] == 'test_performance_singlemodelFalse')]

fig = plt.figure(figsize=(18, 12))
for i, dataset_name in enumerate(df_part1['dataset_name'].unique()):
    ax = fig.add_subplot(3, 3, i+1)
    ax.set_title(f"Openml_id={dataset_name}")
    ax.set(ylabel='Balanced Accuracy')
    ax.set(xlabel='Level')
    ax.grid(True)
    for i, model in enumerate(df_part1['model'].unique()):
        this_data = df_part1[(df_part1['model']==model) & (df_part1['dataset_name']==dataset_name)]
        ax.errorbar('level', 'performance_mean', yerr='performance_std',  capsize=3, capthick=3, elinewidth=2, linewidth=1, linestyle='dashed', data=this_data, color=color[i], label=model, alpha=0.5)
plt.tight_layout()
plt.legend(loc='lower center',  bbox_to_anchor=(-0.75, -0.35), ncol=7)
plt.savefig(f"plot_diminishing_returns_stacking.pdf")
plt.close()

#######################################
#                2
# Then we want to select a safe number
# of repetitions
#######################################
# We assume like 6 datasets
for level in [0, 1, 2]:
    df_part2 = df[(df['level'] == level) & (df['hue'] == 'test_performance_singlemodelFalse')]
    fig = plt.figure(figsize=(18, 12))
    for i, dataset_name in enumerate(df_part2['dataset_name'].unique()):
        ax = fig.add_subplot(3, 3, i+1)
        ax.set_title(f"Openml_id={dataset_name} level={level}")
        ax.set(ylabel='Balanced Accuracy')
        ax.set(xlabel='5-fold-repetitions')
        ax.grid(True)
        for i, model in enumerate(df_part2['model'].unique()):
            this_data = df_part2[(df_part2['model']==model) & (df_part2['dataset_name']==dataset_name)]
            ax.errorbar('repeat', 'performance_mean', yerr='performance_std',  capsize=3, capthick=3, elinewidth=2, linewidth=1, linestyle='dashed', data=this_data, color=color[i], label=model, alpha=0.5)
    plt.tight_layout()
    plt.legend(loc='lower center',  bbox_to_anchor=(-0.75, -0.35), ncol=7)
    plt.savefig(f"plot_repeat_permodel_effect_data{dataset_name}_level{level}.pdf")
    plt.close()
