import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
import pandas as pd


dfs = []
for filename in [
    '/home/chico/master_thesis/automlbenchmark/misc/13_02_2021_isolatedrepetitions/sklearn/2_folds/all_data.csv',
    '/home/chico/master_thesis/automlbenchmark/misc/13_02_2021_isolatedrepetitions/sklearn/5_folds/all_data.csv',
    '/home/chico/master_thesis/automlbenchmark/misc/13_02_2021_isolatedrepetitions/sklearn/10_folds/all_data.csv',
]:
    df = pd.read_csv(filename)
    df['folds'] = os.path.basename(os.path.dirname(filename)).split('_')[0]
    dfs.append(df)
df = pd.concat(dfs)
df = df.groupby(['hue', 'dataset_name', 'level', 'repeat', 'model', 'folds']).mean().reset_index()
df = df[df['hue']=='test_performance_singlemodelFalse']

# Too many models, remove some
df = df[~df['model'].str.contains("DecisionTreeClassifier")]
df = df[~df['model'].str.contains("LinearDiscriminantAnalysis")]
df = df[~df['model'].str.contains("GradientBoostingClassifier")]

total_outer = len(df['dataset_name'].unique())
total_inner = len(df['model'].unique())
fig = plt.figure(figsize=(18, 12))
outer = gridspec.GridSpec(math.ceil(math.sqrt(total_outer)), math.floor(math.sqrt(total_outer)), wspace=0.2, hspace=0.2)
for i, dataset_name in enumerate(df['dataset_name'].unique()):
    inner = gridspec.GridSpecFromSubplotSpec(total_inner, 1,
                    subplot_spec=outer[i], wspace=0.1, hspace=0.1)

    for j, model in enumerate(df['model'].unique()):
        ax = plt.Subplot(fig, inner[j])
        #t = ax.text(0.5,0.5, 'outer=%d, inner=%d' % (i, j))
        #for level in df['level'].unique():
        for level in [0, 1]:
            for folds in df['folds'].unique():
                df[
                        (df['model']==model) & (df['dataset_name']==dataset_name) & (df['level']==level) & (df['folds']==folds)
                    ].plot(x='repeat', y=['performance'], ax=ax,
                           label=[f"L:{level} K-fold={folds}"], legend=False)
        #ax.set_xticks([])
        #ax.set_yticks([])
        ax.annotate(model,
            xy=(0, 0), xycoords='axes fraction',
            xytext=(0, 0), textcoords='axes fraction',
            horizontalalignment='left', verticalalignment='bottom')
        fig.add_subplot(ax)
        if model != df['model'].unique()[-1] or i >= len(df['dataset_name'].unique())-2:
            pass
        else:
            x_axis = ax.axes.get_xaxis()
            x_axis.set_visible(False)
        if j==0:
            ax.set_title(f"OpenMLID={dataset_name}")
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=10)
plt.tight_layout()
plt.savefig(f"effectdifferentfolds.pdf")
plt.show()
