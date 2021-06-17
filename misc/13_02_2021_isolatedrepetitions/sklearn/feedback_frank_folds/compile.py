import pandas as pd
import math
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14


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

#,hue,repeated_ensemble,performance,model,dataset_name,level,repeat,splits,seed,val_score
#0,test_performance_singlemodelFalse,False,0.610897435897436,HistGradientBoostingClassifier,10101,0,5,2,97,0.6140120415982485
df = pd.read_csv('all_data.csv')
for col in ['level', 'repeat']:
    df[col] = df[col].astype(int)

# Create a test meand and std
df_mean = df.groupby(['hue', 'model', 'dataset_name', 'level', 'repeat', 'splits']).mean().add_suffix('_mean')
df_std = df.groupby(['hue', 'model', 'dataset_name', 'level', 'repeat', 'splits']).std()
df_mean['performance_std'] = df_std['performance']
df = df_mean.reset_index()

#######################################
#                1
# First point we want to make, is that
# stacking many levels has diminishing returns
#######################################
# We assume like 6 datasets
df_part1 = df[(df['repeat'] == 5) & (df['hue'] == 'test_performance_singlemodelFalse') & (df['level']==1)]

fig = plt.figure(figsize=(10, 8))
for i, dataset_name in enumerate(df_part1['dataset_name'].unique()):
    ax = fig.add_subplot(3, 2, i+1)
    ax.set_title(f"Openml_id={dataset_name}")
    ax.set(ylabel='Balanced Accuracy')
    ax.set(xlabel='splits')
    ax.grid(True)
    for i, model in enumerate(df_part1['model'].unique()):
        this_data = df_part1[(df_part1['model']==model) & (df_part1['dataset_name']==dataset_name)]
        ax.errorbar('splits', 'performance_mean', yerr='performance_std',  capsize=3, capthick=3, elinewidth=2, linewidth=1, linestyle='dashed', data=this_data, color=color[i], label=model, alpha=0.5)
plt.tight_layout()
lgd = plt.legend(loc='lower center',  bbox_to_anchor=(-0.14, -0.95), ncol=3, fancybox=True)
plt.savefig(f"subsechowtodefinefolds.pdf", dpi=1000, bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.close()
