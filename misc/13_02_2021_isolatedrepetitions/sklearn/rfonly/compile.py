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
for col in ['level', 'repeat']:
    df[col] = df[col].astype(int)

#######################################
#                1
# First point we want to make, is that
# stacking many levels has diminishing returns
#######################################
# We assume like 6 datasets
fig = plt.figure(figsize=(18, 12))
for i, dataset_name in enumerate(df['dataset_name'].unique()):
    ax = fig.add_subplot(3, 3, i+1)
    ax.set_title(f"Openml_id={dataset_name}")
    ax.set(ylabel='Balanced Accuracy')
    ax.set(xlabel='Level')
    ax.grid(True)
    for i, model in enumerate(df['model'].unique()):
        this_data = df[(df['model']==model) & (df['dataset_name']==dataset_name)]
        ax.errorbar('level', 'performance_mean', yerr='performance_std',  capsize=3, capthick=3, elinewidth=2, linewidth=1, linestyle='dashed', data=this_data, color=color[i], label=model, alpha=0.5)
        xint = range(df['level'].min(), df['level'].max() + 1)
        ax.set_xticks(xint)
plt.tight_layout()
plt.legend(loc='lower center',  bbox_to_anchor=(0.43, -0.35), ncol=7)
plt.savefig(f"plot_not_all_models_see_benefit.pdf")
plt.show()
plt.close()
