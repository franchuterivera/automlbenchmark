import pandas as pd
import math
from matplotlib import pyplot as plt

#,hue,performance,model,dataset_name,level,repeat,seed,val_score
#0,test_performance_singlemodelTrue,0.7315020347761746,RandomForestClassifier,53,0,1,94,0.766404681091421
df = pd.read_csv('/home/chico/master_thesis/automlbenchmark/misc/13_02_2021_isolatedrepetitions/sklearn/5_folds/all_data.csv')
# Just care about single model false -- that is, that we can use any model available at that time
df = df[df['hue']=='test_performance_singlemodelFalse']

# Normalize the performance per dataset per level
for dataset_name in df['dataset_name'].unique():
    minimum = df.loc[(df['dataset_name']==dataset_name), 'performance'].min()
    maximum = df.loc[(df['dataset_name']==dataset_name), 'performance'].max()
    df.loc[(df['dataset_name']==dataset_name), 'performance'] = (df.loc[(df['dataset_name']==dataset_name), 'performance'] - minimum)/(maximum-minimum)

df = df.groupby(['hue', 'model', 'level', 'repeat']).mean().reset_index()

total_tools = len(df['repeat'].unique())

# Plot configuration
fig = plt.figure(figsize=(18,12))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
fig.suptitle(f"Effect of Stacking across repetitions")
for i, repeat in enumerate(df['repeat'].unique()):
    ax = fig.add_subplot(math.ceil(total_tools/3), 3, i+1)
    for model in df['model'].unique():
        df[
            (df['model']==model) & (df['repeat']==repeat)
        ].plot(x='level', y=['performance'], ax=ax,
               label=[f"{model}"], legend=False)
    ax.set_title(f"Repeat={repeat}")
    ax.set(ylabel='Balanced Accuracy')
    ax.grid(True)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=10)
plt.savefig(f"stackingroi.pdf")
plt.close()
