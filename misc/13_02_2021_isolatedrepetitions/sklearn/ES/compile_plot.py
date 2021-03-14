import pandas as pd
import math
from matplotlib import pyplot as plt

#,hue,repeated_ensemble,performance,model,dataset_name,level,repeat,seed,val_score
df = pd.read_csv('/home/chico/master_thesis/automlbenchmark/misc/13_02_2021_isolatedrepetitions/sklearn/ES/all_data.csv')
df = df[df['model']=='EnsembleSelection']
data = df.groupby(['hue', 'repeated_ensemble', 'dataset_name', 'level', 'repeat']).mean().reset_index()

# Just care about single model false -- that is, that we can use any model available at that time
data = data[data['hue']=='test_performance_singlemodelFalse']


total = len(df['dataset_name'].unique())

# Plot configuration
for level in data['level'].unique():
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    fig.suptitle(f"Ensemble Selection of 6 Sklearn Models on 5-k-fold splits")
    for i, dataset_name in enumerate(data['dataset_name'].unique()):
        ax = fig.add_subplot(math.ceil(total/2), 2, i+1)
        for repeated_ensemble in data['repeated_ensemble'].unique():
            if repeated_ensemble:
                level_dict = {
                    0: ['ro-', 'ro--'],
                    1: ['bo-', 'bo--'],
                    2: ['go-', 'go--'],
                    3: ['yo-', 'yo--'],
                }
            else:
                level_dict = {
                    0: ['rv-', 'rv--'],
                    1: ['bv-', 'bv--'],
                    2: ['gv-', 'gv--'],
                    3: ['yv-', 'yv--'],
                }
            data[
                (data['dataset_name']==dataset_name) & (data['level']==level) & (data['repeated_ensemble']==repeated_ensemble)
            ].plot(x='repeat', y=['performance', 'val_score'], ax=ax, style=level_dict[level],
                   label=[f"L:{level} ES-Repeated:{repeated_ensemble} Test", f"L:{level} ES-Repeated:{repeated_ensemble} Val"], legend=False)
        ax.set_title(f"OpenMLID={dataset_name}")
        ax.set(ylabel='Balanced Accuracy', xlabel='Number of CV Repetitions')
        ax.grid(True)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=6)
    plt.show()
    plt.close()
