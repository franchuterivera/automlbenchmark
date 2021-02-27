import pandas as pd
import math
from matplotlib import pyplot as plt

#,hue,performance,model,dataset_name,level,repeat,seed,val_score
#0,test_performance_singlemodelTrue,0.7315020347761746,RandomForestClassifier,53,0,1,94,0.766404681091421
df = pd.read_csv('/home/chico/master_thesis/automlbenchmark/misc/13_02_2021_isolatedrepetitions/sklearn/few_reps/all_data.csv')
data = df.groupby(['hue', 'dataset_name', 'level', 'repeat']).mean().reset_index()
# Just care about single model false -- that is, that we can use any model available at that time
data = data[data['hue']=='test_performance_singlemodelFalse']

total = len(df['dataset_name'].unique())

# Plot configuration
fig = plt.figure()
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
plt.show()
plt.close()
