import pandas as pd
import math
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

#,hue,performance,model,dataset_name,level,repeat,seed,val_score
#0,test_performance_singlemodelTrue,0.7315020347761746,RandomForestClassifier,53,0,1,94,0.766404681091421
df = pd.read_csv('/home/chico/master_thesis/automlbenchmark/misc/13_02_2021_isolatedrepetitions/sklearn/5_folds/all_data.csv')
data = df.groupby(['hue', 'model', 'dataset_name', 'level', 'repeat']).mean().reset_index()
# Just care about single model false -- that is, that we can use any model available at that time
data = data[data['hue']=='test_performance_singlemodelFalse']

total_tools = len(df['model'].unique())

for dataset_name in data['dataset_name'].unique():
    # Plot configuration
    fig = plt.figure(figsize=(18, 12))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    fig.suptitle(f"Dataset={dataset_name} 5 Repetitions of 5 Folds")
    for i, model in enumerate(data['model'].unique()):
        ax = fig.add_subplot(math.ceil(total_tools/3), 3, i+1)
        level_dict = {
            0: ['r', 'r--'],
            1: ['b', 'b--'],
            2: ['g', 'g--'],
            3: ['y', 'y--'],
        }
        for level in data['level'].unique():
            data[
                (data['model']==model) & (data['dataset_name']==dataset_name) & (data['level']==level)
            ].plot(x='repeat', y=['performance', 'val_score'], ax=ax, style=level_dict[level],
                   label=[f"L:{level} Test", f"L:{level} Val"])
        ax.set_title(f"{model}")
        ax.set(ylabel='Balanced Accuracy')
        ax.grid(True)
    #plt.show()
    plt.savefig(f"individual_effect_{dataset_name}.pdf")
    plt.close()
