import pandas as pd
import math
from matplotlib import pyplot as plt

df = pd.read_csv('/home/chico/master_thesis/automlbenchmark/misc/13_02_2021_isolatedrepetitions/sklearn/rfonly/history/all_data.csv')
#,single_model,dataset_name,model,level,repeat,iteration,seed,performance
data = df.groupby(['single_model', 'model', 'dataset_name', 'level', 'repeat', 'iteration']).mean().reset_index()
# Just care about single model
data = data[data['single_model']==False]

total_tools = len(df['model'].unique())


for dataset_name in data['dataset_name'].unique():
    # Plot configuration
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    fig.suptitle(f"Dataset={dataset_name} @ 2-repeats-5-folds-cv")
    for i, model in enumerate(data['model'].unique()):
        ax = fig.add_subplot(math.ceil(total_tools/3), 3, i+1)
        data[
            (data['model']==model) & (data['repeat']==2) & (data['dataset_name']==dataset_name)
        ].plot(x='iteration', y='performance', ax=ax)
        ax2 = ax.twinx()
        data[
            (data['model']==model) & (data['repeat']==2) & (data['dataset_name']==dataset_name)
        ].plot(x='iteration', y='level', ax=ax2, color='r')
        ax.set_title(f"{model}")
        ax.set(ylabel='Balanced Accuracy')
        ax.grid(True)
    plt.show()
    plt.close()
