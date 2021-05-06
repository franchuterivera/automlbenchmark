import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import json

parser = ArgumentParser()
parser.add_argument("-f", "--file", dest="filename",
                    help="write report to FILE", metavar="FILE")
args = parser.parse_args()
df = pd.read_csv(args.filename)

# Rename ensemble intensification
name = [name for name in df['tool'].unique() if 'Ensemble' in name][0]
print(name)
df.loc[df['tool'] == name, 'tool'] = 'EnsembleIntensification'

df['fold'] = df['fold'].astype(str)
df['task_fold'] = df['task'] + df['fold']
data = pd.pivot(
    #df.groupby(['task_fold', 'tool']).mean().reset_index(),
    df,
    #index=['task_fold'],
    index=['task_fold', 'seed'],
    columns=['tool'],
    values=['test']
)
print(data)
data.columns = data.columns.map('|'.join).str.strip('|')
#ax = data.reset_index().plot(
data = data.reset_index()

for task in df['task'].unique():
    fig, ax = plt.subplots(nrows=2, ncols=2)

    for i, row in enumerate(ax):
        for j, col in enumerate(row):
            data[data['task_fold'] == f"{task}{i+j}"].boxplot(
                column=[col for col in data.columns if 'test' in col], ax=col
            )

    #handles, labels = ax.get_legend_handles_labels()
    #ax.legend(handles, labels, loc='center', ncol=5)
    fig.suptitle(f"{task}")
    #plt.xlim((0.675,1))
    #plt.legend(loc='lower right')
    plt.show()
    plt.close()
