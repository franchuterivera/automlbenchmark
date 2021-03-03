import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import json

parser = ArgumentParser()
parser.add_argument("-f", "--file", dest="filename",
                    help="write report to FILE", metavar="FILE")
args = parser.parse_args()
fig = plt.figure()
df = pd.read_csv(args.filename)
data = pd.pivot(
    df.groupby(['task', 'framework']).mean().reset_index(),
    index=['task'],
    columns=['framework'],
    values=['best_individual_val', 'best_individual_test', 'best_ensemble_val', 'best_ensemble_test']
)
data.columns = data.columns.map('|'.join).str.strip('|')
ax = data.reset_index().plot(
    x='task',
    y=[col for col in data.columns if 'best_individual' in col],
    color=['green', 'mediumblue', 'limegreen', 'cornflowerblue'],
    kind='barh',
    #legend=False,
)
#handles, labels = ax.get_legend_handles_labels()
#ax.legend(handles, labels, loc='center', ncol=5)
ax.set_title(f"Individual model debug")
plt.xlim((0.6,1))
plt.legend(loc='lower right')
plt.show()
ax = data.reset_index().plot(
    x='task',
    y=[col for col in data.columns if 'best_ensemble' in col],
    color=['green', 'mediumblue', 'limegreen', 'cornflowerblue'],
    kind='barh',
    #legend=False,
)
#handles, labels = ax.get_legend_handles_labels()
#ax.legend(handles, labels, loc='center', ncol=5)
ax.set_title(f"Ensemblle model debug")
plt.xlim((0.6,1))
plt.legend(loc='lower right')
plt.show()
