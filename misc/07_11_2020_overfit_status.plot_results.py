import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


data = []
for data_file in glob.glob('misc/*/*overfit*csv'):
    data.append(
        pd.read_csv(
            data_file,
            index_col=0,
        )
    )

data = pd.concat(data).reindex()

# Only plot ensemble for now
model = 'best_ensemble_model'
data = data[data['model'] == model]


#for task in np.unique(data['task']):
    #task_frame = data[data['task']==task]
data['test'] = pd.to_numeric(data['test'])
sns.set_style("whitegrid")
sns.pointplot(
    'task',
    'test',
    hue='tool',
    data=data,
    dodge=True,
    join=False,
    palette=sns.color_palette("Set2"),
    ci='sd',
)
ax = sns.lineplot(
    'task',
    'test',
    hue='tool',
    data=data[data['tool']=='autosklearn'],
    drawstyle='steps-post',
    dashes=True,
    ci=None,
)

#plt.show()

experiment_results = {}
for tool_task, test_value in data.groupby(['tool', 'task']).mean()['test'].to_dict().items():
    tool, task = tool_task
    if tool not in experiment_results:
        experiment_results[tool] = {}
    if task not in experiment_results[tool]:
        experiment_results[tool][task] = test_value


summary = []
for tool in experiment_results:
    row = experiment_results[tool]
    row['tool'] = tool
    summary.append(row)

summary = pd.DataFrame(summary)
print(summary)

# The best per task:
for task in [c for c in summary.columns if c != 'tool']:
    best = summary[task].argmax()
    print(f"{task}(best) = {summary['tool'].iloc[best]}")

# How many times better than autosklearn
summary_no_tool_column = summary.loc[:, summary.columns != 'tool']
baseline_results = summary[summary['tool']=='autosklearn'].loc[:, summary[summary['tool']=='autosklearn'].columns != 'tool']
for index, row in summary.iterrows():
    tool = row['tool']
    if tool == 'autosklearn': continue
    print(f"{tool} (better_than_baseline): {np.count_nonzero(summary_no_tool_column.iloc[index] > baseline_results)}")

