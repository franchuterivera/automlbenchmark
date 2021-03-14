import pandas as pd
import matplotlib.pyplot as plt

df_autogluon = pd.read_csv('leaderboard_autogluon.csv')
df_autosklearn = pd.read_csv('leaderboard_autosklearn.csv')

# remove the fold
df_autogluon = df_autogluon.groupby(['tool', 'benchmark', 'task', 'model']).mean().reset_index()
df_autosklearn = df_autosklearn.groupby(['tool', 'benchmark', 'task']).mean().reset_index()

df = []
for task in df_autogluon['task'].unique():
    # Autosklearn does not have it
    if 'KDDCup09' in task: continue

    if not df_autogluon[(df_autogluon['task'] == task) & (df_autogluon['model'] == 'WeightedEnsemble_L2')].empty:
        autogluon_task = df_autogluon[(df_autogluon['task'] == task) & (df_autogluon['model'] == 'WeightedEnsemble_L2')]
    else:
        autogluon_task = df_autogluon[(df_autogluon['task'] == task) & (df_autogluon['model'] == 'WeightedEnsemble_L1')]
    autosklearn_task = df_autosklearn[(df_autosklearn['task'] == task)]
    df.append({
        'task': task,
        'AutoGluon': autogluon_task['score_val'].values[0] - autogluon_task['score_test'].values[0],
        'AutoSklearn': autosklearn_task['ensemble_optimization_score'].values[0] - autosklearn_task['ensemble_test_score'].values[0],
    })

ax = pd.DataFrame(df).plot(
    x='task',
    y=['AutoGluon', 'AutoSklearn'],
    kind='barh',
    title='Overfit of the state of the art AutoML Frameworks',
    grid=True,
)
ax.set_xlabel('Test minus validation Accuracy')
ax.grid(True)
plt.savefig('frameworks_overfit.pdf')
plt.tight_layout()
plt.show()
