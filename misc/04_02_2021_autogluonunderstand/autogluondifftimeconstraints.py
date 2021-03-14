import pandas as pd
import matplotlib.pyplot as plt

ax = pd.read_csv('autogluondifftimeconstraints.csv').plot(
    x='task',
    y=['AutoGluon_14400s1c8G', 'AutoGluon_28800s1c8G', 'AutoGluon_3600s1c8G', 'autosklearn_28800s1c8G', 'autosklearn_3600s1c8G', 'autosklearn_14400s1c8G'],
    kind='barh',
    title='Effect of resources in AutoGluon and Auto-Sklearn strategies',
    grid=True,
)
ax.set_xlabel('Balanced Accuracy')
ax.grid(True)
plt.savefig('autogluondifftimeconstraints.pdf', figsize=(18,12))
plt.tight_layout()
plt.show()
