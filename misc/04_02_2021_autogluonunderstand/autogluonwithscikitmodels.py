import pandas as pd
import matplotlib.pyplot as plt

ax = pd.read_csv('autogluonwithscikitmodels.csv').plot(
    x='tool',
    y=['AutoGluon', 'AutoGluon with Sklearn models', 'Auto-Sklearn'],
    kind='barh',
    title='Effect of using Sklearn models into AutoGluon',
    grid=True,
)
ax.set_xlabel('Balanced Accuracy')
ax.grid(True)
plt.savefig('sklearnintoautogluon.pdf')
plt.tight_layout()
plt.show()
