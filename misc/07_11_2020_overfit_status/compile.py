import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('testmtrain.csv')


pd.pivot_table(
    df,
    values='testmtrain',
    index=['task'],
    columns=['tool']
).reset_index().plot(
    x='task',
    y=[t for t in df['tool'].unique()],
    kind='barh',
)
plt.show()
