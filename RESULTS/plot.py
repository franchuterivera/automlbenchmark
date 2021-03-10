import glob
import pandas as pd

dfA1 = pd.read_csv('AutoPyTorch_refactor_seed1.csv')
dfA1['seed'] = 1
dfA2 = pd.read_csv('AutoPyTorch_refactor_seed2.csv')
dfA2['seed'] = 2
dfA3 = pd.read_csv('AutoPyTorch_refactor_seed3.csv')
dfA3['seed'] = 3
dfa1 = pd.read_csv('autoPyTorch_seed1.csv')
dfa1['seed'] = 1
dfa2 = pd.read_csv('autoPyTorch_seed2.csv')
dfa2['seed'] = 2
dfa3 = pd.read_csv('autoPyTorch_seed3.csv')
dfa3['seed'] = 3

df = pd.concat([dfA1, dfA2, dfA3, dfa1, dfa2, dfa3])
mean = df.groupby(['task', 'framework']).mean()
std = df.groupby(['task', 'framework']).std()
df = mean
df['result_STD'] = std['result']
pd.pivot_table(
    df.reset_index(),
    columns=['framework'],
    index=['task'],
    values=['result', 'result_STD'],
).reset_index().to_csv('old_vs_new.csv')
