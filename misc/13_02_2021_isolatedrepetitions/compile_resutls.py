import pandas as pd
df = pd.read_csv('all_data.csv')
data = pd.pivot_table(df, values='performance', index=['hue', 'dataset_name', 'repeat'], columns=['level'])
data.subtract(data[0], axis=0)
