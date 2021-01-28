import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

for filename in ['df_repeats_146818.csv',  'df_repeats_31.csv',  'df_repeats_53.csv',  'df_repeats_7592.csv',  'df_repeats_9981.csv']:
    df = pd.read_csv(filename, index_col=0)
    # Column to row
    df = df.melt(id_vars=['model', 'repeat', 'dataset_name'], var_name='type', value_name='performance')
    # plot
    g = sns.FacetGrid(df, col="model", col_wrap=2)
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle(f"Task={df['dataset_name'].unique()[0]}")
    g.map_dataframe(sns.lineplot, x="repeat", y="performance", hue='type', palette="colorblind")
    g.set_axis_labels("Repeats", "Balanced Accuracy")
    g.add_legend()
    plt.savefig(f"{df['dataset_name'].unique()[0]}_repeats.pdf")

