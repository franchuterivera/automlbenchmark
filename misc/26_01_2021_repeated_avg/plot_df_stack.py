import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

for filename in [
    'df_stacking_146818.csv',
    'df_stacking_31.csv',
    'df_stacking_53.csv',
]:
    df = pd.read_csv(filename, index_col=0)
    # plot
    g = sns.FacetGrid(df, col="model", col_wrap=2)
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle(f"Task=Stacking {df['dataset_name'].unique()[0]}")
    g.map_dataframe(sns.lineplot, x="stack", y="performance", hue='hue', palette="colorblind")
    g.set_axis_labels("Stack Depth", "Balanced Accuracy")
    g.add_legend()
    plt.savefig(f"{df['dataset_name'].unique()[0]}_stacking.pdf")
