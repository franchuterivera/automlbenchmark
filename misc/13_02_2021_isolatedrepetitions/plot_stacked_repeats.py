import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import glob

path = '/home/chico/master_thesis/automlbenchmark/misc/13_02_2021_isolatedrepetitions/sklearn/hpo'

#for tool in ['autogluon', 'sklearn', 'hpo']:
for tool in ['sklearn', 'hpo']:
    for filename in glob.glob(f"{path}/*{tool}_history*.csv"):
        continue
        df = pd.read_csv(filename, index_col=0)
        # Column to row
        df.drop(columns=['seed'], inplace=True)
        df = df[df['use_train_data'] == True]
        # Tag column with name
        df['single_model'] =  'StackSelf' +  df['single_model'].astype(str)
        df['repeat'] =  'repeat' +  df['repeat'].astype(str)
        df['use_train_data'] =  'X_trainEveryLevel' +  df['use_train_data'].astype(str)

        df = df[df['iteration'] < 151]

        df['type'] = df.apply(lambda x: '-'.join(x[['single_model', 'repeat']].values.tolist()), axis=1)
        # plot
        sns.set_style("whitegrid", {'grid.linestyle': '--'})
        g = sns.FacetGrid(df, row="model", sharey= False, aspect=4)
        #g.set(xticks=df.iteration[0::5])
        plt.subplots_adjust(top=0.9)
        g.fig.suptitle(f"{tool} History Task={df['dataset_name'].unique()[0]}")
        g.map_dataframe(sns.lineplot, x="iteration", y="performance", hue='type')
        g.set_axis_labels("TAE iteration", "Balanced Accuracy")
        g.add_legend()
        #plt.savefig(f"{df['dataset_name'].unique()[0]}_repeats.pdf")
        plt.show()

    for filename in glob.glob(f"{path}/*{tool}_*.csv"):
        if 'history' in filename: continue
        df = pd.read_csv(filename, index_col=0)
        # Column to row
        df = df[df['repeat'] == 10]
        df.drop(columns=['seed'], inplace=True)
        if 'use_train_data' in df.columns:
            df = df[df['use_train_data'] == True]
        # Tag column with name
        df['repeat'] =  'repeat' +  df['repeat'].astype(str)
        df['hue'] = df.apply(lambda x: '-'.join(x[['hue', 'repeat']].values.tolist()), axis=1)
        # plot
        sns.set_style("whitegrid", {'grid.linestyle': '--'})
        g = sns.FacetGrid(df, col="model", col_wrap=2, sharey= False, aspect=4)
        #g.set(xticks=df.iteration[0::5])
        plt.subplots_adjust(top=0.9)
        g.fig.suptitle(f"{tool} Task={df['dataset_name'].unique()[0]}")
        g.map_dataframe(sns.lineplot, x="level", y="performance", hue='hue')
        g.set_axis_labels("level", "Balanced Accuracy")
        g.add_legend()
        #plt.savefig(f"{df['dataset_name'].unique()[0]}_repeats.pdf")
        plt.show()
