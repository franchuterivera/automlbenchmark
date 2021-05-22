import pandas as pd
import math
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure


# color dictionary
color = {
    0: 'r',
    1: 'b',
    2: 'g',
    3: 'y',
    4: 'm',
    5: 'c',
    6: 'k',
}

# ,hue,performance,model,dataset_name,level,repeat,seed,val_score
df = pd.read_csv('all_data.csv')
for col in ['level', 'repeat']:
    df[col] = df[col].astype(int)

df = df[df['level'] <=3]

# Create a test meand and std
#df_mean = df.groupby(['hue', 'dataset_name', 'level', 'repeat', 'multikfold']).mean().add_suffix('_mean')
#df_std = df.groupby(['hue', 'dataset_name', 'level', 'repeat', 'multikfold']).std()
#df_mean['performance_std'] = df_std['performance']
#df = df_mean.reset_index()
#
#df = df[(df['repeat'] == 5) & (df['hue'] == 'test_performance_singlemodelFalse')]
#
#fig = plt.figure(figsize=(18, 12))
#for i, dataset_name in enumerate(df['dataset_name'].unique()):
#    ax = fig.add_subplot(3, 3, i+1)
#    ax.set_title(f"Openml_id={dataset_name}")
#    ax.set(ylabel='Balanced Accuracy')
#    ax.set(xlabel='Level')
#    ax.grid(True)
#    for i, multikfold in enumerate(df['multikfold'].unique()):
#        this_data = df[(df['multikfold']==multikfold) & (df['dataset_name']==dataset_name)]
#        ax.errorbar('level', 'performance_mean', yerr='performance_std',  capsize=3, capthick=3, elinewidth=2, linewidth=1, linestyle='dashed', data=this_data, color=color[i], label=f"multikfold={multikfold}", alpha=0.5)
#plt.tight_layout()
#plt.legend(loc='lower center',  bbox_to_anchor=(-0.75, -0.35), ncol=7)
#plt.savefig(f"plot_multikfold.pdf")
#plt.show()
#plt.close()



# ,hue,performance,model,dataset_name,level,repeat,seed,val_score
df = pd.read_csv('all_data.csv')
for col in ['level', 'repeat']:
    df[col] = df[col].astype(int)

print(df['dataset_name'].unique())
df = df[(df['level'] <=3) & (df['dataset_name']== 146818)]
print(df)

# Create a test meand and std
df_mean = df.groupby(['hue', 'model', 'level', 'repeat', 'multikfold']).mean().add_suffix('_mean')
df_std = df.groupby(['hue', 'model', 'level', 'repeat', 'multikfold']).std()
df_mean['performance_std'] = df_std['performance']
df = df_mean.reset_index()

df = df[(df['repeat'] == 5) & (df['hue'] == 'test_performance_singlemodelFalse')]

fig = plt.figure(figsize=(18, 12))
for i, model in enumerate(df['model'].unique()):
    ax = fig.add_subplot(3, 3, i+1)
    ax.set_title(f"model={model}")
    ax.set(ylabel='Balanced Accuracy')
    ax.set(xlabel='Level')
    ax.grid(True)
    for i, multikfold in enumerate(df['multikfold'].unique()):
        this_data = df[(df['multikfold']==multikfold) & (df['model']==model)]
        ax.errorbar('level', 'performance_mean', yerr='performance_std',  capsize=3, capthick=3, elinewidth=2, linewidth=1, linestyle='dashed', data=this_data, color=color[i], label=f"multikfold={multikfold}", alpha=0.5)
plt.tight_layout()
plt.legend(loc='lower center',  bbox_to_anchor=(-0.75, -0.35), ncol=7)
plt.savefig(f"plot_multikfold_australian.pdf")
plt.show()



## Do the same per model
#
## ,hue,performance,model,dataset_name,level,repeat,seed,val_score
#df = pd.read_csv('all_data.csv')
#for model in df['model'].unique():
#    df = pd.read_csv('all_data.csv')
#    for col in ['level', 'repeat']:
#        df[col] = df[col].astype(int)
#    df = df[(df['level'] <=3) & (df['model'] == model)]
#
#    # Create a test meand and std
#    df_mean = df.groupby(['hue', 'dataset_name', 'level', 'repeat', 'multikfold']).mean().add_suffix('_mean')
#    df_std = df.groupby(['hue', 'dataset_name', 'level', 'repeat', 'multikfold']).std()
#    df_mean['performance_std'] = df_std['performance']
#    df = df_mean.reset_index()
#
#    df = df[(df['repeat'] == 5) & (df['hue'] == 'test_performance_singlemodelFalse')]
#
#    fig = plt.figure(figsize=(18, 12))
#    for i, dataset_name in enumerate(df['dataset_name'].unique()):
#        ax = fig.add_subplot(3, 3, i+1)
#        ax.set_title(f"Openml_id={dataset_name}/{model}")
#        ax.set(ylabel='Balanced Accuracy')
#        ax.set(xlabel='Level')
#        ax.grid(True)
#        for i, multikfold in enumerate(df['multikfold'].unique()):
#            this_data = df[(df['multikfold']==multikfold) & (df['dataset_name']==dataset_name)]
#            ax.errorbar('level', 'performance_mean', yerr='performance_std',  capsize=3, capthick=3, elinewidth=2, linewidth=1, linestyle='dashed', data=this_data, color=color[i], label=f"multikfold={multikfold}", alpha=0.5)
#    plt.tight_layout()
#    plt.legend(loc='lower center',  bbox_to_anchor=(-0.75, -0.35), ncol=7)
#    plt.savefig(f"plot_multikfold_{model}.pdf")
#    plt.show()
#    plt.close()
