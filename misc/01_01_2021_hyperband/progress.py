import numpy as np

import pandas as pd

dataframe = []
for repeats in [1, 3, 10]:
    for cv in [5, 10]:
        initial_budget = 1
        max_budget = cv*repeats
        for eta in [2, 3]:
            s_max = int(np.floor(np.log(max_budget / initial_budget) / np.log(eta)))
            for s in reversed(range(s_max+1)):
                # compute min budget for new SH run
                sh_initial_budget = eta ** -s * max_budget
                # sample challengers for next iteration (based on HpBandster package)
                n_challengers = int(np.floor((s_max + 1) / (s + 1)) * eta ** s)
                dataframe.append({
                    'Repeats': repeats,
                    'cv': cv,
                    'eta': eta,
                    's': s,
                    'N': n_challengers,
                    'B': sh_initial_budget,
                })
df = pd.DataFrame(dataframe)
df.to_csv('Hyperband.csv')
