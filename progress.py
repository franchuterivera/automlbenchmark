import numpy as np

import pandas as pd

dataframe = []
for repeats in [50]:
    initial_budget = 10
    max_budget = repeats
    for eta in [2, 3, 4]:
        s_max = int(np.floor(np.log(max_budget / initial_budget) / np.log(eta)))
        for s in reversed(range(s_max+1)):
            # compute min budget for new SH run
            sh_initial_budget = eta ** -s * max_budget
            # sample challengers for next iteration (based on HpBandster package)
            n_challengers = int(np.floor((s_max + 1) / (s + 1)) * eta ** s)
            dataframe.append({
                'Repeats': repeats,
                'eta': eta,
                's': s,
                'N': n_challengers,
                'B': sh_initial_budget,
            })
df = pd.DataFrame(dataframe)
df.to_csv('Hyperband.csv')
print(df)
