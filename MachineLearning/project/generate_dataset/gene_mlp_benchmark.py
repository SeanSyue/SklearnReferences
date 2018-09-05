import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize


def trans_pdays_reg(x):
    if x == 999:
        return 0
    else:
        return x


def trans_pdays_inf(x):
    if x == 999:
        return 1
    else:
        return -1


def norm_fn(x):
    return (2*(x - np.min(x)) / (np.max(x) - np.min(x)))-1


def trans_y(x):
    if x == 'yes':
        return 1
    elif x == 'no':
        return -1


reader = pd.read_csv('C:/bank/data_set/benchmark/for_benchmark2.csv')


df_dummy = pd.get_dummies(reader.iloc[:, :10])

df_pdays_inf = pd.DataFrame({'pdays_inf': reader['pdays'].apply(lambda x: trans_pdays_inf(x))})

df_pdays_reg = reader[['pdays']].applymap(lambda x: trans_pdays_reg(x))
# ---- Alternative: ----
# replace_ = reader.replace(999, 0)
# df_pdays_reg = replace_[['pdays']]
df_norm = pd.concat([df_pdays_reg, reader.iloc[:, -10:-1]], axis=1).apply(lambda x: norm_fn(x))
# ---- Alternative: ----
# norm_ = normalize(pd.concat([df_pdays_reg, reader.iloc[:, -10:-1]], axis=1))
# df_norm = pd.DataFrame(norm_, columns=reader.iloc[:, -11:-1].columns)

df_y = reader['y'].apply(lambda x: trans_y(x))
# ---- Alternative: ----
# df_y = reader['y'].replace(('yes', 'no'), (1, -1))

bench = pd.concat([df_dummy, df_pdays_inf, df_norm, df_y], axis=1)


bench.to_csv('C:/bank/data_set/benchmark/bank_benchmark_mlp.csv', index=False)
