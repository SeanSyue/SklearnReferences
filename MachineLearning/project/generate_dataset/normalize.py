import pandas as pd
import numpy as np

reader = pd.read_csv('D:/bank/data_set/bank-additional-full.csv', sep=";")
df = reader[['duration', 'campaign', 'pdays', 'previous',
             'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']]


df_normalized = df.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
# df_normalized = df_normalized[['duration', 'campaign', 'pdays', 'previous', 'emp.var.rate']]

df_normalized.to_csv('D:/bank/data_set/normalized.csv', index=False)
