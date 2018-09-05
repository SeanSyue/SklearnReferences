import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 80)

reader = pd.read_csv('D:/bank/dataset/bank-additional-full.csv', sep=";")
df_age = reader[['age']]

# df_str = df_age.astype(str)
# dummy = pd.get_dummies(df_str)
# dummy.to_csv('D:/bank/dataset/dummify_age_full.csv')

df_age_sub10 = df_age/10
df_str = df_age_sub10.astype(int).astype(str)
dummy = pd.get_dummies(df_str)
dummy.to_csv('D:/bank/dataset/dummify_age_sub10.csv')
