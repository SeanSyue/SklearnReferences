
import pandas as pd

reader = pd.read_csv('D:/bank/dataset/bank-additional-full.csv', sep=";")
df1 = reader[['duration', 'campaign', 'pdays', 'previous']]
# df2 = reader[['emp.var.rate'], ['cons.price.idx'], ['cons.conf.idx'], ['euribor3m'], ['nr.employed']]


for name in df1.columns:
        print("----------")
        print(df1[name].dtype)
        print(df1[name].value_counts())
