import pandas as pd

dataframe = pd.read_csv("C:\\bank-additional-full.csv", sep=";")
cols =['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
data_1 = dataframe[cols]
data_dummies = pd.get_dummies(data_1)
result_df = pd.concat([data_dummies, dataframe], axis=1)
result_df['output'] = result_df['y'].apply(lambda x: 1 if x == 'yes' else 0)

grouped = result_df.groupby("y")
# print(grouped.groups)
nr_employed = grouped["nr.employed"].describe()
print(nr_employed)
# nr_employed2 = nr_employed.unstack(level=-1)
# print(nr_employed2)
