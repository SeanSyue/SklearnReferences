import pandas as pd

data_frame = pd.read_csv("C:\\bank-additional-full.csv", sep=";")

cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
print("=====================================================================================")
data_1 = data_frame[cols]
print(data_1.head())  # type: DataFrame
print("=====================================================================================")
data_dummies = pd.get_dummies(data_1)
print(data_dummies.head())  # type: DataFrame
print("=====================================================================================\n concatenate:")
# To concatenate along a certain axis, axis=1 for concatenating along columns.
result_df = pd.concat([data_dummies, data_frame], axis=1)  # type: #DataFrame, row: 41188, col: 74
print(result_df.head())
print("type:", type(result_df))
print("shape:", result_df.shape)
print("=====================================================================================")
print(result_df.columns.values)
print("=====================================================================================")
# Applies function along input axis of DataFrame. If result_df['y']='yes', then result_df['output']=1.
result_df['output'] = result_df['y'].apply(lambda x: 1 if x == 'yes' else 0)
print(result_df['output'].head(10))
print("type:", type(result_df['output']))


print("==++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++==")
# x = result_df.loc[:, :'nr.employed']
# print(x.head())
y = pd.DataFrame({'output': result_df.output})
print(y.head())
print("y type", type(y))
