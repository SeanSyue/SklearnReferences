import pandas as pd


def dummify_y(file):
    reader = pd.read_csv(file)
    y_dummy = pd.get_dummies(reader['y'])
    reader['y_no'] = y_dummy.iloc[:, :1]
    reader['y_yes'] = y_dummy.iloc[:, :-1]
    df_out = reader.drop('y', axis=1)
    return df_out


train_file = 'C:/bank/data_set/dataset_v1/bank_train_up.csv'
train_out = dummify_y(train_file)
