import pandas as pd
import numpy as np
from MachineLearning.project.generate_dataset import project_dummify


def pre_processor(df, drop_duration=False):

    def func(x):
        if x < 30:
            return 'below30'
        elif 30 <= x <= 65:
            return '30to65'
        elif x > 65:
            return 'over65'

    df_age_raw = df[['age']]
    df_age = df_age_raw.applymap(lambda x: func(x))

    df_obj = df[['job', 'marital', 'education', 'default',
                 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']]

    concat = pd.concat([df_age, df_obj], axis=1)
    df_dummies = pd.get_dummies(concat)

    if drop_duration is True:
        df_num = df[['duration', 'campaign', 'pdays', 'previous',
                     'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']]
    else:
        df_num = df[['campaign', 'pdays', 'previous',
                     'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']]

    df_norm = df_num.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))

    df_label = df['y'].apply(lambda x: 1 if x == 'yes' else 0)

    # df_out = pd.concat([df_dummies, df_norm, df_label], axis=1)
    df_out = pd.concat([df_dummies, df_label], axis=1)

    return df_out


reader_train = pd.read_csv('C:/bank/data_set/origin_data_sets/bank-additional-full.csv', sep=";")
reader_test = pd.read_csv('C:/bank/data_set/origin_data_sets/bank-additional.csv', sep=";")

df_train = reader_train.append(reader_test).drop_duplicates(keep=False).reset_index(drop=True)

train_out = pre_processor(df_train)
test_out = pre_processor(reader_test)

# train_out.to_csv('C:/bank/data_set/bank_dummy_train.csv', index=False)
# test_out.to_csv('C:/bank/data_set/bank_dummy_test.csv', index=False)

reader = pd.read_csv('C:/bank/data_set/bank_dummy_train.csv')
print(reader.shape)

# train_out.to_csv('D:/bank/data_set/bank_train_duration_dropped.csv', index=False)
# test_out.to_csv('D:/bank/data_set/bank_test_duration_dropped.csv', index=False)

# train_out = train_out[['contact_cellular', 'poutcome_failure', 'duration', 'pdays', 'previous']]
# train_out.to_csv('D:/bank/data_set/mix_example.csv', index=False)
