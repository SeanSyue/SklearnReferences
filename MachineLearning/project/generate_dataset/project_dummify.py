import pandas as pd


def dummify_age(df):
    def trans_fn(x):
        if x < 30:
            return 'below30'
        elif 30 <= x <= 50:
            return '30to50'
        elif x > 50:
            return 'over50'
    trans = df[['age']].applymap(lambda x: trans_fn(x))
    age_dummy = pd.get_dummies(trans)
    return age_dummy


def dummify_campaign(df):
    def trans_fn(x):
        if x >= 3:
            return 'over2'
        else:
            return x
    trans = df[['campaign']].applymap(lambda x: trans_fn(x))
    df_dummy = pd.get_dummies(trans)
    return df_dummy


def dummify_pdays(df):
    def trans_fn(x):
        if x == 999:
            return 'nc'
        elif 0 <= x <= 6:
            return '0to6'
        elif x > 6:
            return 'over6'

    trans = df[['pdays']].applymap(lambda x: trans_fn(x))
    df_dummy = pd.get_dummies(trans)
    return df_dummy


def dummify_previous(df):
    def trans_fn(x):
        if x == 1:
            return 'once'
        elif x > 1:
            return 'over1'
    trans = df[['previous']].applymap(lambda x: trans_fn(x))
    df_dummy = pd.get_dummies(trans)
    return df_dummy


def dummify_price_idx(df):
    def trans_fn(x):
        if 92 < x < 93:
            return '92'
        elif 93 < x < 94:
            return '93'
        elif x > 94:
            return '94'
    trans = df[['cons.price.idx']].applymap(lambda x: trans_fn(x))
    price_idx_dummy = pd.get_dummies(trans, prefix='price_idx')
    return price_idx_dummy


def dummify_conf_idx(df):
    def trans_fn(x):
        if -51 < x < -41:
            return 'lo '
        elif -41 < x < -35:
            return 'avg'
        elif -35 < x < -26:
            return 'hi'
    trans = df[['cons.conf.idx']].applymap(lambda x: trans_fn(x))
    price_idx_dummy = pd.get_dummies(trans, prefix='conf_idx')
    return price_idx_dummy


def dummify_euribor3m(df):
    def trans_fn(x):
        if x < 3:
            return "lo"
        elif x > 3:
            return "hi"
    trans = df[['euribor3m']].applymap(lambda x: trans_fn(x))
    df_dummy = pd.get_dummies(trans, prefix='euri')
    return df_dummy


def dummify_nr_employed(df):
    def trans_fn(x):
        if x < 5050:
            return "lo"
        elif 5050 < x < 5200:
            return "mid"
        elif x > 5200:
            return "hi"
    trans = df[['nr.employed']].applymap(lambda x: trans_fn(x))
    df_dummy = pd.get_dummies(trans, prefix='n_employ')
    return df_dummy


# reader = pd.read_csv('D:/bank/data_set/bank-additional-full.csv', sep=";")
#
#
#
# df_obj = reader[['job', 'marital', 'education', 'default',
#                  'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']]
#
# concat = pd.concat([df_age, df_obj], axis=1)
# df_dummies = pd.get_dummies(concat)
# df_dummies = df_dummies[['job_admin.', 'age_30to65', 'age_below30', 'age_over65', 'job_blue-collar']]
#
# df_dummies.to_csv('D:/bank/data_set/dum_example.csv', index=False)
