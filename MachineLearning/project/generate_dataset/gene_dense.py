import pandas as pd


def trans_age(x):
    if x < 30:
        return 0
    elif 30 <= x <= 50:
        return 1
    elif x > 50:
        return -1


def trans_duraion(x):
    if x < 400:
        return -1
    else:
        return 1


def trans_pdays(x):
    if x == 999:
        return 1
    else:
        return -1


def trans_campaign(x):
    if x < 3:
        return 1
    else:
        return -1


def trans_previous(x):
    if x < 2:
        return 1
    else:
        return -1


def dense_transform(reader):
    reader['age'] = reader['age'].apply(lambda x: trans_age(x))

    reader['job'].replace(('admin.', 'blue-collar', 'technician', 'services', 'management', 'retired',
                           'entrepreneur', 'self-employed', 'housemaid', 'unemployed', 'student', 'unknown'),
                          (-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6), inplace=True)

    reader['marital'].replace(('married', 'single', 'divorced', 'unknown'), (-2, -1, 1, 2), inplace=True)

    reader['education'].replace(('university.degree', 'high.school', 'basic.9y', 'professional.course',
                                 'basic.4y', 'basic.6y', 'unknown', 'illiterate'),
                                (-4, -3, -2, -1, 1, 2, 3, 4), inplace=True)

    reader['default'].replace(('no', 'unknown', 'yes'), (-1, 0, 1), inplace=True)

    reader['housing'].replace(('no', 'unknown', 'yes'), (-1, 0, 1), inplace=True)

    reader['loan'].replace(('no', 'unknown', 'yes'), (-1, 0, 1), inplace=True)

    reader['contact'].replace(('cellular', 'telephone'), (-1, 1), inplace=True)

    reader['month'].replace(('mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'),
                            (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5), inplace=True)

    reader['day_of_week'].replace(('mon', 'tue', 'wed', 'thu', 'fri'), (-2, -1, 0, 1, 2), inplace=True)

    reader['duration'] = reader['duration'].apply(lambda x: trans_duraion(x))

    reader['campaign'] = reader['campaign'].apply(lambda x: trans_campaign(x))

    reader['pdays'] = reader['pdays'].apply(lambda x: trans_pdays(x))

    reader['previous'] = reader['previous'].apply(lambda x: trans_previous(x))

    reader['poutcome'].replace(('failure', 'nonexistent', 'success'), (-1, 0, 1), inplace=True)

    reader['y'].replace(('no', 'yes'), (-1, 1), inplace=True)


reader = pd.read_csv('C:/bank/data_set/for_dense.csv')

# ------------------------------------------------------------------------------------------------------------------

dense_transform(reader)

# ------------------------------------------------------------------------------------------------------------------
print(reader)
# reader.to_csv('C:/bank/data_set/bank_dense.csv', index=False)
