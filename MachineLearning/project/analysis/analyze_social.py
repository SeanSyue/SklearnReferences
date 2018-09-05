import pandas as pd

reader = pd.read_csv('C:/bank/data_set/analyze/social.csv')
reader2 = reader.set_index('month', drop=False)
# print(reader2.index.unique().values[2])
# print(reader2.loc['mar', 'cons.conf.idx'].value_counts())
# print(reader2['month'].value_counts())


def printtttt(df):
    idx_q = df.index.unique()
    i = 0
    for idx in idx_q:
        print(f"=========================================\n"
              f"==============index:{idx}==================\n"
              f"=========================================\n"
              f"---------------------------------")
        for col in df.columns:
            while idx_q.values[i] == idx:
                print(f"{df.loc[idx, col].value_counts()}\n---------------------------------")
                break
        i += 1


printtttt(reader2)

# name = reader2['month'].unique()
# mar = reader2.loc[name == 'mar']
# print(type(name))

