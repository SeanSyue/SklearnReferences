import pandas as pd
import sqlite3
from pandas.io import sql

reader = pd.read_csv('D:/bank/data_set/bank-additional-full.csv', sep=";")
con = sqlite3.connect('D:/bank/data_set/age_job.sqlite')
df_age_job = reader[['age', 'job']]
sql.to_sql(df_age_job, con=con, name='age_job')
sel = pd.read_sql("SELECT * FROM age_job WHERE job='retired' ORDER BY age ASC", con)
print(sel)
