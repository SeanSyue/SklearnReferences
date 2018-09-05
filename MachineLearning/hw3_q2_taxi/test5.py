import pandas as pd
import numpy as np

# dfs = pd.read_csv("D:\Taxi2/submission_train996.csv")
# Travel_Time = pd.DataFrame(columns=['TRAVEL_TIME'])
# Travel_Time = pd.concat([Travel_Time['TRAVEL_TIME'], dfs['TRAVEL_TIME']], axis=1)
# print(Travel_Time)
# a = np.mean(Travel_Time, axis=1)
# print(a)


prediction_files = []
dfs = []
for i in range(1000):
    prediction_files.append("submission_train{}.csv".format(i))
    dfs.append(pd.read_csv("D:/Taxi2/" + prediction_files[i]))

IDs = dfs[0]['TRIP_ID'].values
Travel_Time = pd.DataFrame(columns=['TRAVEL_TIME'])

for i in range(1000):
    Travel_Time = pd.concat([Travel_Time['TRAVEL_TIME'], dfs[i]['TRAVEL_TIME']], axis=1)

# Read the original test file
df_test = pd.read_csv("D:/Taxi2/test_final.csv")

# Ensemble all results
y_test = np.maximum(np.mean(Travel_Time, axis=1), (df_test['Current_Snapshots'] - 1) * 15)
print("Travel_Time.shape:\n", Travel_Time.shape)
print("y_test:\n", y_test)
print(type(IDs))
