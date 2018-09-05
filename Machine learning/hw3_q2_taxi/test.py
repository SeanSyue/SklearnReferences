import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 50)
df = pd.read_csv("D:/Taxi2/Train_Part991.csv")
df_X = df[['Call_Type__A', 'Call_Type__B', 'Call_Type__C',
                        'Time_of_Day', 'Hour_of_Day', 'Day_of_Week', 'Month_of_Year',
                        'Hour_TT', 'Hour_TL', 'Hour_TC', 'Hour_TS',
                        'Weekday_TT', 'Weekday_TL', 'Weekday_TC', 'Weekday_TS',
                        'Month_TT', 'Month_TL', 'Month_TC', 'Month_TS',
                        'Driver_TT', 'Driver_TL', 'Driver_TC', 'Driver_TS',
                        'Stand_TT', 'Stand_TL', 'Stand_TC', 'Stand_TS',
                        'Caller_TT', 'Caller_TL', 'Caller_TC', 'Caller_TS',
                        'Start_Speed', 'End_Speed', 'Avg_Speed', 'Start_Speed_two', 'End_Speed_two',
                        'Current_Snapshots', 'Current_Snapshots_log']]
y = np.log(df['Travel_Time']).values
# Randomly select subset of q*num_features attributes
column_list = df_X.columns.tolist()
num_features = len(column_list)
ind_selected = np.random.permutation(num_features)[:int(num_features * 0.75)]
# print(ind_selected)
# print(len(ind_selected))
# print(num_features * 0.75)
# feature_selected = [column_list[k] for k in ind_selected]
# print(feature_selected)
# print(len(feature_selected))
print(df.head())
print(df.columns)
print(len(df.columns))
print(type(df.columns))
