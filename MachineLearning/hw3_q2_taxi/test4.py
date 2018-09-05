import pandas as pd
pd.set_option('display.max_columns', 50)
# pd.set_option('display.max_rows', 20)
pd.set_option('display.max_colwidth', 1000)

speed_dict = {'Start_Speed': 2.255119,
              'End_Speed': 7.652231,
              'Avg_Speed': 6.905948,
              'Start_Speed_Two': 4.302278,
              'End_Speed_Two': 7.619596}
chunk_reader = pd.read_csv('D:/Taxi2/train_cleaned_temp.csv', chunksize=10000)
chunk = chunk_reader.get_chunk(10)


chunk = chunk[(chunk.Start_Speed <= 40) & (chunk.End_Speed <= 40) & (chunk.Avg_Speed <= 40) &
              (chunk.Start_Speed_two <= 40) & (chunk.End_Speed_two <= 40) & (chunk.Current_Snapshots < 1000)]
print(chunk)
print(chunk.shape)
# chunk.reset_index(inplace=True)
# chunk.drop(['index'], axis=1, inplace=True)
#
# chunk.loc[chunk.Start_Speed.isnull(), 'Start_Speed'] = speed_dict['Start_Speed']
# chunk.loc[chunk.End_Speed.isnull(), 'End_Speed'] = speed_dict['End_Speed']
# chunk.loc[chunk.Avg_Speed.isnull(), 'Avg_Speed'] = speed_dict['Avg_Speed']
# chunk.loc[chunk.Start_Speed_two.isnull(), 'Start_Speed_two'] = speed_dict['Start_Speed_Two']
# chunk.loc[chunk.End_Speed_two.isnull(), 'End_Speed_two'] = speed_dict['End_Speed_Two']
