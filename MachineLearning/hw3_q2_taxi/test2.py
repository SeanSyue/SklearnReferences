import pandas as pd
import re
import numpy as np
pd.set_option('display.max_columns', 50)
# pd.set_option('display.max_rows', 20)
pd.set_option('display.max_colwidth', 1000)
# read in the training set by chunks, and add engineered features
print("Pre-processing the training set:")
chunk_reader = pd.read_csv("D:/Taxi2/train.csv", chunksize=1000000)
chunk = chunk_reader.get_chunk(1000)
# print(chunk)

# reset index
chunk = chunk[chunk.MISSING_DATA == False]
chunk.reset_index(inplace=True)

# print("=====\n", chunk)

# split the polyline and calculate actual snapshots and travel time
chunk['POLYLINE_Split'] = chunk.POLYLINE.map(lambda x:
                                             re.compile("\[[-+]?\d+.\d+,[-+]?\d+.\d+\]").findall(x))
chunk['Snapshots'] = chunk.POLYLINE_Split.map(lambda x: len(x))
chunk = pd.DataFrame(chunk[chunk.Snapshots > 10])
chunk['Travel_Time'] = chunk['Snapshots'].map(lambda x: (x-1)*15)

# print("++++++\n", chunk)
# print("+++++++\n", chunk.POLYLINE[0])
print("+++++++\n", chunk.POLYLINE_Split[:3])


# Randomly truncate to match the format of the test data
def truncate_func(row):

    path_len = np.random.randint(1, row['Snapshots'])
    return tuple(row['POLYLINE_Split'][:path_len])


chunk['POLYLINE_Split_Truncated'] = chunk.apply(truncate_func, axis=1)

print(chunk['POLYLINE_Split_Truncated'][:3])
for i in range(0, 8):
    print("len of split:{}".format(i), len(chunk.POLYLINE_Split[i]))
    print("len of truncate:{}".format(i), len(chunk.POLYLINE_Split_Truncated[i]))
