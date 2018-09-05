import pandas as pd
chunk_reader = pd.read_csv("D:/Taxi2/train.csv", chunksize=1000000)
chunk = chunk_reader.get_chunk(1000)
dummy = pd.get_dummies(chunk.CALL_TYPE, prefix='Call_Type_')
print(dummy)
