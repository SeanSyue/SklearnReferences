import pandas as pd

data_frame = pd.read_csv("C:\\bank-additional-full.csv", sep=";")
print ("=====================================================================================")
print (data_frame.columns.values)
print ("=====================================================================================")
print (data_frame.info())
print ("=====================================================================================")
print (data_frame.head(10))




