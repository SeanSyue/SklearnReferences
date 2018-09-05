import numpy as np
import pandas as pd

# pd.Dataframe.describe(): Generates descriptive statistics that summarize the central tendency, /
# dispersion and shape of a dataset’s distribution, excluding NaN values.


def examine_data_frame(df):
    for name in df.columns:
        print("==========\n--", df[name].dtype, "--")
#        print (df[ name].dtype)
        if df[ name].dtype is np.dtype('O'):
            print(df[name].value_counts())
            print("Name: ", name)
        else:
            print(df[ name].describe())
        print("==========\n")


data_frame = pd.read_csv("C:\\bank-additional-full.csv", sep=";")

examine_data_frame(data_frame)
