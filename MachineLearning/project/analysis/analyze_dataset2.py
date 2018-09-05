# import pandas as pd
# import numpy as np
#
#
# def examine_data_frame(df):
#     with open("D:/bank/dataset/examine_data_frame.txt", 'a') as text_file:
#         for name in df.columns:
#             print("----------", file=text_file)
#             print(df[name].dtype, file=text_file)
#             if df[name].dtype is np.dtype('O'):
#                 print(df[name].value_counts(), file=text_file)
#             else:
#                 print(df[name].describe(), file=text_file)
#
#
# # def examine_data_frame(df):
# #     for name in df.columns:
# #         print("----------")
# #         print(df[name].dtype)
# #         if df[name].dtype is np.dtype('O'):
# #             print(df[name].value_counts())
# #         else:
# #             print(df[name].describe())
#
#
# reader = pd.read_csv('D:/bank/dataset/bank-additional-full.csv', sep=";")
# examine_data_frame(reader)


