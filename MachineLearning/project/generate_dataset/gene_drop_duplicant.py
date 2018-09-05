import pandas as pd

reader = pd.read_csv('C:/Users/Sean/Desktop/bank/double_up7_original.csv')
reader.drop_duplicates(keep='first', inplace=True)
reader.to_csv('C:/Users/Sean/Desktop/bank/double_8000_no_duplicant.csv', index=False)
