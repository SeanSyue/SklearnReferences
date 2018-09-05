import pandas as pd
from sklearn.model_selection import train_test_split

# --------------------------------------Data Preprocessing--------------------------------------
data_frame = pd.read_csv("C:\\bank-additional-full.csv", sep=";")
cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week']
input_raw = data_frame[cols]  # type: DataFrame
X_df = pd.get_dummies(input_raw)  # Dummify
y_df = pd.DataFrame({'output': data_frame['y'].apply(lambda x: 1 if x == 'yes' else 0)})  # 1 for "yes", 0 for "no"
sheet = pd.concat([X_df, data_frame], axis=1)

# Print out parameters
print("concad:\n", sheet.head())
print("Input parameters:\n", X_df.columns.values, "\n")
print("Length of input para", len(X_df.columns.values))
print("Output parameters:\n", y_df.columns.values, "\n")


X = X_df.values
y = y_df.values

# Split arrays into random train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

print("input training data:\n", X_train[:10])
print("output label:\n", y_train[:10])
