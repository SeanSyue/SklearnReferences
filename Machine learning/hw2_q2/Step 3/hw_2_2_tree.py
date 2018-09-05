import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# --------------------------------------Data Preprocessing--------------------------------------
data_frame = pd.read_csv("C:\\bank-additional-full.csv", sep=";")
cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week']
input_raw = data_frame[cols]  # type: DataFrame
X_df = pd.get_dummies(input_raw)  # Dummify
y_df = pd.DataFrame({'output': data_frame['y'].apply(lambda x: 1 if x == 'yes' else 0)})  # 1 for "yes", 0 for "no"
# Turn into arrays
X = X_df.values
y = y_df.values
# print("X_df\n", X_df)
# print("X", X)
# print("type of X", type(X))
# Split arrays into random train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# --------------------------------------Standardization--------------------------------------
# #  Removes the mean and scales the data to unit variance.
# scaler = StandardScaler()
# # Fit only to the training data
# scaler.fit(X_train)
# # Apply the transformations to the data:
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)

# --------------------------------------Multi-layer perceptron analysis----------------------------
# Training with a decision tree
tree = DecisionTreeClassifier(criterion='entropy', max_depth=10)
tree_result = tree.fit(X_train, y_train)
# Prediction
tree_pred = tree.predict(X_test)
# Using cross validation for 5 times. Each item stands for each CV result
print("Cross validation score:\n", cross_val_score(tree, X, y, cv=5))

importance_series = tree.feature_importances_
print("shape:", importance_series.shape)
importance_rounded = importance_series.round(decimals=3)
print("importance_rounded\n", importance_rounded)
print("type of importance_rounded", type(importance_rounded))
importance_tolist = importance_rounded.tolist()
print("importance_tolist: ", importance_tolist)
print("type of importance_tolist", type(importance_tolist))
print("len of importance_tolist:", len(importance_tolist))
importance_table = X_df.append(importance_tolist, ignore_index=True)
print("importance_table\n", importance_table)
print("type of importance table", type(importance_table))
