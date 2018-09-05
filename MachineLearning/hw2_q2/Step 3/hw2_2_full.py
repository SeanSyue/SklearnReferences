import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# --------------------------------------Data Preprocessing--------------------------------------
data_frame = pd.read_csv("C:\\bank-additional-full.csv", sep=";")
cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week']
input_raw = data_frame[cols]  # type: DataFrame
X_df = pd.get_dummies(input_raw)  # Dummify
y_df = pd.DataFrame({'output': data_frame['y'].apply(lambda x: 1 if x == 'yes' else 0)})  # 1 for "yes", 0 for "no"
# Print out parameters
print("Input parameters:\n", X_df.columns.values, "\n")
print("Output parameters:\n", y_df.columns.values, "\n")
# Turn into arrays
X = X_df.values
y = y_df.values
# Split arrays into random train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# # --------------------------------------Standardization--------------------------------------
# #  Removes the mean and scales the data to unit variance.
# scaler = StandardScaler()
# # Fit only to the training data
# scaler.fit(X_train)
# # Apply the transformations to the data:
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)
# # Print the amount of Training Set and Testing set for X and y respectively
# print("The amount of X_train", len(X_train))
# print("The amount of y_train:", len(y_train))
# print("The amount of X_test", len(X_test))
# print("The amount of y_test:", len(y_test), "\n")

# --------------------------------------Decision tree analysis------------------------------------
# Training
tree = DecisionTreeClassifier(criterion='entropy', max_depth=5)  # Maximum depth equals 5
tree_result = tree.fit(X_train, y_train)
# Prediction
tree_pred = tree.predict(X_test)

# --------------------------------------Multi-layer perceptron analysis----------------------------
# Training
mlp = MLPClassifier(activation='logistic', hidden_layer_sizes=(80, 80,))  # 2 layers with the same number of neurons
mlp_result = mlp.fit(X_train, y_train)
# Prediction
mlp_pred = mlp.predict(X_test)

# --------------------------------------Evaluation results--------------------------------------
# Using cross validation for 5 times.
# Each item stands for each CV result
print("mlp_cross_val_score:\n", cross_val_score(mlp, X, y, cv=5))
print("tree_cross_val_score:\n", cross_val_score(tree, X, y, cv=5))
