import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# --------------------------------------Data Preprocessing--------------------------------------
data_frame = pd.read_csv("C:\\bank-additional-full.csv", sep=";")
cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week']
input_raw = data_frame[cols]  # type: DataFrame
X_df = pd.get_dummies(input_raw)  # Dummify
y_df = pd.DataFrame({'output': data_frame['y'].apply(lambda x: 1 if x == 'yes' else 0)})  # 1 for "yes", 0 for "no"
# Turn into arrays
X = X_df.values
y = y_df.values
# Split arrays into random train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)


# --------------------------------------Standardization--------------------------------------
# #  Removes the mean and scales the data to unit variance.a
# scaler = StandardScaler()
# # Fit only to the training data
# scaler.fit(X_train)
# # Apply the transformations to the data:
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)
# # print("len(X_train):", len(X_train))
# # print("len(y_train):", len(y_train))


# --------------------------------------Multi-layer perceptron analysis----------------------------
# Training with a multi-layer perceptron with one hidden layer.
mlp = MLPClassifier(hidden_layer_sizes=(10,))
mlp_result = mlp.fit(X_train, y_train)
# Prediction
mlp_pred = mlp.predict(X_test)
# Using cross validation for 5 times. Each item stands for each CV result
print("Cross validation score:\n", cross_val_score(mlp, X, y, cv=5))

print("n_layers_ :", mlp.n_layers_)
print("n_iter_:", mlp.n_iter_)
