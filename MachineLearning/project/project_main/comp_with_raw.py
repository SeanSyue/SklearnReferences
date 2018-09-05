import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

# --------------------------------------Data Preprocessing--------------------------------------
df_input = pd.read_csv("C:/bank/data_set/translated.csv")
# print(df_input.iloc[:-1, :-1].columns.values)
# print(df_input.iloc[:-1, :-1].shape)
# print(df_input.iloc[:-1, -1])
# print(df_input.iloc[:-1, -1].shape)
# df_input['poutcome'].apply(lambda x: 'ne' if x == 'nonexistent' else None)
print(df_input['poutcome'])
X = df_input.iloc[:-1, :-1]
y = df_input.iloc[:-1, -1]

# Split arrays into random train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# --------------------------------------Decision tree analysis------------------------------------
# Training
tree = DecisionTreeClassifier(criterion='entropy', max_depth=5)  # Maximum depth equals 5
tree_result = tree.fit(X_train, y_train)
# Prediction
tree_pred = tree.predict(X_test)

# --------------------------------------Multi-layer perceptron analysis----------------------------
# Training
mlp = MLPClassifier(activation='logistic', hidden_layer_sizes=(60,), max_iter=1000)
mlp_result = mlp.fit(X_train, y_train)
# Prediction
mlp_pred = mlp.predict(X_test)

# --------------------------------------Evaluation results--------------------------------------
# Using cross validation for 5 times.
# Each item stands for each CV result
print("mlp_cross_val_score:\n", cross_val_score(mlp, X, y, cv=5))
print("tree_cross_val_score:\n", cross_val_score(tree, X, y, cv=5))
