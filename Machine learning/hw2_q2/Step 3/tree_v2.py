import pandas as pd
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
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
# Split arrays into random train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)


# --------------------------------------Decision tree analysis----------------------------
# Training with a decision tree
tree = DecisionTreeClassifier(criterion='entropy', max_depth=10)
tree_result = tree.fit(X_train, y_train)
# Prediction
tree_pred = tree.predict(X_test)
# Using cross validation for 5 times. Each item stands for each CV result
print("Cross validation score:", cross_val_score(tree, X, y, cv=5))


# --------------------------------------Importance analysis----------------------------
importance = tree.feature_importances_
print("\nImportances:\n", importance)
print("type of feature importance:", type(importance))
print("length of feature importance:", len(importance))


importance_df = X_df[0:0].append(pd.Series(importance.tolist(), index=list(X_df.columns.values)),
                                 ignore_index=True)
print("22\n", importance_df.iloc[:, :10])
print("importance df:\n", importance_df)
importance_df.set_index('job_admin.', inplace=True)
print("importance df modified:\n", importance_df)

mean = importance_df.mean(axis=0)
print("The most important one:", mean.idxmax(axis=0))
print("Value:", mean.max(axis=0))
