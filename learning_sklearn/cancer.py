from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print(cancer.keys())
print(type(cancer.keys()))
# print(cancer['DESCR'])
print(cancer['data'].shape)  # Return array dim in tuple form.

X = cancer['data']
y = cancer['target']

# Split arrays or matrices into random train and test subsets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++Standardization++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit only to the training data
scaler.fit(X_train)
print(scaler.fit(X_train), "\n")

# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print("type(X_train):", type(X_train))  # <class 'numpy.ndarray'>
print("type(X_test):", type(X_test))  # <class 'numpy.ndarray'>
print("type(y_train):", type(y_train))  # <class 'numpy.ndarray'>
print("type(y_test):", type(y_test))  # <class 'numpy.ndarray'>
print("X_train.shape:", X_train.shape)
print("X_test.shape:", X_test.shape)
print("y_train.shape:", y_train.shape)
print("y_test.shape:", y_test.shape, "\n")  # Not using len()

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++Training+++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(30, 30, 30))  # 3 layers with the same number of neurons
mlp.fit(X_train, y_train)
print(mlp.fit(X_train, y_train), "\n")

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++Prediction+++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
predictions = mlp.predict(X_test)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++Evaluation++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from sklearn.metrics import classification_report, confusion_matrix
print("++++++++++++++++++++Evaluation++++++++++++++++++++")
print("confusion_matrix: \n", confusion_matrix(y_test, predictions))
print("--------------------------------------------------")
print("classification_report: \n", classification_report(y_test, predictions))
from sklearn.model_selection import cross_val_score
print("cross_val_score: \n", cross_val_score(mlp, X, y, cv=5))

# ++++++++++++++++++++Weights and biases++++++++++++++++++++
# coefs_ is a list of weight matrices
# intercepts_ is a list of bias vectors
# print("00000000000000000000000000000000000000000000000000000")
# print(mlp.coefs_)
# print("----------------------------------------------------")
# print(mlp.coefs_[0])
# print("----------------------------------------------------")
# print(mlp.intercepts_[0])
# print("----------------------------------------------------")
# print(len(mlp.coefs_))
# print("----------------------------------------------------")
# print(len(mlp.coefs_[0]))
# print("----------------------------------------------------")
# print(len(mlp.intercepts_[0]))
