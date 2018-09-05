import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import time

pd.set_option('display.max_columns', 66)
FILE = 'C:/bank/data_set/bank_train_up.csv'



def mlp_analysis(X_train, y_train, X_test):
    # Training
    mlp = MLPClassifier(activation='logistic', hidden_layer_sizes=(60,), max_iter=1000)
    mlp_result = mlp.fit(X_train, y_train)
    # Prediction
    mlp_pred = mlp.predict(X_test)
    return mlp, mlp_result, mlp_pred


# -----------------------------------------Preprocessing---------------------------------------
# Read csv file.
df_input = pd.read_csv(FILE)
# Mark out features and label.
X = df_input.iloc[:, :-1]
y = df_input.iloc[:, -1]
# Split train and test.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


print("MLP training in session...")
mlp_start = time.time()
mlp, _, y_pred = mlp_analysis(X_train, y_train, X_test)
mlp_end = time.time()

print("mlp train time:", mlp_end-mlp_start)
print("mlp_cross_val_score:\n", cross_val_score(mlp, X, y, cv=5))
print("classification_report:\n", classification_report(y_test, y_pred))
