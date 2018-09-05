import pandas as pd
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import Perceptron, SGDClassifier, PassiveAggressiveClassifier


# FILE = 'C:/bank/data_set/bank_train.csv'
FILE = 'C:/bank/data_set/bank_train_up.csv'


def extract_data(file_input, test_size=0.2):

    df_input = pd.read_csv(file_input)
    X = df_input.iloc[:, :-1]
    y = df_input.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    return X, y, X_train, y_train, X_test, y_test


def full_analysis(model_name, X, y, X_train, y_train, X_test, cv=5):
    model_name = model_name.upper()

    def model_selector(model_name):
        model_selected = {'TREE': DecisionTreeClassifier(criterion='entropy', max_depth=5),
                          'MLP': MLPClassifier(activation='logistic', hidden_layer_sizes=(60,), max_iter=1000),
                          'BNB': BernoulliNB(),
                          'PCP': Perceptron(max_iter=3000),
                          'SGD': SGDClassifier(max_iter=3000),
                          'PAC': PassiveAggressiveClassifier(max_iter=3000)}

        return model_selected[model_name]

    def fit_and_predict(target_model, X_train, y_train, X_test):
        result = target_model.fit(X_train, y_train)
        prediction = target_model.predict(X_test)
        return result, prediction

    target_model = model_selector(model_name)

    print(f"{model_name} training in session...")
    start_time = time.time()
    fit_and_predict(target_model, X_train, y_train, X_test)
    end_time = time.time()

    print("analyzing result...")
    time_ = end_time-start_time
    cv_ = cross_val_score(target_model, X, y, cv=cv)
    print(f"{model_name} training time: {time_}\n"
          f"{model_name} cross validation score:\n"
          f"{cv_}")


def main():
    X, y, X_train, y_train, X_test, _ = extract_data(FILE)

    full_analysis('tree', X, y, X_train, y_train, X_test)
    full_analysis('mlp', X, y, X_train, y_train, X_test)
    full_analysis('bnb', X, y, X_train, y_train, X_test)
    full_analysis('pcp', X, y, X_train, y_train, X_test)
    full_analysis('sgd', X, y, X_train, y_train, X_test)
    full_analysis('pac', X, y, X_train, y_train, X_test)


if __name__ == '__main__':
    main()
