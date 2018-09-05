import pandas as pd
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report


pd.set_option('display.max_columns', 66)

# FILE = 'C:/bank/data_set/bank_train_up.csv'
# FEATURE_COUNT = 65

# FILE = 'C:/bank/data_set/bank_train_duration_dropped_up.csv'
# FEATURE_COUNT = 64

# FILE = 'C:/bank/data_set/bank_train_drop_most_four_up.csv'
# FEATURE_COUNT = 61

# FILE = 'C:/bank/data_set/bank_train_drop_least_two_up.csv'
# FEATURE_COUNT = 63

# FILE = 'C:/bank/data_set/bank_train_drop_head_and_tail_up.csv'
# FEATURE_COUNT = 59

# FILE = 'C:/bank/data_set/bank_train_drop_least_thirteem_up.csv'
# FEATURE_COUNT = 52

# FILE = 'C:/bank/data_set/bank_train_drop_duration_least_thirteem_up.csv'
# FEATURE_COUNT = 51


# =============================Version 2=============================

# FILE = 'C:/bank/data_set/benchmark/bank_benchmark.csv'
# FEATURE_COUNT = 64

# FILE = 'C:/bank/data_set/benchmark/bank_benchmark2.csv'
# FEATURE_COUNT = 64

# FILE = 'C:/bank/data_set/dataset_v2/client_train.csv'
# FEATURE_COUNT = 61

# FILE = 'C:/bank/data_set/dataset_v2/client_train_up.csv'
# FEATURE_COUNT = 61

# FILE = 'C:/bank/data_set/dataset_v2/client_train_up8.csv'
# FEATURE_COUNT = 61

# FILE = 'C:/bank/data_set/dataset_v2/client_train_up9.csv'
# FEATURE_COUNT = 61

# FILE = 'C:/bank/data_set/dataset_v2/client_train_up10.csv'
# FEATURE_COUNT = 61

# FILE = 'C:/bank/data_set/dataset_v2/client_train_up11.csv'
# FEATURE_COUNT = 61

# =============================Version 3=============================
ROOT = 'C:/bank/data_set/dataset_v3/'
FOLDER = ROOT+'client_raw/'

TRAIN_FILE = FOLDER+'client_raw_train.csv'
TEST_FILE = FOLDER+'client_raw_test.csv'
FULL_FILE = FOLDER+'client_raw_full.csv'


def compact_reader(train_file, test_file, full_file):
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    full_data = pd.read_csv(full_file)
    return train_data, test_data, full_data


def data_splitter(train_in, test_in, full_in):
    X_train = train_in.iloc[:, :-1]
    y_train = train_in.iloc[:, -1]
    X_test = test_in.iloc[:, :-1]
    y_test = test_in.iloc[:, -1]
    X = full_in.iloc[:, :-1]
    y = full_in.iloc[:, -1]
    return X_train, y_train, X_test, y_test, X, y


def decision_tree_analysis(X_train, y_train, X_test):
    # Training
    tree_clf = DecisionTreeClassifier(max_depth=35)
    tree_result = tree_clf.fit(X_train, y_train)
    # Prediction
    tree_pred = tree_clf.predict(X_test)
    return tree_clf, tree_result, tree_pred


def main():
    # -----------------------------------------Preprocessing---------------------------------------
    train_in, test_in, full_in = compact_reader(TRAIN_FILE, TEST_FILE, FULL_FILE)
    X_train, y_train, X_test, y_test, X, y = data_splitter(train_in, test_in, full_in)

    # -----------------------------------------Train steps-----------------------------------------
    print("Decision tree training in session...")
    tree_start = time.time()
    tree_clf, _, y_pred = decision_tree_analysis(X_train, y_train, X_test)
    tree_end = time.time()

    # --------------------------------------Evaluation results-------------------------------------
    # with open('time_report.txt', 'a') as t_report:
    #     print("time,", tree_start-tree_end, file=t_report)
    # with open('cross_validation_report.txt', 'a') as cv_report:
    #     print("tree_cross_val_score:\n", cross_val_score(tree_clf, X, y, cv=5), file=cv_report)
    # with open('classification_report.txt', 'a') as cls_report:
    #     print("classification_report:\n", classification_report(y_test, y_pred), file=cls_report)
    print("time,", tree_start-tree_end)
    print("tree_cross_val_score:\n", cross_val_score(tree_clf, X, y, cv=5))
    print("classification_report:\n", classification_report(y_test, y_pred))

    # imp = tree_clf.feature_importances_
    # feature_arr = df_input.drop(['y'], axis=1).columns.values
    # feature_arr = feature_arr.reshape((1, FEATURE_COUNT))[0]
    # imp = imp.reshape((1, FEATURE_COUNT))
    # imp_df = pd.DataFrame(imp, columns=feature_arr)
    # print(imp_df)


if __name__ == '__main__':
    main()
