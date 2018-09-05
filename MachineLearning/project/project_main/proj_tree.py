import pandas as pd
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
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

FILE = 'C:/bank/data_set/benchmark/bank_benchmark.csv'
FEATURE_COUNT = 64

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


def data_splitter(df_input):
    X = df_input.iloc[:, :-1]
    y = df_input.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X, y, X_train, X_test, y_train, y_test


def decision_tree_analysis(X_train, y_train, X_test):
    # Training
    tree_clf = DecisionTreeClassifier(max_depth=35)
    tree_result = tree_clf.fit(X_train, y_train)
    # Prediction
    tree_pred = tree_clf.predict(X_test)
    return tree_clf, tree_result, tree_pred


def main():
    # -----------------------------------------Preprocessing---------------------------------------
    # Read csv file.
    df_input = pd.read_csv(FILE)
    # Split to train and test data.
    X, y, X_train, X_test, y_train, y_test = data_splitter(df_input)

    # -----------------------------------------Train steps-----------------------------------------

    print("Decision tree training in session...")
    tree_start = time.time()
    tree_clf, _, y_pred = decision_tree_analysis(X_train, y_train, X_test)
    tree_end = time.time()

    # --------------------------------------Evaluation results-------------------------------------

    # print("tree train time:", tree_end - tree_start)
    with open("cross_validation_report.txt", 'a') as cv_report:
        print("tree_cross_val_score:\n", cross_val_score(tree_clf, X, y, cv=5), file=cv_report)
    with open("classification_report.txt", 'a') as cls_report:
        print("classification_report:\n", classification_report(y_test, y_pred), file=cls_report)


    # cv = cross_val_score(tree_clf, X, y, cv=5)
    # print(str(cv))




    # imp = tree_clf.feature_importances_
    # print("max_features_ :", tree_clf.max_features_)
    # print("classes_ :", tree_clf.classes_)
    # print("n_classes_ :", tree_clf.n_classes_)
    # print("n_features_ :", tree_clf.n_features_)
    # print("n_outputs_ :", tree_clf.n_outputs_)
    # print("tree_ :", tree_clf.tree_)

    # dot_data = export_graphviz(tree_clf, out_file=None, feature_names=feature_arr, class_names=['y'])
    # graph = graphviz.Source(dot_data)
    # graph.render("dot")

    # feature_arr = df_input.drop(['y'], axis=1).columns.values
    # feature_arr = feature_arr.reshape((1, FEATURE_COUNT))[0]
    # imp = imp.reshape((1, FEATURE_COUNT))
    # imp_df = pd.DataFrame(imp, columns=feature_arr)
    # print(imp_df)

    # plt.scatter([], 0, y_pred)
    # plt.show()


if __name__ == '__main__':
    main()
