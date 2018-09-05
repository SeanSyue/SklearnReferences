import pandas as pd
import time
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from MachineLearning.project.project_main.proj_tree import data_splitter


FILE = 'C:/bank/data_set/benchmark/bank_benchmark_mlp.csv'
TIME_PATH = 'C:/bank/mlp_result/time/'
CV_PATH = 'C:/bank/mlp_result/cv/'
CLS_PATH = 'C:/bank/mlp_result/cls/'


def mlp_analysis(X_train, y_train, X_test, lyr_size, act_fn):
    tree_clf = MLPClassifier(hidden_layer_sizes=(lyr_size,), activation=act_fn, max_iter=5000)
    tree_result = tree_clf.fit(X_train, y_train)
    tree_pred = tree_clf.predict(X_test)
    return tree_clf, tree_result, tree_pred


def fn_str_switch(fn):
    options = {'logistic': 'lo',
               'tanh': 'ta',
               'relu': 're',
               'identity': 'id',
               }
    return options[fn]


def main():
    # -----------------------------------------Preprocessing---------------------------------------
    X, y, X_train, X_test, y_train, y_test = data_splitter(pd.read_csv(FILE))

    # -----------------------------------------Train steps-----------------------------------------
    for act_fn in ['identity', 'logistic', 'tanh', 'relu']:
        fn_str = fn_str_switch(act_fn)

        for lyr_size in range(1, 151):
            print(f"{fn_str}_{lyr_size}...")
            mlp_start = time.time()
            tree_clf, _, y_pred = mlp_analysis(X_train, y_train, X_test, lyr_size=lyr_size, act_fn=act_fn)
            mlp_end = time.time()

            with open(f"{TIME_PATH}t_{fn_str}.txt", 'a') as t_report:
                print(f"{fn_str}_{lyr_size}_t,", mlp_end - mlp_start, file=t_report)
            with open(f"{CV_PATH}cv_{fn_str}.txt", 'a') as cv_report:
                print(f"{fn_str}_{lyr_size}_cv,", cross_val_score(tree_clf, X, y, cv=5), file=cv_report)
            with open(f"{CLS_PATH}cls_{fn_str}.txt", 'a') as cls_report:
                print(f"{fn_str}_{lyr_size}_cls\n", classification_report(y_test, y_pred), file=cls_report)


if __name__ == '__main__':
    main()

