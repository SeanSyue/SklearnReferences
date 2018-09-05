from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd

#df2 = pd.read_csv("C:/Users/Sean/Downloads/bank/bank.csv", enconding="big5", sep=";")
df2 = pd.read_csv("C:/Users/Sean/Downloads/bank/bank.csv", sep=";")
df3 = df2[["age", "job", "marital", "education", "default", "balance", "housing", "loan","contact", "day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "y"]]

##
label_encoder = preprocessing.LabelEncoder()
encoded_job = label_encoder.fit_transform(df3["job"])
df3["job"] = encoded_job
encoded_marital = label_encoder.fit_transform(df3["marital"])
df3["marital"] = encoded_marital
encoded_education = label_encoder.fit_transform(df3["education"])
df3["education"] = encoded_education
encoded_default = label_encoder.fit_transform(df3["default"])
df3["default"] = encoded_default
encoded_housing = label_encoder.fit_transform(df3["housing"])
df3["housing"] = encoded_housing
encoded_housing = label_encoder.fit_transform(df3["housing"])
df3["housing"] = encoded_housing
encoded_loan = label_encoder.fit_transform(df3["loan"])
df3["loan"] = encoded_loan
encoded_y = label_encoder.fit_transform(df3["y"])
df3["y"] = encoded_y

##
x = df3.ix[:,0:7]
y = df3.ix[:,-1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=20171113)

##
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

sc.fit(x_train)
x_train_nor = sc.transform(x_train)
x_test_nor = sc.transfrom(x_test)
##
tree = DecisionTreeClassifier(criterion='gini', max_depth=5)
tree_clf = tree.fit(x_train_nor, y_train)

#
y_test_predicted = tree_clf.predict(x_test_nor)
print("y_test_predicted")

print("y_test.head()")

accuracy = metrics.accuracy_score(y_test, y_test_predicted)
print('ACCURACY', accuracy)