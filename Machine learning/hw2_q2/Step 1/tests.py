import pandas as pd

df = pd.read_csv("C:/bank-additional-full.csv", sep=";")
df2 = df[:30]
print(df2)
print(type(df2))
df3 = df2['"job"']

###Simple test###
import pandas as pd
df4 = pd.DataFrame({'cid':['c01', 'c02', 'c03'], 'time': [43, 543, 34]})
print(df4)
print(df4['time'])
print(type(df4))

### Iris ###
import pandas as pd
csv = pd.read_csv("C:/Users/Sean/Downloads/iris.csv", encoding="big5")
print(csv.head())

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='entropy', max_depth=5)
tree_result = tree.fit(csv[['花萼長度', '花萼寬度', '花瓣長度', '花瓣寬度']], csv[['屬種']])
print(tree_result)
