import pandas as pd
from sklearn.tree import DecisionTreeClassifier

##Read csv file retreived from "https://www.kaggle.com/c/spooky-author-identification"
csv = pd.read_csv("C:/Users/Sean/Downloads/sample_submission/sample_submission.csv")

##Train the decision tree
tree = DecisionTreeClassifier(criterion='entropy', max_depth=5)
tree_result = tree.fit(csv[['MWS', 'EAP', 'HPL']], csv[['id']])
print(tree_result)

##Compare feature importances between each colomn
tree.feature_importances_.tolist()
df = pd.DataFrame({'items': ['MWS', 'EAP', 'HPL'], 'feature_importances_': tree.feature_importances_.tolist()})
df = df.sort_values(by=['feature_importances_'], ascending=True).reset_index(drop=True)
print(df)
