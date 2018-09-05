# https://github.com/scikit-learn/scikit-learn/issues/8399
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

rng = np.random.RandomState(0)
X = rng.normal(size=(50, 2))
y = np.zeros(X.shape[0], dtype=np.int)
y[X[:, 1] > 2] = 1

tree = DecisionTreeClassifier().fit(X, y)

# create a grid for plotting decision functions...
x_lin = np.linspace(X[:, 0].min() - .5, X[:, 0].max() + .5, 1000)
y_lin = np.linspace(X[:, 1].min() - .5, X[:, 1].max() + .5, 1000)
x_grid, y_grid = np.meshgrid(x_lin, y_lin)
X_grid = np.c_[x_grid.ravel(), y_grid.ravel()]


fig, axes = plt.subplots(1, 2)

axes[0].contourf(x_grid, y_grid, tree.predict_proba(X_grid)[:, 0].reshape(x_grid.shape), alpha=.3)
axes[0].scatter(X[:, 0], X[:, 1], c=plt.cm.Vega10(y))

tree2 = DecisionTreeClassifier(min_samples_leaf=2).fit(X, y)
axes[1].contourf(x_grid, y_grid, tree2.predict_proba(X_grid)[:, 0].reshape(x_grid.shape), alpha=.3)
axes[1].scatter(X[:, 0], X[:, 1], c=plt.cm.Vega10(y))
# axes[1].scatter(X[:, 0], X[:, 1])


plt.show()
