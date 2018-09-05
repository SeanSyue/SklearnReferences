from sklearn.neural_network import MLPClassifier
X = [[0., 0.], [1., 1.]]
y = [0, 1]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)

print(clf.fit(X, y))  # Training
print(clf.predict([[2., 2.], [-1., -2.]]))  # Testing

# MLP can fit a non-linear model to the training data.
# clf.coefs_ contains the weight matrices that constitute the model parameters:
print([coef.shape for coef in clf.coefs_])


