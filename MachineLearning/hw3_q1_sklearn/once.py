import numpy as np
import random
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPClassifier


def one_hot_generator(length):
    a = []
    for i in range(0, length):
        i = random.randint(0, 7)
        a.append(i)
    output = np.eye(8)[a]
    return output


n_train = 150
train = one_hot_generator(n_train)
test = one_hot_generator(30)


# --------------------------------------Multi-layer perceptron analysis----------------------------
# Training with a multi-layer perceptron with one hidden layer.
iteration = 15000
mlp = MLPClassifier(hidden_layer_sizes=2, max_iter=iteration)
mlp_result = mlp.fit(train, train)
# Prediction
train_out = mlp.predict(train)
test_out = mlp.predict(test)
print("train:\n", train[n_train-10:])
print("train out:\n", train_out[n_train-10:])
print("test:\n", test[:8])
print("test out:\n", test_out[:8])

error = mean_squared_error(test, test_out)
print("mean_squared_error:", error)

print("n_iter_:", mlp.n_iter_)
print("weight:\n", mlp.coefs_)
print("weight[0]:\n", mlp.coefs_[0])
print("weights[0][0]\n", mlp.coefs_[0][0])
print("sum of weights:", sum(mlp.coefs_[0][0]))


