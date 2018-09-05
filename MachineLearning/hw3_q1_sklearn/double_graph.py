import numpy as np
import random
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt


def one_hot_generator(length):
    digits = 8  # How many digits in each output list
    a = []
    for i in range(0, length):
        i = random.randint(0, digits-1)
        a.append(i)
    output = np.eye(digits)[a]
    return output


train = one_hot_generator(100)
test = one_hot_generator(30)

iter_list_raw = []
test_error_list_raw = []
train_error_list_raw = []
for i in range(50, 2500, 50):
    print("==================== max_iter: {} ====================".format(i))
    # Training with a multi-layer perceptron with one hidden layer.
    mlp = MLPClassifier(hidden_layer_sizes=1, max_iter=i, tol=1e-8)
    mlp_result = mlp.fit(train, train)
    # Prediction
    train_out = mlp.predict(train)
    test_out = mlp.predict(test)
    train_error = mean_squared_error(train, train_out)
    test_error = mean_squared_error(test, test_out)

    train_error_list_raw.append(train_error)
    test_error_list_raw.append(test_error)
    iter_list_raw.append(mlp.n_iter_)

    print("train:\n", train[:8])
    print("train out:\n", train_out[:8])
    print("test:\n", test[:8])
    print("test out:\n", test_out[:8])

    print("train_error:", train_error)
    print("test_error:", test_error)
    print("n_iter_:", mlp.n_iter_)
    print("weights:", mlp.coefs_[0][0])

iter_list, test_list = zip(*sorted(zip(iter_list_raw, test_error_list_raw)))
iter_list, train_list = zip(*sorted(zip(iter_list_raw, train_error_list_raw)))
print("iter_list:\n", iter_list)

plt.style.use('ggplot')
fig = plt.figure(figsize=(16, 8))

plt.plot(iter_list, test_list, '-')
plt.bar(iter_list, train_list, align='center')
plt.show()
