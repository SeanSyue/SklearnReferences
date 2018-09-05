import pandas as pd
import numpy as np
import random


def one_hot_generator(length, digits=8, reshape=False):
    # How many digits in each output list
    a = []
    for n in range(0, length):
        n = random.randint(0, digits-1)
        a.append(n)
    output = np.eye(digits)[a]
    if reshape is True:
        output = np.reshape(output, (digits, length))
    return output


ROW = 20


s = one_hot_generator(4, ROW, reshape=True)
s.astype(int)

p1 = np.random.normal(0, 0.1, ROW)
p1 = np.reshape(p1, (ROW, 1))
p2 = np.random.normal(0, 0.1, ROW)
p2 = np.reshape(p2, (ROW, 1))
p = np.append(p1, p2, axis=1)


print(s.shape)
print(p1.shape)
print(p2.shape)
print(p.shape)

sp = np.append(s, p, axis=1)


df = pd.DataFrame(sp, columns=['col1', 'col2', 'col3', 'col4', 'col5', 'col6'])
print(df)

