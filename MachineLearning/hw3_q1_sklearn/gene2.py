import numpy as np
import random
sample = np.eye(8)[random.sample(range(8), 8)]
print("sample:\n", sample)

_spl_param = 6  # Range: 1~7

a = []
for i in range(0, 100):
    i = random.randint(0, _spl_param-1)
    a.append(i)
print("a:", a)
print("len of a", len(a))

train = sample[a]
test = sample[_spl_param:]
print("train:\n", train)
print("test:\n", test)
print("len of train:", len(train))
print("len of test:", len(test))
