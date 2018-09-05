import numpy as np
import random
sample = np.eye(8)[random.sample(range(8), 8)]
print("sample:\n", sample)

_spl_param = 6  # Range: 1~7
train = sample[:_spl_param]
test = sample[_spl_param:]
print("train:\n", train)
print("test:\n", test)

