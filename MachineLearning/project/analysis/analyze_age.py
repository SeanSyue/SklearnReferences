import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 80)

reader = pd.read_csv('C:/bank/data_set/origin_data_sets/bank-additional-full.csv', sep=";")

counts = reader['age'].value_counts()
counts = counts.sort_index()
print(counts)
print(type(counts))
print(counts.shape)

plt.style.use('ggplot')
fig = plt.figure(figsize=(16, 8))
plt.bar(counts.index, counts)
plt.show()
