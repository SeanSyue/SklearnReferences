import pandas as pd
df = pd.DataFrame(columns=['col1', 'col2'])
df = df.append(pd.Series(['a', 'b'], index=['col1', 'col2']), ignore_index=True)
df = df.append(pd.Series(['d', 'e'], index=['col1', 'col2']), ignore_index=True)
print(df)
print(type(['a', 'b']))
print(type(['col1', 'col2']))
