import pandas as pd

dt = pd.DataFrame({'col1': [5,6], 'col2':[3,4]})
print(dt)

df = pd.DataFrame({'col1': [7], 'col2':[8]})
print(df)

dt.append(df, ignore_index=True)
print(dt)



