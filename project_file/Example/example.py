import pandas as pd

dt = pd.DataFrame({'col1': [5,6], 'col2':[3,4]})
print(dt)

t = [(x, dt.loc[x]['col1']) for x in dt.index]
print(t)
