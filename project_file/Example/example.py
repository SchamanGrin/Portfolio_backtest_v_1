import time
import pandas as pd
import numpy as np
from performance import twrr, xirr, xirr_1



data = pd.read_csv(
    'symbol/SPY.csv', header=0, index_col=0,
                names=['timestamp', 'open', 'low', 'high', 'close', 'volume'], parse_dates=['timestamp']
)
start_date = pd.to_datetime('2010-01-01')
data.reindex(pd.to_datetime(data.index, '%Y%m%d'))

data_cashflow = pd.DataFrame(data.loc[data.index >= start_date]['close'][::-1])
data_cashflow.loc[:,'cashflow'] = 0.*len(data_cashflow.index)
data_cashflow['cashflow'][0]=-data_cashflow['close'][0]
data_cashflow.rename(columns={'close':'total'}, inplace=True)
data_cashflow.loc[data_cashflow.index[-1], 'cashflow'] = data_cashflow['total'][-1]


#xirr = xirr_1(data_cashflow['cashflow'].loc[np.abs(data_cashflow['cashflow']) > 1E-10])*100
'''
def func_dfxirr(dfxirr, x, i):
    dfxirr.loc[i] += x
    return dfxirr[np.abs(dfxirr) > 1E-10]

time_0_0 = time.time()
df_xirr_copy = data_cashflow['cashflow'].copy()
data_xirr_0 = [xirr_1(func_dfxirr(df_xirr_copy[:i].copy(), data_cashflow['total'][i], i)) \
               for i in data_cashflow.index[1:]]
print(f'{time.time() - time_0_0:.2f} сек.')


#print(xirr)
#twrr_total, twrr, twrr_data = twrr(data_cashflow)

time_1 = time.time()
data_xirr_1 = []
df_xirr_copy = data_cashflow['cashflow'].copy()
df_x0 = df_xirr_copy[:1]
for i in data_cashflow.index[1:]:
    t = df_xirr_copy[df_xirr_copy.index == i]
    df_x0.append(t)
    df_xirr = df_x0.copy()
    df_xirr[-1] += data_cashflow['total'][i]
    data_xirr_1 += [xirr_1(df_xirr.loc[np.abs(df_xirr) > 1E-10])]

print(f'{time.time() - time_1:.2f} сек.')

print(sum(data_xirr_0[i] - data_xirr_1[i] for i in range(len(data_xirr_0))))
'''

df_xirr_func = data_cashflow['cashflow'].copy()




