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


xirr = xirr_1(data_cashflow['cashflow'].loc[np.abs(data_cashflow['cashflow']) > 1E-10])*100

data_xirr = []
for i in data_cashflow.index[1:]:
    df_xirr = data_cashflow['cashflow'].copy()
    df_xirr.loc[i] += data_cashflow['total'][i]
    data_xirr += [xirr_1(df_xirr.loc[np.abs(df_xirr) > 1E-10])]

print(xirr)
#twrr_total, twrr, twrr_data = twrr(data_cashflow)





print(0)



