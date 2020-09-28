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

#twrr = twrr(data_cashflow)

data_cashflow.loc[data_cashflow.index[-1],'cashflow']=data_cashflow['total'][-1]
xirr = xirr_1(data_cashflow[np.abs(data_cashflow['cashflow']) > 1e-10])

data_cashflow['YEAR'] = pd.to_datetime(data_cashflow.index, format = '%Y')
dfy = data_cashflow['YEAR'].unique()
# dfs = dict(tuple(data_cashflow.groupby(data_cashflow['YEAR'].dt.year)))
xirr_year = []
for year in dfy:
    data_year = data_cashflow[data_cashflow.index.year == year]

    xirr_year += [data_cashflow[data_cashflow.index.year == year]]

print(0)



