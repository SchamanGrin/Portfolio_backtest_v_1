import time

import pandas as pd
from performance import twrr, twrr_1



data = pd.read_csv(
    'symbol/SPY.csv', header=0, index_col=0,
                names=['timestamp', 'open', 'low', 'high', 'close', 'volume'], parse_dates=['timestamp']
)
start_date = pd.to_datetime('2010-01-01')
data.reindex(pd.to_datetime(data.index, '%Y%m%d'))

data_cashflow = pd.DataFrame(data.loc[data.index >= start_date]['close'][::-1])
data_cashflow.loc[:,'cashflow'] = 0.*len(data_cashflow.index)
data_cashflow['cashflow'][0]=-data_cashflow['close'][0]
data_cashflow['cashflow'][-1]=data_cashflow['close'][-1]
data_cashflow.rename(columns={'close':'total'}, inplace=True)

#data_cashflow['year'] = data_cashflow.index.year

start_twrr_1 = time.time()
result = twrr_1(data_cashflow)
print(f'Время функции twrr_1: {time.time() - start_twrr_1:.2f}с')


start_twrr = time.time()
result = twrr(data_cashflow)
print(f'Время функции twrr: {time.time() - start_twrr:.2f}с')


