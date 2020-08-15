import pandas as pd
from performance import twrr

data = pd.read_csv(
    'symbol/SPY.csv', header=0, index_col=0,
                names=['timestamp', 'open', 'low', 'high', 'close', 'volume'], parse_dates=['timestamp']
)
start_date = pd.to_datetime('2020-03-08')
data.reindex(pd.to_datetime(data.index, '%Y%m%d'))

data_cashflow = pd.DataFrame(data.loc[data.index >= start_date]['close'][::-1])
data_cashflow.loc[:,'cashflow'] = 0.*len(data_cashflow.index)
data_cashflow['cashflow'][0]=-data_cashflow['close'][0]
data_cashflow['cashflow'][-1]=data_cashflow['close'][-1]
data_cashflow.rename(columns={'close':'total'}, inplace=True)

result = twrr(data_cashflow)

for key, value in result.items():
    print(f'{key}:{value}')



