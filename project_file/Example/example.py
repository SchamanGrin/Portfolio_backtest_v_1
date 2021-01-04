from performance import *
from pathlib import Path

def read_csv(path):
    return pd.read_csv(path, header=0, index_col=0,
                     names=['timestamp', 'close', 'cf', 'count','cf and total end', 'total', 'result'], parse_dates=['timestamp'], sep=';',
                     dayfirst=True, decimal=',')

path = Path('testcase') / 'positive_far_many_cf.csv'

data = read_csv(path)

cf_data = pd.DataFrame(data[['cf and total end']])
cf_data = cf_data['cf and total end'].reset_index()
cf_data['timestamp'] = cf_data['timestamp'].astype('datetime64[ns]')
arr_cf = cf_data[['timestamp', 'cf and total end']].to_numpy()

data_rename = data[['total', 'cf']].copy()
data_rename.rename(columns={'cf':'cashflow'}, inplace=True)
#dict_data = dict(zip(data.index,data['cf']))

revenue = create_return(data_rename, ['twrr'])['data']
cr = create_sharpe_ratio(revenue['twrr'])
cd = create_drawdowns(revenue['twrr'])
print(cr)
print(cd)
