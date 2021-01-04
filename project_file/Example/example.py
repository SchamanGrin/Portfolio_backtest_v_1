from performance import *
from pathlib import Path

def read_csv(path):
    return pd.read_csv(path, header=0, index_col=0,
                     names=['timestamp', 'total', 'cf', 'twrr'], parse_dates=['timestamp'], sep=';',
                     dayfirst=True, decimal=',')

path = Path('testcase') / 'twrr_far_many_cf.csv'

data = read_csv(path)

'''cf_data = pd.DataFrame(data[['cf and total end']])
cf_data = cf_data['cf and total end'].reset_index()
cf_data['timestamp'] = cf_data['timestamp'].astype('datetime64[ns]')
arr_cf = cf_data[['timestamp', 'cf and total end']].to_numpy()'''

data_rename = data[['total', 'cf']].copy()
data_rename.rename(columns={'cf':'cashflow'}, inplace=True)
#dict_data = dict(zip(data.index,data['cf']))

revenue = create_return(data_rename, ['twrr'])
print(revenue['twrr'])
