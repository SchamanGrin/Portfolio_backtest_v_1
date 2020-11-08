import numpy as np
import pandas as pd
from pathlib import Path

from performance import create_return, xirr

def read_csv(path):
    df = pd.read_csv(path, header=0, index_col=0,
                names=['timestamp', 'close', 'cf', 'count', 'total', 'xirr'], parse_dates=['timestamp'], sep=';',
                    dayfirst = True)

    return df

#tc_path = 'testcase'
tc_list = Path(Path.cwd() / 'testcase').iterdir()

for name in tc_list:
    data = read_csv(name)
    cf_data = data[['total', 'cf']]
    cf_data.iloc[-1, 1] += data['total'][-1]
    cf_data = cf_data['cf'].reset_index()
    cf_data['timestamp'] = cf_data['timestamp'].astype('datetime64[ns]')
    arr_cf = cf_data[['timestamp', 'cf']].to_numpy()
    mwrr = xirr(arr_cf)
    #mwrr = create_return(cf_data, ['mwrr'])
    print(mwrr == data['xirr'[0]])

