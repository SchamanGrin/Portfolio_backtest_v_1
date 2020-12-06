import numpy as np
import pandas as pd
import xirr

from pathlib import Path

import performance as perfom

def read_csv(path):
    return pd.read_csv(path, header=0, index_col=0,
                     names=['timestamp', 'close', 'cf', 'count','cf and total end', 'total', 'result'], parse_dates=['timestamp'], sep=';',
                     dayfirst=True, decimal=',')

path = Path('/home/grin/PycharmProjects/Portfolio_backtest_v_1/project_file/auto_test/testcase') / 'negative_closely_1_cf.csv'

data = read_csv(path)

cf_data = pd.DataFrame(data[['cf and total end']])
cf_data = cf_data['cf and total end'].reset_index()
cf_data['timestamp'] = cf_data['timestamp'].astype('datetime64[ns]')
arr_cf = cf_data[['timestamp', 'cf and total end']].to_numpy()


res_xirr = perfom.xirr(arr_cf)

res_dict = {x:y for x,y in arr_cf}
bib_xirr = xirr(res_dict)
print(res_xirr == bib_xirr)
