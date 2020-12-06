import numpy as np
import pandas as pd
import time
import xirr
print(xirr.__version__)

from pathlib import Path
# from datetime import date

import performance as perfom

def read_csv(path):
    return pd.read_csv(path, header=0, index_col=0,
                     names=['timestamp', 'close', 'cf', 'count','cf and total end', 'total', 'result'], parse_dates=['timestamp'], sep=';',
                     dayfirst=True, decimal=',')

path = Path('testcase') / 'negative_closely_1_cf.csv'

data = read_csv(path)

cf_data = pd.DataFrame(data[['cf and total end']])
cf_data = cf_data['cf and total end'].reset_index()
cf_data['timestamp'] = cf_data['timestamp'].astype('datetime64[ns]')
arr_cf = cf_data[['timestamp', 'cf and total end']].to_numpy()

t1 = time.time()
res_xirr = perfom.create_return(data[['total','cf']], ['mwrr'])
print(f'perfom:{time.time() - t1}')


t2 = time.time()
res_dict = {x: y for x, y in arr_cf}
bib_xirr = xirr.xirr(res_dict)
print(f'math:{time.time() - t2}')
print(res_xirr == bib_xirr)
