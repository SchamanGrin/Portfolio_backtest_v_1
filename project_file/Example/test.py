import numpy as np
import pandas as pd
from pathlib import Path





def read_csv(path):
    df = pd.read_csv(path, header=0, index_col=0,
                     names=['date', 'total', 'cf', 'twrr'], parse_dates=['date'], sep=';',
                     dayfirst=True, decimal=',')
    return df


date = read_csv('testcase/twrr.csv')
cf = date[['total', 'cf']]
twrr = twrr(cf)
print()
