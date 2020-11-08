from performance import *
import os
import pandas as pd

def read_csv(path):
    df = pd.read_csv(path, header=0, index_col=0,
                names=['timestamp', 'close', 'cf', 'count', 'total'], parse_dates=['timestamp'], sep=';',
                    dayfirst = True)

    return df


class test_perfomance():
    tc_path = '/testcase/'
    tc_list = os.listdir(tc_path)

    def test_xirr(self):
        for name in self.tc_list:
            cf_data = read_csv(name)


