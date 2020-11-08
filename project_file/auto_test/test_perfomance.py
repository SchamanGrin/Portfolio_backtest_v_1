from performance import *
import pandas as pd
import unittest
from pathlib import Path

def read_csv(path):
    df = pd.read_csv(path, header=0, index_col=0,
                     names=['timestamp', 'close', 'cf', 'count', 'total', 'xirr'], parse_dates=['timestamp'], sep=';',
                     dayfirst=True, decimal=',')

    return df

def prepare_data(data):
    cf_data = pd.DataFrame(data[['total', 'cf']])
    cf_data = cf_data['cf'].reset_index()
    cf_data['timestamp'] = cf_data['timestamp'].astype('datetime64[ns]')
    arr_cf = cf_data[['timestamp', 'cf']].to_numpy()
    return arr_cf


class test_return(unittest.TestCase):

    tc_names = Path(Path.cwd() / 'testcase').iterdir()
    data4xirr = {}
    for name in tc_names:
        tc_path = Path(name)
        tc_data = read_csv(tc_path)
        data4xirr[name.stem] = {'data':prepare_data(tc_data), 'answer':tc_data['xirr'][0]}


    def test_xirr_diferent_cf(self):
        dt = self.data4xirr['diferent_cf']
        self.assertEqual(round(xirr(dt['data']), 4), round(dt['answer'],4))

    def test_xirr_positive_far_many_cf(self):
        dt = self.data4xirr['positive_far_many_cf']
        self.assertEqual(round(xirr(dt['data']), 4), round(dt['answer'],4))

    def test_xirr_negative_far_many_cf(self):
        dt = self.data4xirr['negative_far_many_cf']
        self.assertEqual(round(xirr(dt['data']), 4), round(dt['answer'],4))

    def test_xirr_negative_closely_many_cf(self):
        dt = self.data4xirr['negative_closely_many_cf']
        self.assertEqual(round(xirr(dt['data']), 4), round(dt['answer'],4))

    def test_xirr_positive_closely_1_cf(self):
        dt = self.data4xirr['positive_closely_1_cf']
        self.assertEqual(round(xirr(dt['data']), 4), round(dt['answer'],4))

    def test_xirr_negative_closely_1_cf(self):
        dt = self.data4xirr['negative_closely_1_cf']
        self.assertEqual(round(xirr(dt['data']), 4), round(dt['answer'],4))

    def test_xirr_negative_far_1_cf(self):
        dt = self.data4xirr['negative_far_1_cf']
        self.assertEqual(round(xirr(dt['data']), 4), round(dt['answer'],4))

    def test_xirr_positive_far_1_cf(self):
        dt = self.data4xirr['positive_far_1_cf']
        self.assertEqual(round(xirr(dt['data']), 4), round(dt['answer'],4))







if __name__ == '__main__':
    unittest.main()

