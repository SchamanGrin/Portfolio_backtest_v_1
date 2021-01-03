from performance import create_return, xirr
import pandas as pd
import unittest
from pathlib import Path

def read_csv(path):
    df = pd.read_csv(path, header=0, index_col=0,
                     names=['timestamp', 'close', 'cf', 'count','cf and total end', 'total', 'result'], parse_dates=['timestamp'], sep=';',
                     dayfirst=True, decimal=',')

    return df



class test_return(unittest.TestCase):

    tc_names = Path(Path.cwd() / 'testcase').iterdir()
    test_data = {}

    for name in tc_names:
        tc_path = Path(name)
        tc_data = read_csv(tc_path)
        test_data[name.stem] = {'data':tc_data[['total', 'cf']], 'result':tc_data['result'][0]}


    def test_xirr_diferent_cf(self):
        dt = self.test_data['diferent_cf']
        self.assertEqual(round(create_return(dt['data'], ['mwrr'])['mwrr'], 4), round(dt['result'],4))

    def test_xirr_positive_far_many_cf(self):
        dt = self.test_data['positive_far_many_cf']
        self.assertEqual(round(create_return(dt['data'], ['mwrr'])['mwrr'], 4), round(dt['result'],4))

    def test_xirr_negative_far_many_cf(self):
        dt = self.test_data['negative_far_many_cf']
        self.assertEqual(round(create_return(dt['data'], ['mwrr'])['mwrr'], 4), round(dt['result'],4))

    def test_xirr_negative_closely_many_cf(self):
        dt = self.test_data['negative_closely_many_cf']
        self.assertEqual(round(create_return(dt['data'], ['mwrr'])['mwrr'], 4), round(dt['result'],4))

    def test_xirr_positive_closely_1_cf(self):
        dt = self.test_data['positive_closely_1_cf']
        self.assertEqual(round(create_return(dt['data'], ['mwrr'])['mwrr'], 4), round(dt['result'],4))

    def test_xirr_negative_closely_1_cf(self):
        dt = self.test_data['negative_closely_1_cf']
        self.assertEqual(round(create_return(dt['data'], ['mwrr'])['mwrr'], 4), round(dt['result'],4))

    def test_xirr_negative_far_1_cf(self):
        dt = self.test_data['negative_far_1_cf']
        self.assertEqual(round(create_return(dt['data'], ['mwrr'])['mwrr'], 4), round(dt['result'],4))

    def test_xirr_positive_far_1_cf(self):
        dt = self.test_data['positive_far_1_cf']
        self.assertEqual(round(create_return(dt['data'], ['mwrr'])['mwrr'], 4), round(dt['result'],4))

    def negative_close_many_twrr(self):
        dt = self.test_data['negative_close_many_twrr']
        self.assertEqual(round(create_return(dt['data'], ['twrr'])['twrr'], 4), round(dt['result'],4))






if __name__ == '__main__':
    unittest.main()

