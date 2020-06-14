import time
import requests


class AlphaVantage(object):
    """
    Создает датафрейм с данными по символу с нужным таймфремом (минуты, часы, дни, месяцы) и сохраняет его в csv

    """

    def __init__(self, conf, symbol_list, timeframe='TIME_SERIES_DAILY'):


        self.symbol_list = symbol_list
        self.timeframe = timeframe

        self.api_name = 'Alpha Vantage'
        self.key = conf.keys[self.api_name]['key']
        self.url = conf.keys[self.api_name]['url']


    def take_csv(self, csv_dir='symbol/', outputsize='compact'):

        for symbol in self.symbol_list:
            data = {
                'function': self.timeframe,
                'symbol': symbol,
                'outputsize': outputsize,
                'apikey': self.key,
                'datatype': 'csv'
            }
            csv_data = requests.get(self.url, params=data, verify=False)
            path = csv_dir + symbol + '.csv'
            with open(path, 'wb') as output:
                output.write(csv_data.content)

            time.sleep(10)
