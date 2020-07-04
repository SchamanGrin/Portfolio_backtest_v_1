import configparser
import numpy as np

from pathlib import Path
from datetime import datetime



class backconfig(object):

    def to_date(self, x):
        #переводим текстовый формат даты из конфиг файла в формат numpy data
        return np.datetime64(x)


    def __init__(self):

        self.conf_path = Path(Path.cwd() / 'config' / 'conf.cf')

        self.apikey_path = Path(Path(__file__) / '..' / '..' / 'keys' / 'api_data.cf').resolve()

        self.conf_data = {
            'DEFAULT': {'tickets': 'SPY', 'start_date': '2010-01-01', 'initial_capital': '1000.0', 'add_funds': '100.0'},
            'data_type': {'list': {'tickets': str.split}, 'date': {'start_date': self.to_date},
                          'money': {'initial_capital': float, 'add_funds': float}}
        }

        self.key_data = {
            'DEFAULT':{'url':'https://www.alphavantage.co/query?', 'key':''},
            'data_type':{'alpha vantage':{'url':str, 'key':str}}
        }


        if not self.conf_path.exists():
            self._create_config_file(self.conf_data, self.conf_path)

        if not self.apikey_path.exists():
            self._create_config_file(self.key_data, self.apikey_path)

        #Если нет каталога /symbol создаем его
        if not Path(Path(__file__).parent / 'symbol').exists():
            Path.mkdir(Path(Path(__file__).parent / 'symbol'))

        self.values = self._filling_config_data(self.conf_data, self.conf_path)
        self.keys = self._filling_config_data(self.key_data, self.apikey_path)



    def _create_config_file(self, data, path):

        if not path.parent.exists():
            Path.mkdir(path.parent)
        config = configparser.ConfigParser()
        dt = data['data_type']
        for s in dt.keys():
            config.add_section(s)
            for k in dt[s].keys():
                config[s][k] = data['DEFAULT'][k]

        with open(path, 'w') as conf_file:
            config.write(conf_file)

    def _filling_config_data(self, data, path):

        config = configparser.ConfigParser()
        config.read(path)
        dt = data['data_type']

        for s in config.sections():
            for o, v in config.items(s):
                if not config[s][o]:
                    print(f'Заполните свойство {o} раздела [{s}] файла {path}')
                    exit()
        #На основании функций из словаря data_type
        conf_data = {s: {o: dt[s][o](v) for o, v in config.items(s)} for s in config.sections()}

        return conf_data

