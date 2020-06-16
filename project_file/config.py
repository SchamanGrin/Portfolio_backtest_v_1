import configparser
import datetime
from pathlib import Path


class backconfig(object):

    def __init__(self):

        self.conf_path = Path(Path.cwd() / 'config' / 'conf.cf')

        self.apikey_path = Path(Path(__file__) / '..' / '..' / 'keys' / 'api_data.cf').resolve()

        if not self.conf_path.exists():
            self._create_config_file()

        if not self.apikey_path.exists():
            self._create_key_file()

        #Если нет каталога /symbol создаем его
        if not Path(Path(__file__).parent / 'symbol').exists():
            Path.mkdir(Path(Path(__file__).parent / 'symbol'))


        self.values = self._filling_configs()
        self.keys = self._filling_keys()

    def _create_config_file(self):
        """
        Функция создает конфигурационный файл / каталог, где он должен лежать, заполненный по умолчанию,


        """

        if not self.conf_path.parent.exists():
            Path.mkdir(self.conf_path.parent)

        str_conf = """[list]
tickets = SPY
[date]
start_date = 01.01.2010
[money]
initial_capital =  1000
add_funds = 100"""
        f = Path(self.conf_path)
        try:
            Path(f).write_text(str_conf)
        except FileNotFoundError:
            print('Каталог или файл не найден')
            exit(1)

    def _filling_configs(self):

        """
        Функция создает справочник из конфигурационного файла, приводя к нужным типам данных, исходя из секции
        !!!! проверить на велосипеды
         возвращает словарь с приведенными типами данных
        """

        def to_date(x):
            return datetime.datetime.strptime(x, '%d.%m.%Y')

        def to_list(x):
            return x.split()

        conf = configparser.RawConfigParser()
        conf.read(self.conf_path)
        configs = {s: {o: v for o, v in conf.items(s)} for s in conf.sections()}
        func = {'list': to_list, 'date': to_date, 'money': float}

        for s in configs:
            for o in configs[s].keys():
                if configs[s][o]:
                    configs[s][o] = func[s](configs[s][o])
                else:
                    print(f'Заполните параметр {o} в файле {self.conf_path}')
                    exit(1)

        return configs

    def _create_key_file(self):
        """
        Функция создает файл с ключами, вне каталога проекта, в файл записываю ключи и URL используемых по API источников данных
        :return:
        """
        if not self.apikey_path.parent.exists():
            Path.mkdir(self.apikey_path.parent)

        str_key = """[Alpha Vantage]
    url = "https://www.alphavantage.co/query?"
    key ="""

        f = Path(self.apikey_path)
        try:
            Path(f).write_text(str_key)
        except FileNotFoundError:
            print('Каталог или файл не найден')
            exit(1)

    def _filling_keys(self):
        """
        Функция формирует словарь с ключами от API
        :return: Словарь с ключами
        """
        conf = configparser.RawConfigParser()
        conf.read(self.apikey_path)
        keys = {s: {o: v for o, v in conf.items(s)} for s in conf.sections()}
        for s in keys:
            for o in keys[s].keys():
                if not keys[s][o]:
                    print(f'Заполните параметр {o} в файле {self.apikey_path}')
                    exit(1)

        return keys
