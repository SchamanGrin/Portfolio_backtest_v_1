# data.py
import datetime
import os
import os.path
import time
from abc import ABCMeta, abstractmethod

import pandas as pd
import requests

from event import MarketEvent


class DataHandler(object):
    """
    DataHandler — абстрактный базовый класс, предоставляющий интерфейс для всех наследованных обработчиков (для живой торговли и работы с историческими данными)

Цель (выделенного) объекта DataHandler заключается в выводе сгенерированного набора баров (OLHCVI) для каждого запрощенного финансового инструмента.

Это нужно для получения понимания о том, как будет функционировать стратегия, при использовании реальных торговых данных. Таким образом реальная и историческая система во всем наборе инструментов бэктестинга рассматриваются одинаково.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def get_latest_bars(self, symbol, N=1):
        """
        Возвращает последние N баров из списка    latest_symbol или меньше, если столько баров еще недоступно.
        """
        raise NotImplementedError("Should implement get_latest_bars()")

    @abstractmethod
    def update_bars(self):
        """
        Накладывает последний бар на последнюю структуру инструмента для всех инструментов в списке.
        """
        raise NotImplementedError("Should implement update_bars()")


class HistoricCSVDataHandler(DataHandler):
    """
    HistoricCSVDataHandler создан для чтения CSV-файло с диска и создания интерфейса для получения «последнего» бара, как при реальной торговле.

    """

    def __init__(self, events, csv_dir, symbol_list, start_date):
        """
        Инициализирует обработчик исторических данных запросом местоположения CSV-файлов и списка инструментов.

        Предполагается, что все файлы имеют форму  'symbol.csv', где symbol — это строка списка.


        Параметры:
        events - очередь событий.
        csv_dir - Абсолютный путь к директории с CSV-файлами.
        symbol_list - Список строк инструментов.
        """
        self.events = events
        self.csv_dir = csv_dir
        self.symbol_list = symbol_list
        #этот костыль надо переделать при переписывании работы с датами. Здесь мы тип date переводим в текстовую строку. Чуть дальше будет обратное преобразование.
        #убрать, переписав всю логику работы с датами
        self.start_date = start_date.strftime('%Y-%m-%d')

        self.symbol_data = {}
        self.latest_symbol_data = {}
        self.continue_backtest = True

        self._open_convert_csv_files()

    def _open_convert_csv_files(self):
        """
       Открывает CSV-файлы из директории, конвертирует их в pandas DataFrames внутри словаря инструментов.

Для данного обработчика предположим, что данные берутся из фида DTN IQFeed, и работа идет с этим форматом.
        """
        comb_index = None
        for s in self.symbol_list:
            # Загрузка CSV-файла без заголовочной информации, индексированный по дате

            self.symbol_data[s] = pd.io.parsers.read_csv(
                os.path.join(self.csv_dir, '%s.csv' % s),
                header=0, index_col=0,
                names=['timestamp', 'open', 'low', 'high', 'close', 'volume']
                # names=['timestamp', 'open', 'low', 'high', 'close', 'volume', 'oi']
                # ???не понятно, что это за oi такое. На нем код ломается.????
            )

            # если даты нет в датасете, ищем первую дату после стартовой
            #Если дата позже первой  даты в датасете, даем исключение
            #datetime.datetime.strptime(self.symbol_data[s].index[0], '%Y-%m-%d')

            if self.start_date >= self.symbol_data[s].index[0]:
                raise Exception('Date not in dataset')
            else:
                while self.start_date not in self.symbol_data[s].index:
                    # Если даты нет в датасете, , берем следующий день
                    new_date = datetime.datetime.strptime(self.start_date, '%Y-%m-%d').date() + datetime.timedelta(1)
                    self.start_date = new_date.strftime('%Y-%m-%d')


            self.symbol_data[s] = self.symbol_data[s][:self.start_date].iloc[::-1]

            # Комбинируется индекс для «подкладывания» значений
            if comb_index is None:
                comb_index = self.symbol_data[s].index
            else:
                comb_index.union(self.symbol_data[s].index)

            # Set the latest symbol_data to None
            self.latest_symbol_data[s] = []

        # Reindex the dataframes
        for s in self.symbol_list:
            self.symbol_data[s] = self.symbol_data[s].reindex(index=comb_index, method='pad').iterrows()

    def _get_new_bar(self, symbol):
        """
        Возвращает последний бар из дата-фида в формате:
        (sybmbol, datetime, open, low, high, close, volume).
        """
        for b in self.symbol_data[symbol]:
            yield tuple([symbol, datetime.datetime.strptime(b[0], '%Y-%m-%d'),  # %H:%M:%S'),
                         b[1][0], b[1][1], b[1][2], b[1][3]])
            # b[1][0], b[1][1], b[1][2], b[1][3], b[1][4]])

    def get_latest_bars(self, symbol, N=1):
        """
        Возвращает N последних баров из списка latest_symbol, или N-k, если доступно меньше.

        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print('That symbol is not available in the historical data set.')
        else:
            return bars_list[-N:]

    def update_bars(self):
        """
        Отправляет последний бар в структуру данных инструментов для всех инструментов в списке.
"""
        for s in self.symbol_list:
            try:
                bar = next(self._get_new_bar(s))

            except StopIteration:
                self.continue_backtest = False
            else:
                if bar is not None:
                    self.latest_symbol_data[s].append(bar)
                    self.events.put(MarketEvent(bar[1]))


