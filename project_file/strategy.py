# strategy.py

import datetime
import numpy as np
import pandas as pd
from queue import Queue

from abc import ABCMeta, abstractmethod

from event import SignalEvent

class Strategy(object):
    """
    Strategy — абстрактный базовый класс, предоставляющий интерфейс для подлежащих (наследованных) объектов для обработки стратегии.


    Цель выделенного объекта Strategy заключается в генерировании сигнальных объектов для конкретных инструментов на основе входящих баров (OLHCVI), сгенерированных объектом DataHandler.

    Эту конфигурацию можно использовать как для работы с историческими данными, так и для работы на реальном рынке — объект Strategy не зависит от от источника данных, он получает бары из очереди.

    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def calculate_signals(self):
        """
        Предоставляет механизмы для вычисления списка сигналов.

        """
        raise NotImplementedError("Should implement calculate_signals()")

class BuyAndHoldStrategy(Strategy):
    """
    Крайне простая стратегия, которая входит в длинную позициию при полуении бара и никогда из нее не выходит.

    Используется в качестве механизма тестирования класса Strategy и бенчмарка для сравнения разных стратегий.
    """

    def __init__(self, bars, events):
        """
        Инициализирует стратегию buy and hold.

        Параметры:
        bars - Объект DataHandler, который предоставляет информацию о барах
        events - Объект очереди событий.
        """
        self.bars = bars
        self.symbol_list = self.bars.symbol_list
        self.events = events

        # Когда получен сигнал на покупку и удержание акции, устанавливается в True
        self.bought = self._calculate_initial_bought()

    def _calculate_initial_bought(self):
        """
        Добавляются ключи в словарь bought и устанавливаются в False.
        """
        bought = {}
        for s in self.symbol_list:
            bought[s] = False
        return bought

    def calculate_signals(self, event):
        """
       Для "Buy and Hold" генерируем один сигнал на инструмент. Это значит, что мы только открываем длинные позиции с момента инициализации стратегии.

        Параметры:
        event - Объект MarketEvent.
        """
        if event.type == 'MARKET':
            for s in self.symbol_list:
                bars = self.bars.get_latest_bars(s, N=1)
                if bars is not None and bars != []:
                    if self.bought[s] == False:
                    # (Symbol, Datetime, Type = LONG, SHORT or EXIT)
                        signal = SignalEvent(bars[0][0], bars[0][1], 'LONG')
                        self.events.put(signal)
                        #self.bought[s] = True


