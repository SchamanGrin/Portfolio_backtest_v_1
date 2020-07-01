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

    def __init__(self, bars, events, port):
        """
        Инициализирует стратегию buy and hold.
        Параметры:
        bars - Объект DataHandler, который предоставляет информацию о барах
        events - Объект очереди событий.
        """
        self.bars = bars
        self.symbol_list = self.bars.symbol_list
        self.events = events
        self.portfolio = port

        # Когда получен сигнал на покупку и удержание акции, устанавливается в True
        #self.bought = self._calculate_bought(event)


    def _calculate_bought(self):
        """
        Функия возвращает словарь, разрещающий / запрещающий покупку той или иной бумаги.
        Бумага покупается один раз в зависимости от того, хватит ли на нее денег  и покупалась ли она ранее. Если не покупалась
        и денег хватает, то возвращает True иначе Else
        :return: словарь с бумагами и значеним False, если не покупать и True, если покупать
        """

        bought = {}
        free_cash = self.portfolio.current_holdings['cash']
        for s in self.symbol_list:
            bought[s] = False
            bars = self.bars.get_latest_bars(s, N=1)
            cost = bars[0][4]*self.portfolio.buy_quantity
            if self.portfolio.current_positions[s] == 0 and free_cash > cost:
                bought[s] = True
                free_cash -= cost

        return bought



    def calculate_signals(self, event):
        """
       Для "Buy and Hold" генерируем один сигнал на инструмент. Это значит, что мы только открываем длинные позиции .
        Параметры:
        event - Объект MarketEvent.
        """
        if event.type == 'MARKET':
            bought = self._calculate_bought()
            for s in self.symbol_list:
                bars = self.bars.get_latest_bars(s, N=1)
                if bars is not None and bars != []:
                    if bought[s]:
                        # (Symbol, Datetime, Type = LONG, SHORT or EXIT)
                        signal = SignalEvent(bars[0][0], bars[0][1], 'LONG')
                        self.events.put(signal)