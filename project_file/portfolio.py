# portfolio.py
import numpy as np
import pandas as pd

from abc import ABCMeta, abstractmethod

from event import FillEvent, OrderEvent
from performance import create_sharpe_ratio, create_drawdowns, xirr, twrr, xirr_1

class Portfolio(object):
    """
    Класс Portfolio обрабатывает позиции и рыночную стоимость всех инструментов на основе баров: секунда, минута, 5 минут, 30 мин, 60 минут или день.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def update_signal(self, event):
        """
        Использует SignalEvent для генерации новых ордеров в соответствие с логикой портфолио.

        """
        raise NotImplementedError("Should implement update_signal()")

    @abstractmethod
    def update_fill(self, event):
        """
        Обновляет текущие позиции и зарезервированные средства в портфолио на основе      FillEvent.

        """
        raise NotImplementedError("Should implement update_fill()")

    @abstractmethod
    def update_market(self, event):
        """
        Обновляет текущие позиции и зарезервированные средства в портфолио на основе      FillEvent.

        """
        raise NotImplementedError("Should implement update_market()")


class NaivePortfolio(Portfolio):
    """

Объект NaivePortfolio создан для слепой (т.е. без всякого риск-менеджмента)  отправки приказов на покупку/продажу установленного количество акций, в брокерскую систему. Используется для тестирования простых стратегий вроде BuyAndHoldStrategy.
    """

    def __init__(self, bars, events, start_date, initial_capital=100000.0, buy_quantity=100.0):
        """
        Инициализирует портфолио на основе информации из баров и очереди событий. Также включает дату и время начала и размер начального капитала (в долларах, если не указана другая валюта).

        Parameters:
        bars - The DataHandler object with current market data.
        events - The Event Queue object.
        start_date - The start date (bar) of the portfolio.
        initial_capital - The starting capital in USD.
        """
        self.bars = bars
        self.events = events
        self.symbol_list = self.bars.symbol_list
        self.start_date = start_date
        self.initial_capital = initial_capital
        self.buy_quantity = buy_quantity

        self.all_positions = self.construct_all_positions()
        self.current_positions = dict((k, v) for k, v in [(s, 0) for s in self.symbol_list])

        self.all_holdings = self.construct_all_holdings()
        self.current_holdings = self.construct_current_holdings()

    def construct_all_positions(self):
        """
     Конструирует список позиций, используя start_date для определения момента, с которой должен начинаться временной индекс.
        """
        d = dict((k, v) for k, v in [(s, 0) for s in self.symbol_list])
        d['datetime'] = self.start_date
        return [d]

    def construct_all_holdings(self):
        """
        Конструирует список величин текущей стоимости позиций, используя start_date для определения момента, с которой должен начинаться временной индекс.
        """
        d = dict((k, v) for k, v in [(s, 0.0) for s in self.symbol_list])
        d['datetime'] = self.start_date
        d['cash'] = self.initial_capital
        d['commission'] = 0.0
        d['total'] = self.initial_capital
        return [d]

    def construct_current_holdings(self):
        """
        Конструирует словарь, который будет содержать мгновенное значение портфолио по всем инструментам.

        """
        d = dict((k, v) for k, v in [(s, 0.0) for s in self.symbol_list])
        d['cash'] = self.initial_capital
        d['commission'] = 0.0
        d['total'] = self.initial_capital
        return d

    def update_timeindex(self, event):
        """
        Добавляет новую запись в матрицу позиций для текущего бара рыночных данных. Отражает ПРЕДЫДУЩИЙ бар, т.е. на этой стадии известны все рыночные данные (OLHCVI). Используется MarketEvent из очередий событий.

        """
        bars = {}
        for sym in self.symbol_list:
            bars[sym] = self.bars.get_latest_bars(sym, N=1)

        # Update positions
        dp = dict((k, v) for k, v in [(s, 0) for s in self.symbol_list])
        dp['datetime'] = bars[self.symbol_list[0]][0][1]

        for s in self.symbol_list:
            dp[s] = self.current_positions[s]

        # Append the current positions
        self.all_positions.append(dp)

        # Update holdings
        dh = dict((k, v) for k, v in [(s, 0) for s in self.symbol_list])
        dh['datetime'] = bars[self.symbol_list[0]][0][1]
        dh['cash'] = self.current_holdings['cash']
        dh['commission'] = self.current_holdings['commission']
        dh['total'] = self.current_holdings['cash']

        for s in self.symbol_list:
            # Approximation to the real value
            market_value = self.current_positions[s] * bars[s][0][5]
            dh[s] = market_value
            dh['total'] += market_value

        # Append the current holdings
        self.all_holdings.append(dh)

    def update_market(self, event):
        """
         Обновляем значение на основе события рынка.
         Добавляем комисси за рыночные данные и бездействие
        """
        if event.type == 'MARKET':
            self.current_holdings['commission']+=event.commission
            self.current_holdings['cash'] -= event.commission
            self.current_holdings['total'] -= event.commission
            #if self.current_holdings['total']<100000:
    def update_positions_from_fill(self, fill):
        """
        Обрабатывает объект FillEvent и обновляет матрицу позиций так, чтобы она отражала новые позиции.

        Parameters:
        fill - The FillEvent object to update the positions with.
        """
        # Check whether the fill is a buy or sell
        fill_dir = 0
        if fill.direction == 'BUY':
            fill_dir = 1
        if fill.direction == 'SELL':
            fill_dir = -1

        # Список позиций обновляется новыми значениями
        self.current_positions[fill.symbol] += fill_dir * fill.quantity

    def update_holdings_from_fill(self, fill):
        """
        Использует объект FillEvent и обновляет матрицу holdings для отображения изменений.

    Параметры:
        fill - Объект FillEvent, который используется для обновлений.
        """
        # Check whether the fill is a buy or sell
        fill_dir = 0
        if fill.direction == 'BUY':
            fill_dir = 1
        if fill.direction == 'SELL':
            fill_dir = -1

        # Update holdings list with new quantities
        fill_cost = self.bars.get_latest_bars(fill.symbol)[0][5]  # Close price
        cost = fill_dir * fill_cost * fill.quantity
        self.current_holdings[fill.symbol] += cost
        self.current_holdings['commission'] += fill.commission
        self.current_holdings['cash'] -= (cost + fill.commission)
        self.current_holdings['total'] -= (cost + fill.commission)


    def update_fill(self, event):
        """
        Обновляет текущие позиции в портфолио и их рыночную стоимость на основе FillEvent.

        """
        if event.type == 'FILL':
            self.update_positions_from_fill(event)
            self.update_holdings_from_fill(event)



    def generate_naive_order(self, signal):
        """
        Просто передает OrderEvent как постоянное число акций на основе сигнального объекта без анализа рисков.

        Параметры:
        signal - Сигнальная информация SignalEvent.
        """
        order = None

        symbol = signal.symbol
        direction = signal.signal_type
        #strength = signal.strength

        #mkt_quantity = floor(100 * strength)
        mkt_quantity = self.buy_quantity
        cur_quantity = self.current_positions[symbol]
        order_type = 'MKT'

        if direction == 'LONG' and cur_quantity == 0:
            order = OrderEvent(symbol, order_type, mkt_quantity, 'BUY')
        if direction == 'SHORT' and cur_quantity == 0:
            order = OrderEvent(symbol, order_type, mkt_quantity, 'SELL')

        if direction == 'EXIT' and cur_quantity > 0:
            order = OrderEvent(symbol, order_type, abs(cur_quantity), 'SELL')
        if direction == 'EXIT' and cur_quantity < 0:
            order = OrderEvent(symbol, order_type, abs(cur_quantity), 'BUY')
        return order

    def update_signal(self, event):
        """
        На основе SignalEvent генерирует новые приказы в соответствии с логикой портфолио.

        """
        if event.type == 'SIGNAL':
            order_event = self.generate_naive_order(event)
            self.events.put(order_event)

    def create_equity_curve_dataframe(self):
        """
        Создает pandas DataFrame из списка словарей all_holdings.

        """
        curve = pd.DataFrame(self.all_holdings)
        curve.set_index('datetime', inplace=True)
        curve['returns'] = curve['total'].pct_change()
        curve['equity_curve'] = (1.0 + curve['returns']).cumprod()
        self.equity_curve = curve

    def output_summary_stats(self):
        """
        Создает список статистических показателей для портфолио — коэффициент Шарпа и данные по просадке.
        """
        self.create_equity_curve_dataframe()
        total_return = self.equity_curve['equity_curve'][-1]
        returns = self.equity_curve['returns']
        pnl = self.equity_curve['equity_curve']

        sharpe_ratio = create_sharpe_ratio(returns)
        max_dd, dd_duration = create_drawdowns(pnl)

        stats = [("Total Return", "%0.2f%%" % ((total_return - 1.0) * 100.0)),
                 ("Sharpe Ratio", "%0.2f" % sharpe_ratio),
                 ("Max Drawdown", "%0.2f%%" % (max_dd * 100.0)),
                 ("Drawdown Duration", "%d" % dd_duration)]
        return stats

class NaivePortfolio_add_founds(Portfolio):
    """

Объект NaivePortfolio создан для слепой (т.е. без всякого риск-менеджмента)  отправки приказов на покупку/продажу установленного количество акций, в брокерскую систему. Используется для тестирования простых стратегий вроде BuyAndHoldStrategy.
Портрфель пополняется два раза в месяц одинаковыми суммами. Покупка совершается, только если есть деньги на покупку.

    """

    def __init__(self, bars, events, start_date, initial_capital=0, buy_quantity=10.0, add_funds=500):
        """
        Инициализирует портфолио на основе информации из баров и очереди событий. Также включает дату и время начала и размер начального капитала (в долларах, если не указана другая валюта).

        Parameters:
        bars - The DataHandler object with current market data.
        events - The Event Queue object.
        start_date - The start date (bar) of the portfolio.
        initial_capital - The starting capital in USD.
        """
        self.bars = bars
        self.events = events
        self.symbol_list = self.bars.symbol_list
        self.start_date = start_date
        self.initial_capital = initial_capital
        self.buy_quantity = buy_quantity

        self.add_funds = add_funds

        self.last_add_date = start_date

        self.last_buy = start_date

        self.cashflow = self.construct_cashflow()

        self.all_positions = self.construct_all_positions()
        self.current_positions = dict((k, v) for k, v in [(s, 0) for s in self.symbol_list])

        self.all_holdings = self.construct_all_holdings()
        self.current_holdings = self.construct_current_holdings()

    def construct_cashflow(self):
        """
     Конструирует список денежного потока используя стартовую дату и стартовый капитал в качестве первой отрицательной выплаты.
     !! Проверить код на вменяемость !!
        """

        d = pd.DataFrame([dict((k,v) for k,v in [(s,0) for s in self.symbol_list])])
        d['datetime'] = self.start_date
        d['total'] = - self.initial_capital
        d.set_index('datetime', inplace=True)

        return d

    def construct_all_positions(self):
        """
     Конструирует список позиций, используя start_date для определения момента, с которой должен начинаться временной индекс.
        """
        d = dict((k, v) for k, v in [(s, 0) for s in self.symbol_list])
        d['datetime'] = self.start_date
        return [d]

    def construct_all_holdings(self):
        """
        Конструирует список величин текущей стоимости позиций, используя start_date для определения момента, с которой должен начинаться временной индекс.
        """
        d = dict((k, v) for k, v in [(s, 0.0) for s in self.symbol_list])
        d['datetime'] = self.start_date
        d['cash'] = self.initial_capital
        d['commission'] = 0.0
        d['total'] = self.initial_capital


        return [d]

    def construct_current_holdings(self):
        """
        Конструирует словарь, который будет содержать мгновенное значение портфолио по всем инструментам.

        """
        d = dict((k, v) for k, v in [(s, 0.0) for s in self.symbol_list])
        d['cash'] = self.initial_capital
        d['commission'] = 0.0
        d['total'] = self.initial_capital

        return d

    def update_timeindex(self, event):
        """
        Добавляет новую запись в матрицу позиций для текущего бара рыночных данных. Отражает ПРЕДЫДУЩИЙ бар, т.е. на этой стадии известны все рыночные данные (OLHCVI). Используется MarketEvent из очередий событий.

        """
        bars = {}
        for sym in self.symbol_list:
            bars[sym] = self.bars.get_latest_bars(sym, N=1)

        # Update positions
        dp = dict((k, v) for k, v in [(s, 0) for s in self.symbol_list])
        dp['datetime'] = bars[self.symbol_list[0]][0][1]

        for s in self.symbol_list:
            dp[s] = self.current_positions[s]

        # Append the current positions
        self.all_positions.append(dp)

        # Update holdings
        dh = dict((k, v) for k, v in [(s, 0) for s in self.symbol_list])
        dh['datetime'] = bars[self.symbol_list[0]][0][1]
        dh['cash'] = self.current_holdings['cash']
        dh['commission'] = self.current_holdings['commission']
        dh['total'] = self.current_holdings['cash']



        for s in self.symbol_list:
            # Approximation to the real value
            market_value = self.current_positions[s] * bars[s][0][5]
            dh[s] = market_value
            dh['total'] += market_value
            
        # Append the current holdings
        self.all_holdings.append(dh)


    def update_market(self, event):
        """
         Обновляем значение на основе события рынка.
         Добавляем комисси за рыночные данные и бездействие
        """

        if event.type == 'MARKET':


            if self.all_holdings[-1]['datetime'].month != event.datetime.month:
                full_cost = 0.0
                self.current_holdings['commission']+=full_cost
                self.current_holdings['cash'] -= full_cost
                self.current_holdings['total'] -= full_cost





        """
        Пополняем портфель, в нужные даты
        """
        if event.datetime > self.last_add_date:
            self.adding_funds()
            #добавляем строку в dataframe cashflow
            if self.add_funds > 0:
                self.cashflow.loc[event.datetime, 'total'] = -self.add_funds

            #Получаем первое число следующего месяца, для переноса даты пополнения
            next_month = np.datetime64(self.last_add_date, 'M') + np.timedelta64(1, 'M')
            self.last_add_date = np.datetime64(next_month, 'D')




    def update_positions_from_fill(self, fill):
        """
        Обрабатывает объект FillEvent и обновляет матрицу позиций так, чтобы она отражала новые позиции.

        Parameters:
        fill - The FillEvent object to update the positions with.
        """
        # Check whether the fill is a buy or sell
        fill_dir = 0
        if fill.direction == 'BUY':
            fill_dir = 1
        if fill.direction == 'SELL':
            fill_dir = -1

        # Список позиций обновляется новыми значениями
        self.current_positions[fill.symbol] += fill_dir * fill.quantity

    def update_holdings_from_fill(self, fill):
        """
        Использует объект FillEvent и обновляет матрицу holdings для отображения изменений.

    Параметры:
        fill - Объект FillEvent, который используется для обновлений.
        """
        # Check whether the fill is a buy or sell
        fill_dir = 0
        if fill.direction == 'BUY':
            fill_dir = 1
        if fill.direction == 'SELL':
            fill_dir = -1

        # Update holdings list with new quantities
        fill_cost = self.bars.get_latest_bars(fill.symbol)[0][5]  # Close price
        cost = fill_dir * fill_cost * fill.quantity
        self.current_holdings[fill.symbol] += cost
        self.current_holdings['commission'] += fill.commission
        self.current_holdings['cash'] -= (cost + fill.commission)
        self.current_holdings['total'] -= (cost + fill.commission)

        #Заносим денежный поток на покупку
        self.cashflow.loc[fill.timeindex, fill.symbol] = -(cost + fill.commission)

    def update_fill(self, event):
        """
        Обновляет текущие позиции в портфолио и их рыночную стоимость на основе FillEvent.

        """
        if event.type == 'FILL':
            self.update_positions_from_fill(event)
            self.update_holdings_from_fill(event)



    def generate_naive_order(self, signal):
        """
        Просто передает OrderEvent как постоянное число акций на основе сигнального объекта без анализа рисков.

        Параметры:
        signal - Сигнальная информация SignalEvent.
        """
        order = None

        symbol = signal.symbol
        direction = signal.signal_type
        #strength = signal.strength

        #mkt_quantity = floor(100 * strength)
        mkt_quantity = self.buy_quantity
        cur_quantity = self.current_positions[symbol]
        order_type = 'MKT'
        # Формируется приказ, если хватает денег


        if direction == 'LONG' and cur_quantity == 0:
            order = OrderEvent(symbol, order_type, mkt_quantity, 'BUY', signal.datetime)
        if direction == 'SHORT' and cur_quantity == 0:
            order = OrderEvent(symbol, order_type, mkt_quantity, 'SELL', signal.datetime)

        if direction == 'EXIT' and cur_quantity > 0:
            order = OrderEvent(symbol, order_type, abs(cur_quantity), 'SELL', signal.datetime)
        if direction == 'EXIT' and cur_quantity < 0:
            order = OrderEvent(symbol, order_type, abs(cur_quantity), 'BUY', signal.datetime)
        return order

    def update_signal(self, event):
        """
        На основе SignalEvent генерирует новые приказы в соответствии с логикой портфолио.

        """
        if event.type == 'SIGNAL':
            price = self.bars.get_latest_bars(event.symbol)[0][5]

            if self.current_holdings['cash'] > self.buy_quantity*price:
                order_event = self.generate_naive_order(event)
                self.events.put(order_event)

    def create_equity_curve_dataframe(self):
        """
        Создает pandas DataFrame из списка словарей all_holdings.

        """
        curve = pd.DataFrame(self.all_holdings)
        curve.set_index('datetime', inplace=True)
        curve['returns'] = curve['total'].pct_change()
        curve['equity_curve'] = (1.0 + curve['returns']).cumprod()
        self.equity_curve = curve

    def output_summary_stats(self):
        """
        Создает список статистических показателей для портфолио — коэффициент Шарпа и данные по просадке.
        """

        #Считаем взвещенную по времени норму доходности портфеля
        #Готовим dataframe
        df = pd.DataFrame(self.all_holdings)
        df.set_index('datetime', inplace=True)
        df.loc[self.cashflow.index, 'cashflow'] = self.cashflow.total
        df.fillna(0, inplace=True)

        twrr_tr, twrr_mean, twrr_years = twrr(df[['total', 'cashflow']])



        self.create_equity_curve_dataframe()

        #добавляем итоговые значения к cashflow
        for col in list(self.cashflow):
            self.cashflow.loc[self.equity_curve.index[-1], col] = self.equity_curve[col][-1]
        self.cashflow.fillna(0.0, inplace=True)

        #Считаем внутреннюю норму доходности (взвешенную по денежной стоимости норму доходности)
        xirr_list = []
        for col in list(self.cashflow):
            xirr_list.append(xirr_1(self.cashflow[col])*100.0)



        stats = [('XIRR', f'{xirr_list:.2f}%'),
                 ('TWRR total return', f'{twrr_tr:.2f}%'),
                 ('TWRR mean', f'{twrr_mean:.2f}%'),
                 #проверить вывод на костыли
                 ('TWRR by year', twrr_years.iloc[:,0].apply(lambda x: f'{x:.2f}%'))]
        return stats

    def adding_funds(self):
        """
        Процедура пополнения портфеля деньгами
        """

        self.current_holdings['cash'] += self.add_funds
        self.current_holdings['total'] += self.add_funds

