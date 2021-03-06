# config.py
import os
import queue  # import Queue

from alpha import AlphaVantage

from data import HistoricCSVDataHandler
from execution import SimulatedExecutionHandler
from portfolio import NaivePortfolio, NaivePortfolio_add_founds
from strategy import BuyAndHoldStrategy
from config import backconfig



conf = backconfig()

symbol_list = conf.values['list']['tickets']
dir_path = 'symbol/'
dir_list = [x.split('.')[0] for x in os.listdir(dir_path)]

'Собираем список тикетов, которых нет в каталоге'
av_list = []
for symbol in symbol_list:
    if symbol not in dir_list:
        av_list.append(symbol)

if av_list:
    AlphaVantage(symbol_list=av_list, key=conf.keys['alpha vantage']['key'], url=conf.keys['alpha vantage']['url']).take_csv(outputsize='full')


events = queue.Queue()
start_date = conf.values['date']['start_date']

bars = HistoricCSVDataHandler(events, dir_path, symbol_list, start_date)
port = NaivePortfolio_add_founds(bars, events, start_date, initial_capital=conf.values['money']['initial_capital'], buy_quantity=10.0, add_funds=conf.values['money']['add_funds'])
strategy = BuyAndHoldStrategy(bars, events, port)
broker = SimulatedExecutionHandler(events)

while True:
    # Обновляем бары (код для бэктестинга, а не живой торговли)
    if bars.continue_backtest == True:
        bars.update_bars()
    else:
        break

    # Обрабатываем события
    while True:
        try:
            event = events.get(False)
        except queue.Empty:
            break
        else:
            if event is not None:
                if event.type == 'MARKET':
                    port.update_market(event)
                    port.update_timeindex(event)
                    strategy.calculate_signals(event)
                elif event.type == 'SIGNAL':
                    port.update_signal(event)

                elif event.type == 'ORDER':
                    broker.execute_order(event)

                elif event.type == 'FILL':
                    port.update_fill(event)

    # следующий удар сердца через 10 минут
    # time.sleep(60)

#Вывод результатов по портфелю
print('Стоимость портфеля:')
for key, value in port.all_holdings[-1].items():
    print(f'{key}: {value}')

print('****************')
print('Количество позиций:')
for key, value in port.current_positions.items():
    
    print(f'{key}: {value}')

print('****************')

#Вывод результатов по бэктесту
print('Результативность порфтеля:')
result = port.output_summary_stats()
for ind in result:
    print(ind[0],':',ind[1])

