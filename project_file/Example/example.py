import time
import numpy as np
import pandas as pd
import scipy.optimize as op
from performance import twrr, xirr, xnpv

def twrr(cf):
    """
    Расчет взвешенной по времени доходности, основанной на не равных периодах внесения и изъятия денег в формате DataFrame

    !!!В последней строке положительный cashflow указывать не нужно!!!

    :param cashflows: pandas DataFrame с столбцами ['total', 'cashflow'], имя столбца не важно, важна последовательность.
     index - дата в формает timestamp или datetime64, в которую был произведегн денежный поток и определена стоимость портфеля
     total - стоимость портфеля на соответствующую дату в формате float
     cashflow - размер денежного потока на соответствующую дату в формате float

    :return:
    *twrr - взвешенная по времени среднегодовая доходность
    *total_return - доходность накопительным итогом за весь период
    *
    *значения взвешенной по времени доходности на каждый период в входном датафрейме
    """

    start_date = cf.index[0]

    # Формируем периоды расчета изменения стоимости портфеля между пополнениями / изъятиями
    # Формируем столбец с накопительным количеством строк с не нулевым cashflow и смещаем на один, что бы в интервал
    # попадала строка с следующим casflow для расчета общей стоимоти до изменения
    """ cf['interval'] = np.cumsum(cf[cf.columns[1]] != 0)
    cf['interval'] = cf['interval'].shift(1)"""

    cf.loc[:,'interval'] = np.cumsum(cf[cf.columns[1]] != 0).shift(1)


    """cf['change'] = (cf['total'] + cf['cashflow']) / cf['total'].shift(periods=1)
    cf['change'].fillna(1, inplace=True)"""

    # Формируем столбец с расчетом изменения общей стоимости портфеля за каждый период по сравнению
    # с стоимостью в предыдущим периодом убирая из расчета cashflow за текущий период.
    cf['change'] = ((cf['total'] + cf['cashflow']) / cf['total'].shift(periods=1)).fillna(1)

    """cf['prod_interval'] = cf.groupby('interval').change.cumprod()
    cf['prod_interval'].fillna(1, inplace=True)"""

    # Для каждой строки считаем накопительное произведение изменение общей стоимости портфеля в рамках периода
    cf['prod_interval'] = cf.groupby('interval').change.cumprod().fillna(1)

    cf['prod_previous_period'] = cf['prod_interval'][cf['cashflow'] != 0]
    cf['prod_previous_period'].fillna(1, inplace=True)
    cf['revenue'] = cf.prod_previous_period.cumprod()


    cf['revenue'][cf['cashflow'] == 0 ] = cf.revenue * cf.prod_interval


    cf['annual return'] = cf.revenue**(365./(cf.index - start_date).days) - 1

    return [cf['revenue'][-1], cf['revenue', 'annual return']]




data = pd.read_csv(
    'symbol/SPY.csv', header=0, index_col=0,
                names=['timestamp', 'open', 'low', 'high', 'close', 'volume'], parse_dates=['timestamp']
)

start_date = np.datetime64('2010-01-01')
end_date = np.datetime64('2010-01-30')
data.reindex(np.array(data.index, dtype='datetime64'))
date = ['2010-01-04', '2010-01-08','2010-01-15', '2010-01-29']
count = [10,10,-10,-10]

data_cashflow = pd.DataFrame(data.loc[end_date:start_date, 'close'][::-1])
data_cashflow.loc[date,'count'] = count
data_cashflow['cashflow'] = -data_cashflow['count']*data_cashflow['close']
data_cashflow.fillna(0, inplace=True)
data_cashflow['total'] = np.cumsum(data_cashflow['count'])*data_cashflow['close']


twrr_total = twrr(data_cashflow[['total', 'cashflow']])
print()


