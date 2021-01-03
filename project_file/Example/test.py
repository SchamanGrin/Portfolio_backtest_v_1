import numpy as np
import pandas as pd
import time
import scipy
import xirr

from pathlib import Path
# from datetime import date

import performance as perfom

def create_return(cashflows, method = ['twrr', 'mwrr']):
    """
        Метод расчитывает годовую доходность портфеля с учетом пополнений и изъятий денежных средств портфеля.
        Расчет производится на основе DataFrame с данными о стоимости портфеля и денежного потока
        на период. Для расчета используется два метода:
        - взвешенный по деньгам (money-weighted return)
        - взвешенный по времени (time-weighted return)


        :param cashflows: pandas DataFrame с столбцами ['total', 'cashflow'], имя столбца не важно, важна последовательность.
         index - дата в формает timestamp или datetime64, в которую был произведегн денежный поток и определена стоимость портфеля
         total - стоимость портфеля на соответствующую дату в формате float
         cashflow - размер денежного потока на соответствующую дату в формате float

        :param method: список методов для расчета доходности, определяет метод по которому будет считаться доходность:
         'twrr' - time-weighted rate of return. Доходность, взвешенная по времени
         'mwrr' -  money-weighted rate of return. Долходность, взвешенная по деньгам
         может принимать значения:
         ['twrr']
         ['twrr', 'mwrr']
         ['mwrr']

        :return: словарь {'twrr';{'return': float, 'data': DataFrame}, 'mwrr':{'return': float, 'data': DataFrame}}
         return - значение годовой доходности посчитанной соответствующим методом в формате float
         data - массив значений доходности в каждый момент времени
        """

    SECONDS_PER_YEAR = 365.0*24*60*60

    def xnpv(valuesPerDate, rate):
        '''Calculate the irregular net present value.
        '''

        if rate == -1.0:
            return float('inf')

        t0 = min(valuesPerDate.keys())

        if rate <= -1.0:
            return sum(
                [-abs(vi) / (-1.0 - rate) ** ((ti - t0).total_seconds() / SECONDS_PER_YEAR) for ti, vi in valuesPerDate.items()])

        return sum([vi / (1.0 + rate) ** ((ti - t0).total_seconds() / SECONDS_PER_YEAR) for ti, vi in valuesPerDate.items()])

    def xirr(valuesPerDate):
        '''Calculate the irregular internal rate of return.'''

        if not valuesPerDate:
            return None

        if all(v >= 0 for v in valuesPerDate.values()):
            return float("inf")
        if all(v <= 0 for v in valuesPerDate.values()):
            return -float("inf")

        result = None
        try:
            result = scipy.optimize.newton(lambda r: xnpv(valuesPerDate, r), 0)
        except (RuntimeError, OverflowError):  # Failed to converge?
            result = scipy.optimize.brentq(lambda r: xnpv(valuesPerDate, r), -0.999999999999999, 1e20, maxiter=10 ** 6)

        if not isinstance(result, complex):
            return result
        else:
            return None

    def twrr(cf):
        """
        Расчет взвешенной по времени доходности, при неравномерных на не равномерных периодах внесения и изъятия денег в формате DataFrame


        :param cashflows: pandas DataFrame с столбцами ['total', 'cashflow'], имя столбца не важно, важна последовательность.
         index - дата в формает timestamp или datetime64, в которую был произведегн денежный поток и определена стоимость портфеля
         total - стоимость портфеля на соответствующую дату в формате float
         cashflow - размер денежного потока на соответствующую дату в формате float

         каждая строка равна дневным значениям.

        :return: [twrr, annual return, revenue]
        *twrr - взвешенная по времени среднегодовая доходность за весь период. формат: float
        *annual return - годовая доходность на каждый день
        *total_return - доходность накопительным итогом за весь период
        *
        *значения взвешенной по времени доходности на каждый период в входном датафрейме
        """

        start_date = cf.index[0]
        cf = cf.copy()

        # Формируем периоды расчета изменения стоимости портфеля между пополнениями / изъятиями
        # Формируем столбец с накопительным количеством строк с не нулевым cashflow и смещаем на один, что бы в интервал
        # попадала строка с следующим casflow для расчета общей стоимоти до изменения

        cf.loc[:, 'interval'] = np.cumsum(cf[cf.columns[1]] != 0).shift(1)

        # Формируем столбец с расчетом изменения общей стоимости портфеля за каждый период по сравнению
        # с стоимостью в предыдущим периодом убирая из расчета cashflow за текущий период.
        cf['change'] = ((cf['total'] + cf['cashflow']) / cf['total'].shift(periods=1)).fillna(1)

        # Для каждой строки считаем накопительное произведение изменение общей стоимости портфеля в рамках периода
        cf['prod_interval'] = cf.groupby('interval').change.cumprod().fillna(1)

        # Готовим вспомогательный столбец для хранения полного изменения по каждому периоду
        cf['prod_previous_period'] = cf['prod_interval'][cf['cashflow'] != 0]
        cf['prod_previous_period'].fillna(1, inplace=True)

        # Расчитываем накопленную доходность умножая изменение за текущий период на общие изменения за прошлые периоды
        cf['revenue'] = cf.prod_previous_period.cumprod()

        # Рассчитываем годовую доходность, умножая на приведенный к году текущий срок с даты старта портфеля
        cf['revenue'][cf['cashflow'] == 0] = cf.revenue * cf.prod_interval

        cf['annual return'] = cf.revenue ** (365. / (cf.index - start_date).days) - 1

        return [cf['annual return'][-1], cf['annual return']]

    result = {}
    if 'twrr' in method:
        twrr, data = twrr(cashflows)
        result['twrr'] = twrr
        cashflows['twrr'] = data

    t = time.time()
    if 'mwrr' in method:
        # переводим DataFrame в numpy для скорости выполнения оптимизации
        cf_np = cashflows[cashflows.columns[1]].reset_index().to_numpy()

        dict_res = []
        # для каждого значения стоимости портфеля, считаем mwrr
        for i in range(1, len(cf_np)):
            arr_cf = cf_np[:i + 1].copy()
            # прибавляем положительный итоговый денежный поток на дату, равный стоимости портфеля
            arr_cf[i, 1] += cashflows[cashflows.columns[0]][i]
            dict_t = {k: v for k, v in arr_cf[np.abs(arr_cf[:, 1]) > 1e-10]}
            dict_res += [xirr(dict_t)]

        cashflows.loc[:, 'mwrr'] = [0] + dict_res

        result['mwrr'] = cashflows['mwrr'][-1]
    result['data'] = cashflows

    return result

def create_return_xirr(cashflows, method = ['twrr', 'mwrr']):
    """
    Метод расчитывает доходность портфеля с учетом пополнений и изъятий денежных средств портфеля.
    Расчет производится на основе DataFrame с ежедневными данными о стоимости портфеля и денежного потока
    на дату. Для расчета используется два метода:
    - взвешенный по деньгам (money-weighted return)
    - взвешенный по времени (time-weighted return)


    :param cashflows: pandas DataFrame с столбцами ['total', 'cashflow'], имя столбца не важно, важна последовательность.
     index - дата в формает timestamp или datetime64, в которую был произведегн денежный поток и определена стоимость портфеля
     total - стоимость портфеля на соответствующую дату в формате float
     cashflow - размер денежного потока на соответствующую дату в формате float

    :param method: список методов для расчета доходности, определяет метод по которому будет считаться доходность:
     'twrr' - time-weighted rate of return. Доходность, взвешенная по времени
     'mwrr' -  money-weighted rate of return. Долходность, взвешенная по деньгам
     может принимать значения:
     ['twrr']
     ['twrr', 'mwrr']
     ['mwrr']

    :return: словарь {'twrr';{'return': float, 'data': DataFrame}, 'mwrr':{'return': float, 'data': DataFrame}}
     return - значение годовой доходности посчитанной соответствующим методом в формате float
     data - массив значений доходности в каждый момент времени
    """

    def twrr(cf):
        """
        Расчет взвешенной по времени доходности, при неравномерных на не равномерных периодах внесения и изъятия денег в формате DataFrame


        :param cashflows: pandas DataFrame с столбцами ['total', 'cashflow'], имя столбца не важно, важна последовательность.
         index - дата в формает timestamp или datetime64, в которую был произведегн денежный поток и определена стоимость портфеля
         total - стоимость портфеля на соответствующую дату в формате float
         cashflow - размер денежного потока на соответствующую дату в формате float

         каждая строка равна дневным значениям.

        :return: [twrr, annual return, revenue]
        *twrr - взвешенная по времени среднегодовая доходность за весь период. формат: float
        *annual return - годовая доходность на каждый день
        *total_return - доходность накопительным итогом за весь период
        *
        *значения взвешенной по времени доходности на каждый период в входном датафрейме
        """

        start_date = cf.index[0]
        cf = cf.copy()

        # Формируем периоды расчета изменения стоимости портфеля между пополнениями / изъятиями
        # Формируем столбец с накопительным количеством строк с не нулевым cashflow и смещаем на один, что бы в интервал
        # попадала строка с следующим casflow для расчета общей стоимоти до изменения

        cf.loc[:, 'interval'] = np.cumsum(cf[cf.columns[1]] != 0).shift(1)

        # Формируем столбец с расчетом изменения общей стоимости портфеля за каждый период по сравнению
        # с стоимостью в предыдущим периодом убирая из расчета cashflow за текущий период.
        cf['change'] = ((cf['total'] + cf['cashflow']) / cf['total'].shift(periods=1)).fillna(1)

        # Для каждой строки считаем накопительное произведение изменение общей стоимости портфеля в рамках периода
        cf['prod_interval'] = cf.groupby('interval').change.cumprod().fillna(1)

        # Готовим вспомогательный столбец для хранения полного изменения по каждому периоду
        cf['prod_previous_period'] = cf['prod_interval'][cf['cashflow'] != 0]
        cf['prod_previous_period'].fillna(1, inplace=True)

        # Расчитываем накопленную доходность умножая изменение за текущий период на общие изменения за прошлые периоды
        cf['revenue'] = cf.prod_previous_period.cumprod()

        # Рассчитываем годовую доходность, умножая на приведенный к году текущий срок с даты старта портфеля
        cf['revenue'][cf['cashflow'] == 0] = cf.revenue * cf.prod_interval

        cf['annual return'] = cf.revenue ** (365. / (cf.index - start_date).days) - 1

        return [cf['annual return'][-1], cf['annual return']]

    result = {}
    if 'twrr' in method:
        twrr, data  = twrr(cashflows)
        result['twrr'] = twrr
        cashflows['twrr'] = data

    t = time.time()
    if 'mwrr' in method:
        #переводим DataFrame в numpy для скорости выполнения оптимизации
        cf_np = cashflows[cashflows.columns[1]].reset_index().to_numpy()

        dict_res = []

        # для каждого значения стоимости портфеля, считаем mwrr
        for i in range(1, len(cf_np)):
            arr_cf = cf_np[:i+1].copy()
            #прибавляем положительный итоговый денежный поток на дату, равный стоимости портфеля
            arr_cf[i, 1] += cashflows[cashflows.columns[0]][i]
            dict_t = {k:v for k,v in arr_cf[np.abs(arr_cf[:, 1]) > 1e-10]}
            dict_res += [xirr.xirr(dict_t)]

        cashflows.loc[:,'mwrr'] = [0] + dict_res

        result['mwrr'] = cashflows['mwrr'][-1]
    result['data'] = cashflows

    return result

def read_csv(path):
    return pd.read_csv(path, header=0, index_col=0,
                     names=['timestamp', 'close', 'cf', 'count','cf and total end', 'total', 'result'], parse_dates=['timestamp'], sep=';',
                     dayfirst=True, decimal=',')

path = Path('testcase') / 'negative_far_many_cf.csv'

data = read_csv(path)

cf_data = pd.DataFrame(data[['cf and total end']])
cf_data = cf_data['cf and total end'].reset_index()
cf_data['timestamp'] = cf_data['timestamp'].astype('datetime64[ns]')
arr_cf = cf_data[['timestamp', 'cf and total end']].to_numpy()

data_rename = data[['total', 'cf']].copy()
data_rename.rename(columns={'cf':'cashflow'}, inplace=True)
dict_data = dict(zip(data.index,data['cf']))

print('моя функция:')
t1 = time.time()
res = perfom.create_return(data[['total','cf']], ['mwrr'])['mwrr']
print(f'mwrr:{res:.8f}  за {time.time() - t1:.2f} c.')
print('----------------------')

print('внешний xirr')
t2 = time.time()
res_xirr = create_return_xirr(data[['total','cf']], ['mwrr'])['mwrr']
print(f'mwrr:{res_xirr:.8f}  за {time.time() - t2:.2f} c.')
print('----------------------')

print('переписанный xirr')
t3 = time.time()
res_xirr_dict = create_return(data[['total','cf']], ['mwrr'])['mwrr']
print(f'mwrr:{res_xirr_dict:.8f}  за {time.time() - t3:.2f} c.')
