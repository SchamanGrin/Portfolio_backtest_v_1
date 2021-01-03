# performance.py

import numpy as np
import pandas as pd

from scipy import optimize as op

SECONDS_PER_YEAR = 365.0 * 24 * 60 * 60

def create_sharpe_ratio(returns, periods=252):
    """
    Создает коэффициент Шарпа для стратегии, основанной на бенчмарке ноль (нет информации о рисках ).

    Параметры:
    returns -  Series из Pandas представляет процент прибыли за период - Дневной (252), Часовой (252*6.5), Минутный (252*6.5*60) и т.п..
    """
    return np.sqrt(periods) * (np.mean(returns)) / np.std(returns)


def create_drawdowns(equity_curve):
    """
    Вычисляет крупнейшее падение от пика до минимума кривой PnL и его длительность. Требует возврата  pnl_returns в качестве pandas Series.

    Параметры:
    pnl - pandas Series, представляющая процент прибыли за период.

    Прибыль:
    drawdown, duration - Наибольшая просадка и ее длительность
    """
    # Подсчет общей прибыли
    # и установка High Water Mark
    # Затем создаются серии для просадки и длительности
    hwm = [0]
    eq_idx = equity_curve.index
    drawdown = pd.Series(index=eq_idx)
    duration = pd.Series(index=eq_idx)

    # Цикл проходит по диапазону значений индекса
    for t in range(1, len(eq_idx)):
        cur_hwm = max(hwm[t - 1], equity_curve[t])
        hwm.append(cur_hwm)
        drawdown[t] = hwm[t] - equity_curve[t]
        duration[t] = 0 if drawdown[t] == 0 else duration[t - 1] + 1
    return drawdown.max(), duration.max()

def xnpv(valuesPerDate, rate):
    """
    Calculate the net present value of a series of cashflows at irregular intervals.
    Расчет чистой приведенной стоимости денежных потоков на нерегулярных интервалах
    Arguments
    Аргументы:
    ---------
    * rate: the discount rate to be applied to the cash flows
    *rate: коэффициент дисконтирвоания
    * cashflows: a list object in which each element is a tuple of the form (date, amount), where date is a python datetime.date object and amount is an integer or floating point number. Cash outflows (investments) are represented with negative amounts, and cash inflows (returns) are positive amounts.
    *cashflows:
    Returns
    -------
    * returns a single value which is the NPV of the given cash flows.
    Возращает значение приведенной стоимости денежных потоков
    Notes
    ---------------
    * The Net Present Value is the sum of each of cash flows discounted back to the date of the first cash flow. The discounted value of a given cash flow is A/(1+r)**(t-t0), where A is the amount, r is the discout rate, and (t-t0) is the time in years from the date of the first cash flow in the series (t0) to the date of the cash flow being added to the sum (t).
    Приведенная стоимость денежных потоков это сумма каждого денежного потока приведенного к дате первого денежного потока.
    * This function is equivalent to the Microsoft Excel function of the same name.
    Эта функция соответсвует Excel функции с таким же наименованием
    """

    if rate == -1.0:
        return float('inf')

    t0 = min(valuesPerDate.keys())

    if rate <= -1.0:
        return sum(
            [-abs(vi) / (-1.0 - rate) ** ((ti - t0).total_seconds() / SECONDS_PER_YEAR) for ti, vi in
             valuesPerDate.items()])

    return sum(
        [vi / (1.0 + rate) ** ((ti - t0).total_seconds() / SECONDS_PER_YEAR) for ti, vi in valuesPerDate.items()])


def xirr(valuesPerDate):
    """
    Calculate the Internal Rate of Return of a series of cashflows at irregular intervals.
    Расчет внутренней нормы доходности при нерегулярных денежных потоках

    Arguments
    Аргументы
    ---------
    * valuesPerDate:  * справочник где:
     key - дата денежного потока, объект типа datetime
     value - значение денежного потока, объект типа foat
    Входящий денежный поток (инвестциии) представлены отрицательными значениями, а исходящий денежный поток (доход)
    представлены положительными значениями

    Returns
    --------
    * Returns the IRR as a single value
    * Возвращает годовую взвешенную по деньгам доходность, как значение

    Notes
    ----------------
    * The Internal Rate of Return (IRR) is the discount rate at which the Net Present Value (NPV) of a series of cash flows is equal to zero. The NPV of the series of cash flows is determined using the xnpv function in this module. The discount rate at which NPV equals zero is found using the secant method of numerical solution.
    * This function is equivalent to the Microsoft Excel function of the same name.
    * For users that do not have the scipy module installed, there is an alternate version (commented out) that uses the secant_method function defined in the module rather than the scipy.optimize module's numerical solver. Both use the same method of calculation so there should be no difference in performance, but the secant_method function does not fail gracefully in cases where there is no solution, so the scipy.optimize.newton version is preferred.
    """
    if not valuesPerDate:
        return None

    if all(v >= 0 for v in valuesPerDate.values()):
        return float("inf")
    if all(v <= 0 for v in valuesPerDate.values()):
        return -float("inf")

    result = None
    try:
        result = op.newton(lambda r: xnpv(valuesPerDate, r), 0)
    except (RuntimeError, OverflowError):  # Failed to converge?
        result = op.brentq(lambda r: xnpv(valuesPerDate, r), -0.999999999999999, 1e20, maxiter=10 ** 6)

    if not isinstance(result, complex):
        return result
    else:
        return None

def twrr(cashflow):
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

    """

    start_date = cashflow.index[0]
    cf = cashflow.copy()

    # Формируем периоды расчета изменения стоимости портфеля между пополнениями / изъятиями
    # Формируем столбец с накопительным количеством строк с не нулевым cashflow и смещаем на один, что бы в интервал
    # попадала строка с следующим casflow для расчета общей стоимоти до изменения

    cf.loc[:, 'interval'] = np.cumsum(cf[cf.columns[1]] != 0).shift(1)

    # Формируем столбец с расчетом изменения общей стоимости портфеля за каждый период по сравнению
    # с стоимостью в предыдущим периодом убирая из расчета cashflow за текущий период.
    cf['change'] = ((cf[cf.columns[0]] + cf[cf.columns[1]]) / cf[cf.columns[0]].shift(periods=1)).fillna(1)

    # Для каждой строки считаем накопительное произведение изменение общей стоимости портфеля в рамках периода
    cf['prod_interval'] = cf.groupby('interval').change.cumprod().fillna(1)

    # Готовим вспомогательный столбец для хранения полного изменения по каждому периоду
    cf['prod_previous_period'] = cf['prod_interval'][cf[cf.columns[1]] != 0]
    cf['prod_previous_period'].fillna(1, inplace=True)

    # Расчитываем накопленную доходность умножая изменение за текущий период на общие изменения за прошлые периоды
    cf['revenue'] = cf.prod_previous_period.cumprod()

    # Рассчитываем годовую доходность, умножая на приведенный к году текущий срок с даты старта портфеля
    cf['revenue'][cf[cf.columns[1]] == 0] = cf.revenue * cf.prod_interval

    cf['annual return'] = cf.revenue ** (365. / (cf.index - start_date).days) - 1

    return [cf['annual return'][-1], cf['annual return']]


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
         data - массив значений доходности в каждый момент времени, данные дополняются в переданный на вход масссив данных
        """



    for items in method:
        if items not in ['twrr', 'mwrr']:
            raise Exception(f'{items} не входит в список используемых методоы расчета')


    result = {}
    if 'twrr' in method:
        p_twrr, data = twrr(cashflows)
        result['twrr'] = p_twrr
        cashflows.loc[:,'twrr'] = data


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


