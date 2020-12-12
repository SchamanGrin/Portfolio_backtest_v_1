# performance.py

import numpy as np
import pandas as pd

PERIOD_PER_DAYS = 365.0

from scipy import optimize as op


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

def xnpv(rate, cashflows):
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
    global PERIOD_PER_DAYS
    t0 = cashflows[0, 0]

    if rate == -1.0:
        return np.inf
    r = 1 + rate

    if rate <= -1:
        r = -r
        return sum([-abs(cf) / r**((t - t0).days / PERIOD_PER_DAYS) for t, cf in cashflows])
    if rate == 0:
        return sum(cashflows[:, 1])

    return sum([cf / r**((t - t0).days / PERIOD_PER_DAYS) for t, cf in cashflows])


def xirr(cashflows, guess=0.1):
    """
    Calculate the Internal Rate of Return of a series of cashflows at irregular intervals.
    Расчет внутренней нормы доходности при нерегулярных денежных потоках

    Arguments
    Аргументы
    ---------
    * cashflows:  * денежный поток, двумерный массив, со столбцами дата, сумма,
    где дата - объект типа данных numpy.datetime64[ns] и сумма целое число или число с плавующей точкой.
    Входящий денежный поток (инвестциии) представлены отрицательными значениями, а исходящий денежный поток (доход)
    представлены положительными значениями
    * guess (optional, default = 0.1): guess начальная точка, с которой начинается численный перебор
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
    s = np.sum(cashflows[:,1])
    if s == 0:
        return 0
    elif s < 0:
        guess *= -1

    try:
        result = op.newton(lambda r: xnpv(r, cashflows), guess, maxiter=100)
    except:
        result = op.brentq(lambda r: xnpv(r, cashflows), -0.999999999999999, 1e20, maxiter=10**6)

    if not isinstance(result, complex):
        return result
    else:
        return None


def twrr(cf):
    """
    Расчет взвешенной по времени доходности, основанной на не равных периодах внесения и изъятия денег в формате DataFrame

    !!!В последней строке положительный cashflow указывать не нужно!!!

    :param
    *cf - pandas dataframe с столбцами:
    index: дата в формате numpy.datetime64
    total: ежедневные значения стоимостей портфеля и отдельных бумаг

    :return:
    *total_return - доходность накопительным итогом за весь период
    *twrr - взвешенная по времени среднегодовая доходность
    *значения взвешенной по времени доходности на каждый период в входном датафрейме
    """
    # Формируем периоды между поступлениями / изъятиями денежных средств
    cf['twrr_interval'] = np.cumsum(cf[cf.columns[1]] != 0) - 1

    # считаем доходность внутри года по периодам в разах
    cf['procent change'] = cf.groupby([cf.index.year, 'twrr_interval']).total.pct_change() + 1
    cf['procent change'].fillna(1, inplace=True)

    # считаем доходность в процентах
    cf['revenue'] = np.cumprod(cf['procent change']) - 1

    # считаем доходность по годам
    cf_year_revenue = cf.groupby([cf.index.year, 'twrr_interval'])[['procent change']].prod().reset_index()

    # расчитываем среднюю годовую взвешенную по времени доходность за весь период владения, используя в качестве количества лет количество дней в датасете/365
    t = int((cf.index[-1] - cf.index[0]).days) / 365.
    twrr = ((cf_year_revenue['procent change'].prod()) ** (1. / t) - 1)

    return [twrr, cf['revenue']]



def create_return(cashflows, method = ['twrr', 'mwrr']):
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
    def xnpv(rate, cashflows):

        t0 = cashflows[0,0]
        if rate <= -1:
            return sum([-abs(cf)/ (-1 - rate) ** (np.timedelta64((t-t0), "D")/(np.timedelta64(1, 'D') * 365)) for t, cf in cashflows])
        if rate == 0:
            return sum(cashflows[:, 1])

        return sum([cf/ (1 + rate) ** (np.timedelta64((t-t0), "D")/(np.timedelta64(1, 'D') * 365)) for t, cf in cashflows])

    def xirr(cashflows, guess=0.1):

        if all(v >= 0 for v in cashflows[:,1]):
            return float("inf")
        if all(v <= 0 for v in cashflows[:,1]):
            return -float("inf")

        result = None

        try:
            result = op.newton(lambda r: xnpv(r, cashflows), tol=1E-4, x0=guess)
        except:
            result = op.brentq(lambda r: xnpv(r, cashflows), -0.999999999999999, 1e20, maxiter=10**6)

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
        twrr, data  = twrr(cashflows)
        result['twrr'] = twrr
        cashflows['twrr'] = data

    if 'mwrr' in method:
        #переводим DataFrame в numpy для скорости выполнения оптимизации
        cf_np = cashflows[cashflows.columns[1]].reset_index().to_numpy()

        arr_res = []
        # для каждого значения стоимости портфеля, считаем mwrr
        for i in range(1, len(cf_np)):
            arr_cf = cf_np[:i+1].copy()
            #прибавляем положительный итоговый денежный поток на дату, равный стоимости портфеля
            arr_cf[i, 1] += cashflows[cashflows.columns[0]][i]
            arr_res += [xirr(arr_cf[np.abs(arr_cf[:, 1]) > 1e-10])]


        arr_res = [1] + arr_res
        cashflows['mwrr'] = arr_res

        result['mwrr'] = arr_res[-1]

    result['data'] = cashflows

    return result


