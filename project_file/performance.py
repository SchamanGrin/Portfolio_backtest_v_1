# performance.py

import numpy as np
import pandas as pd

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

    t0 = cashflows[0, 0]
    if rate <= -1:
        return -1
    if rate == 0:
        return sum(cashflows[:, 1])

    return np.sum([cf / (1 + rate) ** (np.timedelta64((t - t0), "D") / (np.timedelta64(1, 'D') * 365)) for t, cf in cashflows])


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
        return op.newton(lambda r: xnpv(r, cashflows), guess)
    except:
        return op.minimize(lambda r: xnpv(r, cashflows), x0=guess, tol=1E-5, bounds=op.Bounds(-1.0, 0.0),
                    method="trust-constr").x[0]

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



def create_return(cashflows, method = ['twrr', 'mwrr'], period = 'day'):
    """
    Метод расчитывает доходность портфеля с учетом пополнений и ихъятий денежных средств портфеля.
    Расчет производится на основе DataFrame с данными о стоимости портфеля и денежного потока
    на дату. Для расчета используется два метода:
    - взвещенный по деньгам (money-weighted return)
    - взвещенный по времени (time-weighted return)


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

    :param period: периодичность данных в DataFrame, определеяет, какие промежутки времени   может принимать значения:
    'minute', 'hour', 'day', 'month', 'year'


    :return: словарь {'twrr';{'return': float, 'data': DataFrame}, 'mwrr':{'return': float, 'data': DataFrame}}
     return - значение годовой доходности посчитанной соответствующим методом в формате float
     data - массив значений доходности в каждый момент времени
    """
    def xnpv(rate, cashflows):

        t0 = cashflows[0,0]
        if rate <= -1:
            return -1
        if rate == 0:
            return sum(cashflows[:, 1])

        return np.sum([cf/ (1 + rate) ** (np.timedelta64((t-t0), "D")/(np.timedelta64(1, 'D') * 365)) for t, cf in cashflows])

    def xirr(cashflows, guess=0.1):

        s = np.sum(cashflows[:, 1])
        if s == 0:
            return 0
        elif s < 0:
            guess *= -1
        try:
            return op.newton(lambda r: xnpv(r, cashflows), tol=1E-4, x0=guess)
        except:
            op.minimize(lambda r: xnpv(r, cashflows), x0=guess, tol=1E-5, bounds=op.Bounds(-1.0, 0.0),
                        method="trust-constr").x[0]

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
        #Формируем периоды между поступлениями / изъятиями денежных средств
        cf['twrr_interval'] = np.cumsum(cf[cf.columns[1]] != 0) - 1


        #считаем доходность внутри года по периодам в разах
        cf['procent change'] = cf.groupby([cf.index.year, 'twrr_interval']).total.pct_change() + 1
        cf['procent change'].fillna(1, inplace = True)

        #считаем доходность в процентах
        cf['revenue'] = np.cumprod(cf['procent change'])-1

        #считаем доходность по годам
        cf_year_revenue = cf.groupby([cf.index.year, 'twrr_interval'])[['procent change']].prod().reset_index()

        #расчитываем среднюю годовую взвешенную по времени доходность за весь период владения, используя в качестве количества лет количество дней в датасете/365
        t = int((cf.index[-1]-cf.index[0]).days)/365.
        twrr = ((cf_year_revenue['procent change'].prod()) ** (1. / t) - 1)

        return [twrr, cf['revenue']]

    result = {}
    if 'twrr' in method:
        twrr, data  = twrr(cashflows)
        result['twrr'] = {'return': twrr, 'data':data}

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

        result['mwrr'] = {'return': arr_res[-1], 'data': pd.DataFrame(arr_res, index=arr_res[0], columns=['revenue'])}

    return result