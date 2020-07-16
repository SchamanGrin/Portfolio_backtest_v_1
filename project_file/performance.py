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

    chron_order = sorted(cashflows, key=lambda x: x[0])
    t0 = chron_order[0][0]  # t0 is the date of the first cash flow

    return sum([cf / (1 + rate) ** ((t - t0).days / 365.0) for (t, cf) in chron_order])


def xirr(cashflows, guess=0.1):
    """
    Calculate the Internal Rate of Return of a series of cashflows at irregular intervals.
    Расчет внутренней нормы доходности при нерегулярных денежных потоках

    Arguments
    Аргументы
    ---------
    * cashflows: a list object in which each element is a tuple of the form (date, amount), where date is a python datetime.date object and amount is an integer or floating point number. Cash outflows (investments) are represented with negative amounts, and cash inflows (returns) are positive amounts.
    * денежный поток, список объектов, где каждый элемент кортеж вида (дата, сумма), где дата - объект типа данных datetime.date и сумма целое число или число с правубщей точкой. Входящий денежный поток (инвестциии) представлены отрицательными значениями, а исходящий денежный поток (доход) представлены положительными значениями
    * guess (optional, default = 0.1): a guess at the solution to be used as a starting point for the numerical solution.
    Returns
    --------
    * Returns the IRR as a single value

    Notes
    ----------------
    * The Internal Rate of Return (IRR) is the discount rate at which the Net Present Value (NPV) of a series of cash flows is equal to zero. The NPV of the series of cash flows is determined using the xnpv function in this module. The discount rate at which NPV equals zero is found using the secant method of numerical solution.
    * This function is equivalent to the Microsoft Excel function of the same name.
    * For users that do not have the scipy module installed, there is an alternate version (commented out) that uses the secant_method function defined in the module rather than the scipy.optimize module's numerical solver. Both use the same method of calculation so there should be no difference in performance, but the secant_method function does not fail gracefully in cases where there is no solution, so the scipy.optimize.newton version is preferred.
    """

    return op.newton(lambda r: xnpv(r, cashflows), guess)

def twrr(all_holdings):
    """
    Расчет взвешенной по времени доходности, основанной на не равных периодах
    :param all_holdings: ежедневные значения стоимостей ортфеля и отдельных бумаг
    :return: единственное значение  взвешенной по времени доходности портфеля
    """

    #Проверить на костыли!!!
    df = pd.DataFrame(all_holdings)
    df.set_index('datetime', inplace=True)
    df['twrr_interwal'] = [0]*len(df.index)

    # если значение в поле cash стало больше предыдущего, значит были добавлены день, значит надо делаем следующий период для расчета взвещеной по времени доходности
    for row in df.index[1:]:
        df['twrr_interwal'][row] = df.iloc[df.index.get_loc(row)-1]['twrr_interwal'] + ((df['cash'][row] - df.iloc[df.index.get_loc(row)-1]['cash']) > 0)


    print(df)

    return df