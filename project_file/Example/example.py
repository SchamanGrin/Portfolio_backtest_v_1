import time
import numpy as np
import pandas as pd
import scipy.optimize as op
from performance import twrr, xirr, xirr_1, xnpv

'''
def xnpv_np(rate, cashflows):

    t0 = cashflows[0,1]
    delta = np.timedelta64(1, 'D') * 365
    t = 0
    t_list = []
    if rate <= -1:
        return -1
    if rate == 0:
        return sum(cashflows[:, 0])

    for i in range(len(cashflows)):
        st = np.timedelta64((cashflows[i, 1] - t0), "D") / delta
        temp = cashflows[i, 0] / (1 + rate) ** st
        t += temp
        t_list += [temp]
    #t = np.sum([cf/ (1 + rate) ** (np.timedelta64((t-t0), "D")/delta) for cf, t in cashflows])
    return t
'''

def xnpv(rate, cashflows):

    t0 = cashflows[0,1]
    if rate <= -1:
        return -1
    if rate == 0:
        return sum(cashflows[:, 0])

    return np.sum([cf/ (1 + rate) ** (np.timedelta64((t-t0), "D")/(np.timedelta64(1, 'D') * 365)) for cf, t in cashflows])

def xirr(cashflows, guess=0.1):

    s = np.sum(cashflows[:,0])
    if s == 0:
        return 0
    elif s < 0:
        guess *= -1

    return op.newton(lambda r: xnpv(r, cashflows), tol=1E-4, x0=guess)

def xirr_np_bounds(cashflows, guess=0.1):
    # при отрицательном s r не может бытыть меньше -1, нужно ввести ограничения.

    s = np.sum(cashflows[:,0])

    if s == 0:
        return 0
    elif s < 0:
        guess *= -1

    try:
        r_op = op.newton(lambda r: xnpv_np(r, cashflows), tol=1E-4, x0=guess)
        if isinstance(r_op, float):
            pass
            # return r_op
        else:
            r_op = r_op[0]
        bounds = None
        if s < 0:
            if r_op > 0:
                bounds = op.Bounds(-1.0, 0.0)
        else:
            if r_op < 0:
                bounds = op.Bounds(0.0, np.inf)
        if bounds:
            r_op = op.minimize(lambda r: xnpv_np(r, cashflows), x0=guess, tol=1E-3, bounds=bounds, method="trust-constr")
            r_op = r_op.x[0]
        return r_op
    except:
        if s < 0:
            bounds = op.Bounds(-1.0, 0.0)
        else:
            bounds = op.Bounds(0.0, np.inf)
        r_op = op.minimize(lambda r: xnpv_np(r, cashflows), x0=guess, tol=1E-5, bounds=bounds, method="trust-constr")
        return r_op.x[0]



data = pd.read_csv(
    'symbol/SPY.csv', header=0, index_col=0,
                names=['timestamp', 'open', 'low', 'high', 'close', 'volume'], parse_dates=['timestamp']
)
#start_date = pd.to_datetime('2010-01-01')
#data.reindex(pd.to_datetime(data.index, '%Y%m%d'))
start_date = np.datetime64('2010-01-01')
data.reindex(np.array(data.index, dtype='datetime64'))


data_cashflow = pd.DataFrame(data.loc[data.index >= start_date]['close'][::-1])
data_cashflow.loc[:,'cashflow'] = 0.*len(data_cashflow.index)
data_cashflow['cashflow'][0]=-data_cashflow['close'][0]
data_cashflow.rename(columns={'close':'total'}, inplace=True)

'''
#twrr_total, twrr, twrr_data = twrr(data_cashflow)


time_1 = time.time()
data_xirr_1 = []
df_xirr_copy = data_cashflow['cashflow'].copy()
df_x0 = df_xirr_copy[:1]
for i in data_cashflow.index[1:]:
    df_x0 = df_x0.append(df_xirr_copy[df_xirr_copy.index == i])
    df_xirr = df_x0.copy()
    df_xirr[-1] += data_cashflow['total'][i]
    data_xirr_1 += [xirr_1(df_xirr.loc[np.abs(df_xirr) > 1E-10])]

print(data_xirr_1)
print(f'pandas {len(data_xirr_1)} {time.time() - time_1:.2f} сек.')

#print(sum(data_xirr_0[i] - data_xirr_1[i] for i in range(len(data_xirr_0))))
'''
time_1 = time.time()
df_t = data_cashflow.copy()
df_t['date'] = data_cashflow.index
df_t['date'] = df_t['date'].astype('datetime64[ns]')
arr_cashflow = df_t[['cashflow', 'date']].to_numpy()
arr_total = df_t['total'].to_numpy()


'''
arr_xirr = []
time_1 = time.time()
arr_xirr = []
for i in range(1,len(arr_cashflow)):
    arr_cf = arr_cashflow[:i+1].copy()
    arr_cf[i, 0] += data_cashflow['total'].iloc[i]
    arr_xirr += [xirr([(d,x) for x,d in arr_cf[np.abs(arr_cf[:, 0]) > 1E-10]])]



print(f'вектор с изначальной xirr {time.time() - time_1:.2f} сек')
'''


time_np = time.time()
arr_xirr_np = []
arr_xirr = []
for i in range(1,len(arr_cashflow)):
    arr_cf = arr_cashflow[:i+1].copy()
    arr_cf[i, 0] += arr_total[i]
    arr_xirr_np += [xirr_np_bounds(arr_cf[np.abs(arr_cf[:, 0]) > 1E-10])]
    arr_xirr += [xirr_np(arr_cf[np.abs(arr_cf[:, 0]) > 1E-10])]

#print(f'numpy {len(arr_xirr_np)} {time.time() - time_1:.2f} сек')
#df_t['xirr'] = np.concatenate([[0],arr_xirr_np])
#print(df_t)
print(np.allclose(arr_xirr_np, arr_xirr))



'''
t_v = time.time()
arr_xirr_v = []
def cf(i):
    arr_cf_f = arr_cashflow[:i+1].copy()
    arr_cf_f[i, 0] += arr_total[i]
    #t = xirr([(d, x) for x, d in arr_cf_f[np.abs(arr_cf_f[:, 0]) > 1E-10]])
    t = xirr_np(arr_cf_f[np.abs(arr_cf_f[:, 0]) > 1E-10])
    return t #xirr_np(arr_cf_f[np.abs(arr_cf_f[:, 0]) > 1E-10])


f = np.vectorize(cf, otypes=[np.float64])
res = np.array(f(range(1,len(arr_cashflow))))


print(f'вектор {len(arr_xirr_v)} {time.time() - t_v:.2f} сек.')

#print(np.allclose(arr_xirr, arr_xirr_v))






time_f = time.time()



def xirrdata(x):
    i = len(x)
    xd = x.copy()
    if i>1:
        xd[i - 1] += data_cashflow['total'][i-1]
        t = xirr_np(xd[np.abs(xd)>1E-10])
    else:
        t = 1.
    return t

data_cashflow['temp'] = data_cashflow['cashflow'].expanding().apply(xirrdata)

print(f'{time.time() - time_f:.2f} сек.')
xirr_f = data_cashflow['temp'].to_list()
print(np.allclose(arr_xirr, xirr_f[1:]))
'''
