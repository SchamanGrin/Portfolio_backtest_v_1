import time
import pandas as pd
import numpy as np
import scipy.optimize as op
from performance import twrr, xirr, xirr_1


#guess=0.1

def xnpv_np(rate, cashflows):


    t0 = cashflows[0,1]
    #global guess
    #guess = rate
    return np.sum(cashflows[:,0]/ (1 + rate) ** ((np.sum(np.diff(cashflows[:,1]))/ np.timedelta64(1, 'D'))/ 365.0))
    #return sum(cf/ (1 + rate) ** ((t - t0).days / 365.0) for cf,t in cashflows)

def xirr_np(cashflows, guess=0.1):

    return op.newton(lambda r: xnpv_np(r, cashflows), guess)


data = pd.read_csv(
    'symbol/SPY.csv', header=0, index_col=0,
                names=['timestamp', 'open', 'low', 'high', 'close', 'volume'], parse_dates=['timestamp']
)
#start_date = pd.to_datetime('2010-01-01')
#data.reindex(pd.to_datetime(data.index, '%Y%m%d'))
start_date = np.datetime64('2020-07-22')
data.reindex(np.array(data.index, dtype='datetime64'))


data_cashflow = pd.DataFrame(data.loc[data.index >= start_date]['close'][::-1])
data_cashflow.loc[:,'cashflow'] = 0.*len(data_cashflow.index)
data_cashflow['cashflow'][0]=-data_cashflow['close'][0]
data_cashflow.rename(columns={'close':'total'}, inplace=True)

'''
#twrr_total, twrr, twrr_data = twrr(data_cashflow)


#data_cashflow.loc[data_cashflow.index[-1], 'cashflow'] = data_cashflow['total'][-1]
#print(data_cashflow)
#xirr = xirr_1(data_cashflow['cashflow'].loc[np.abs(data_cashflow['cashflow']) > 1E-10])*100

def func_dfxirr(dfxirr, x, i):
    dfxirr.loc[i] += x
    return dfxirr[np.abs(dfxirr) > 1E-10]

time_0_0 = time.time()
df_xirr_copy = data_cashflow['cashflow'].copy()
data_xirr_0 = [xirr_1(func_dfxirr(df_xirr_copy[:i].copy(), data_cashflow['total'][i], i)) \
               for i in data_cashflow.index[1:]]
print(f'{len(data_xirr_0)} {time.time() - time_0_0:.2f} сек.')


#print(xirr)



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
arr_xirr = []

'''
time_1 = time.time()
arr_xirr = []
for i in range(1,len(arr_cashflow)):
    arr_cf = arr_cashflow[:i+1].copy()
    arr_cf[i, 0] += data_cashflow['total'].iloc[i]
    arr_xirr += [xirr([(d,x) for x,d in arr_cf[np.abs(arr_cf[:, 0]) > 1E-10]])]



print(f'{time.time() - time_1:.2f} сек')
'''

time_np = time.time()
arr_xirr_np = []
for i in range(1,len(arr_cashflow)):
    arr_cf = arr_cashflow[:i+1].copy()
    arr_cf[i, 0] += arr_total[i]
    #arr_xirr_np += [xirr_np(arr_cf[np.abs(arr_cf[:, 0]) > 1E-10])]
    arr_xirr_np += [xirr([(d,x) for x,d in arr_cf[np.abs(arr_cf[:, 0]) > 1E-10]])]

print(f'numpy {len(arr_xirr_np)} {time.time() - time_1:.2f} сек')


t_v = time.time()
arr_xirr_v = []
def cf(i):
    arr_cf_f = arr_cashflow[:i+1].copy()
    arr_cf_f[i, 0] += arr_total[i]
    t = xirr([(d, x) for x, d in arr_cf_f[np.abs(arr_cf_f[:, 0]) > 1E-10]])
    return t#xirr_np(arr_cf_f[np.abs(arr_cf_f[:, 0]) > 1E-10])


f = np.vectorize(cf, otypes=[np.float64])
res = np.array(f(range(1,len(arr_cashflow))))
res_0 = res[0]
arr_xirr_v = np.append(arr_xirr_v, np.array(f(range(1,len(arr_cashflow)))))
print(f'вектор {len(arr_xirr_v)} {time.time() - t_v:.2f} сек.')

print(np.allclose(arr_xirr_np, arr_xirr_v))



'''


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
