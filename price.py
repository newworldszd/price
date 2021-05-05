import yfinance as yf
import volatility as vl
import tushare as ts
import math
from random import sample
import pandas as pd
from numpy import *
import statsmodels.api as sm
pro = ts.pro_api('a50655bb53005d74491b03a49590a920ff4558fdd30bd192ed106917')
data = pro.us_basic()
data1 = data['ts_code'].tolist()
data2 = sample(data1,1)
yf.pdr_override()
AAOI = yf.Ticker("AIV")
aaoi = yf.download("AIV", start='2019-07-01', end='2021-03-12', progress=False)
aaoi.dropna(axis=0, how='any')
result = AAOI.options
print(result)
opt = AAOI.option_chain('2021-06-18')
print(opt)
print(type(opt))
data_final = pd.DataFrame(opt.calls)
print(aaoi)
S = 8.85
K = data_final["strike"][0]
T = 0.25
r = 0.0025
close = aaoi['Close'].tolist()
for i in range(len(close) - 1):
    close[i] = math.log(close[i + 1]) - math.log(close[i])
close.pop()
print(close)
vol = std(close) * sqrt(365)
print(vol)
price = vl.bs_call(S, K, T, r, vol)
price1 = data_final["lastPrice"]
print(data_final)
print(price1)
implied_vol = vl.find_vol(price1[0], S, K, T, r)
print(price)
print(implied_vol)