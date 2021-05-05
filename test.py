from scipy.stats import levy_stable
import levy as levy
import pylab
import math
from random import sample
import numpy as np
import pandas as pd
import tushare as ts
import numpy.random as npr
from scipy import stats
import statsmodels.api as sm
from scipy.stats import ks_2samp
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
frame=[]
pro = ts.pro_api('a50655bb53005d74491b03a49590a920ff4558fdd30bd192ed106917')
data = pro.us_basic()
data1 = data['ts_code'].tolist()
data2 = ['BTC-USD']
print(len(data2))
print(data2)
for g in data2:
    yf.pdr_override()
    df = yf.download(g, start='2016-07-01', end='2020-07-18',progress=False)
    df.dropna(axis=0,how='any')
    if len(df) < 100:
        continue
    close = df['Close'].tolist()
    print(close)
    close1 = df['Close'].tolist()
    for i in range(len(close)-1 ):
        close[i] = math.log(close[i +1]) - math.log(close[i])
    close.pop()
    for i in range(len(close1) - 1):
        close1[i] = math.log(close1[i+1])
    close1.pop()
    result = stats.describe(close, bias=False)

    pylab.hist(close,color = 'r', bins=1000, range=(-0.3, 0.3))
    pylab.xlabel(g+"股票收益率直方图", fontproperties='SimHei')# heavy tails make the cutoff essential
    pylab.savefig(g+"股票收益率直方图.jpg")
    pylab.close()
    close2 = sm.add_constant(close1)
    regr = sm.OLS(close, close1)
    res = regr.fit()
    a = res.params[0].item()
    residual = np.zeros(len(close)).tolist()
    for i in range(len(close)):
        residual[i] = close[i] - a * close1[i]
    residual = np.array(residual)
    alpha = levy.fit_levy(residual, alpha=None, beta=None, location=None, scale=None)
    k = np.zeros(len(close)).tolist()
    for i in range(len(close)):
        for j in range(i):
            k[i - 1] = k[i - 1] + close1[j - 1]
    above = 0
    below = 0
    for i in range(len(close1)):
        above = above + pow(math.exp((i - 1) * 0.01), 2) * (close1[i - 1] - close1[0] - a * i) * k[i - 1]
        below = below + pow(math.exp((i - 1) * 0.01), 2) * pow(k[i - 1], 2)
    theta = above / below
    a1, b1, loc1, scale1 = alpha[0], alpha[1], alpha[2], alpha[3]
    r = levy_stable.rvs(a1, -b1, loc1, scale1, size=len(close))
    for i in range(len(close)):
        r[i -1] = r[i - 1] + theta * close1[i - 1]
    c = ks_2samp(r, close)
    d = stats.ttest_ind(r, close)
    sns.set_palette("hls")  # 设置所有图的颜色，使用hls色彩空间
    sns.kdeplot(r,cut=0.3,bw=.2)
    pylab.hist(r, bins=1000, range=(-0.3, 0.3),color = 'b')  # heavy tails make the cutoff essential
    pylab.xlabel(g + "拟合收益率直方图", fontproperties='SimHei')
    pylab.savefig(g + "拟合收益率直方图.jpg")
    pylab.close()
    c1 = np.mean(close)
    d1 = np.std(close)
    r2 = npr.randn(len(close))
    for i in range(len(close)):
        r2[i] = c1 + r2[i] * d1
    pylab.hist(r2, bins=1000, range=(-0.3, 0.3),color = 'r')  # heavy tails make the cutoff essential
    pylab.xlabel(g + "布朗拟合收益率直方图", fontproperties='SimHei')
    pylab.savefig(g + "布朗拟合收益率直方图.jpg")
    pylab.close()
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.xlim(xmin=-0.25)
    plt.xlim(xmax=0.15)
    plt.xlabel('sample input')
    plt.ylabel('theoretical quantiles')

    ax.scatter(np.sort(r),np.sort(close),c = "r",marker="o")
    ax.plot([-0.2,0.1],[-0.2,0.1],c = "b")
    plt.show()
    r = r.tolist()
    r.append(close)
    print(r)
    frame.append((r))
data_final = pd.DataFrame(frame)
name=['one']
test=pd.DataFrame(columns=name,data=r)
test.to_csv('C:/Users/DELL/PycharmProjects/price/DataA2',encoding='gbk')