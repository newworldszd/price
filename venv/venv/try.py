import numpy as np
from scipy.stats import levy_stable
import numpy.random as npr
import pandas as pd
from scipy import stats
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pylab
import yfinance as yf
import xlrd
import math
import dx
import datetime as dt
a = pd.read_excel("C:/Users/DELL/Desktop/1.xlsx")  # 文件路径
s =a["strike"]
option = a["option"]
fitted = a["fitted"]
contrast = a["contrast"]
implied = a["implied"]
volatility =a["volatility"]

print(s)
print(type(s))
fig = plt.figure()
ax = fig.add_subplot()
ax.set_title("B-S fitted option price")
plt.xlabel('strike')
plt.ylabel('option price')
plt.ylim(ymin = 0)
plt.ylim(ymax = 3)
t1 = ax.scatter(s,option,c="b",marker = "s")
t2 = ax.scatter(s,contrast,c="r",marker = "o")
line = plt.vlines(6.65, 0, 3, colors = "g", linestyles = "dashed")
plt.legend(handles=[t1,t2,line],labels=['market option price','B-S fitted option price','initial stock price'],loc='best')

plt.show()
yf.pdr_override()
df = yf.download('GOLD', start='2016-07-01', end='2020-07-18',progress=False)
df.dropna(axis=0,how='any')
close = df['Close'].tolist()
close1 = np.zeros(len(close))
close1 = close1.tolist()
for i in range(len(close)):
    close1[i] = close[i] + np.random.randn() * 0.1
date=df.index.to_frame().reset_index(drop=True)

plt.xlabel('Date')
plt.ylabel('stock price')
plt.plot(date,close,'r-')
plt.plot(date,close1,'b-')
plt.show()
print(close)
date = ['one']
c = []
c.append(close)
test=pd.DataFrame(data=c)
test.to_csv('C:/Users/DELL/PycharmProjects/price/DataA3')
c = []
c.append(close1)
test=pd.DataFrame(data=c)
test.to_csv('C:/Users/DELL/PycharmProjects/price/DataA4')