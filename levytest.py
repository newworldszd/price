from pystable import *
import pylab
import math
import pandas as pd
import tushare as ts
import yfinance as yf
from scipy import stats
from openstock.stock.client import StockClient
st = StableVar(pn_ST,alpha=1.8,beta=0.8,mu = 0,sigma=1)

# Draw a histogram:
x = rstable(10000,st)
pylab.hist(x,bins=1000,range=(-1,1)) # heavy tails make the cutoff essential
pylab.show()

# My First Stable Process Simulation
pylab.cla()
N = 100000
dt = 1/float(N)
t = pylab.linspace(dt,1,N)
x = map(lambda y: pow(dt, 1/st.params['alpha'])*y, rstable(len(t),st))
x = list(x)
for i in range(len(x)-1):
    x[i+1] = x[i]+x[i+1]

pylab.plot(t,x)
pylab.show()


