import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
t_0 = 0 # define model parameters
t_end = 4
length = 1000
theta = 1.1
mu = 0.8
sigma = 0.3
t = np.linspace(t_0,t_end,length) # define time axis
dt = np.mean(np.diff(t))
y = np.zeros(length)
y0 = np.random.normal(loc=0.0,scale=1.0) # initial condition
drift = lambda y,t: theta*(mu-y) # define drift term, google to learn about lambda
diffusion = lambda y,t: sigma # define diffusion term
noise = np.random.normal(loc=0.0,scale=1.0,size=length)*np.sqrt(dt) #define noise process
# solve SDE
for i in range(1,length):
 y[i] = y[i-1] + drift(y[i-1],i*dt)*dt + diffusion(y[i-1],i*dt)*noise[i]

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

"""
animation example 2
author: Kiterun
"""

fig, ax = plt.subplots()
x = np.linspace(0, 2*np.pi, 200)
y = np.sin(x)
l = ax.plot(x, y)
dot, = ax.plot([], [], 'ro')

def init():
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-1, 1)
    return l

def gen_dot():
    for i in np.linspace(0, 2*np.pi, 200):
        newdot = [i, np.sin(i)]
        yield newdot

def update_dot(newd):
    dot.set_data(newd[0], newd[1])
    return dot,

ani = animation.FuncAnimation(fig, update_dot, frames = gen_dot, interval = 100, init_func=init)
ani.save('sin_dot.gif', writer='imagemagick', fps=30)
plt.show()
