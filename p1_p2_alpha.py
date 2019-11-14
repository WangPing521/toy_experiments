from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
import torch


def Entropy(p):
    H = 0
    sec_term = np.log(p)
    minor_e = p * sec_term
    H = (-1) * minor_e
    return H


def f(x, y, a):
    z1 = Entropy((1/2) * (x + y))
    h1 = Entropy(x)
    h2 = Entropy(y)
    z2 = (1 / 2) * (h1 + h2)
    return z1 - a * z2


x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)

X, Y = np.meshgrid(x, y)

a = 0.5
Z = f(X, Y, a)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 1000, cmap='binary')
ax.set_xlabel('p1')
ax.set_ylabel('p2')
ax.set_zlabel('loss')
plt.title('alpha=0.5')
plt.show()

