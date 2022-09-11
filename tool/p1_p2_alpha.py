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


def term1(x, y):
    z1 = Entropy((1/2) * (x + y))
    return z1


def term2(x, y, a):
    h1 = Entropy(x)
    h2 = Entropy(y)
    z2 = (1 / 2) * (h1 + h2)
    return (1-a) * z2


x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)

X, Y = np.meshgrid(x, y)
a = 0

Z1 = term1(X, Y)
Z2 = term2(X, Y, a)
Z_total = Z1-Z2
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z_total, 1000, cmap='viridis')
ax.set_xlabel('p1')
ax.set_ylabel('p2')
ax.set_zlabel('loss')
plt.title(f'alpha={a}')
plt.show()

