import logging

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np

from hybrid_control.algebra import extract_adjacency
from hybrid_control.plotting.utils import plot_phases
from hybrid_control.environments.library import (
    get_three_region_env,
    get_linearly_seperated_env,
)

logging.basicConfig(level=logging.INFO)


env = get_three_region_env(0, 0, 5)


Ws = np.block([[linear.w] for linear in env.linear_systems])
bs = np.block([linear.b for linear in env.linear_systems])


A = extract_adjacency(Ws, bs)
print(A)


# Make data.
X = np.arange(-10, 10, 0.25)
Y = np.arange(-10, 10, 0.25)
X, Y = np.meshgrid(X, Y)
print(X[0][0])
# Z = np.sin(R)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# Plot the surface.
Z1 = Ws[0][0] * X + Ws[0][1] * Y + bs[0]
surf = ax.plot_surface(X, Y, Z1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

Z2 = Ws[1][0] * X + Ws[1][1] * Y + bs[1]
surf = ax.plot_surface(X, Y, Z2, cmap=cm.coolwarm, linewidth=0, antialiased=False)

Z3 = Ws[2][0] * X + Ws[2][1] * Y + bs[2]
surf = ax.plot_surface(X, Y, Z3, cmap=cm.coolwarm, linewidth=0, antialiased=False)


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

all = np.stack([Z1, Z2, Z3])
Z = np.max(all, axis=0)
# print(Z.shape)
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, Z1 - Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-10, 10)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter("{x:.02f}")

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
