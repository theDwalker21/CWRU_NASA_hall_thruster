# %% Import
import numpy as np
import matplotlib.pyplot as plt



# %% Main Code

# Prepare arrays x, y, z
"""
theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
z = np.linspace(-2, 2, 100)
r = z**2 + 1
x = r * np.sin(theta)
y = r * np.cos(theta)
"""

# Define
mu = 132712000000  # gravitational parameter
r0 = [-149.6 * 10 ** 6, 0.0, 0.0]
v0 = [29.0, -5.0, 0.0]
dt = np.linspace(0.0, 86400 * 700, 5000)  # time is seconds

x = x2
y = y2
z = z2
x2 = -mu / np.sqrt(x ** 2 + y ** 2 + z ** 2) * x
y2 = -mu / np.sqrt(x ** 2 + y ** 2 + z ** 2) * y
z2 = -mu / np.sqrt(x ** 2 + y ** 2 + z ** 2) * z

# %% Plotting

ax1 = plt.figure().add_subplot(projection='3d')
ax1.plot(x, y, z, label='parametric curve')
ax1.legend()


plt.show()


