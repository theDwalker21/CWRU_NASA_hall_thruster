# %% Import Libraries
from numpy import radians
from orbital import earth, KeplerianElements, Maneuver, plot, plot3d

from scipy.constants import kilo
import matplotlib.pyplot as plt



# %% Main Code
# kilo = 1000
"""
orbit = KeplerianElements.with_altitude(1000 * kilo, body=earth)
man = Maneuver.hohmann_transfer_to_altitude(10000 * kilo)
plot(orbit, title='Maneuver 1', maneuver=man)
"""

from orbital import earth_sidereal_day
molniya = KeplerianElements.with_period(
    earth_sidereal_day / 2, e=0.741, i=radians(63.4), arg_pe=radians(270),
    body=earth)

# Simple circular orbit
orbit = KeplerianElements.with_altitude(1000 * kilo, body=earth)




man = Maneuver.hohmann_transfer_to_altitude(10000 * kilo)

# %% Plotting

# Simple Plot
plot(molniya)

# Animation
plot(molniya, title='Molniya 1', animate=True)
#plot3d(molniya1, title='Molniya 2', animate=True)

# Maneuvers
plot(orbit, title='Maneuver 1', maneuver=man)
plot3d(orbit, title='Maneuver 2', maneuver=man)

plt.show()

