# %% Import Libraries
import numpy as np
import matplotlib.pyplot as plt


# %% Math Functions

def rk4(y, dydx, n, x, h, nu, itan):
    """
    This function imports variables and runs a 4th order Runge-Kutta method to simultaneously solve the equations.

    :param y:
    :param dydx:
    :param n:
    :param x:
    :param h:
    :param nu:
    :param itan:
    :return:
    """
    # Define variables and lists
    hh = h*0.5
    h6 = h/6.0
    xh = x+hh
    yt = []
    yout = []

    # Step 1 --------------------
    for i in range(n):
        yt.append(y[i]+hh*dydx[i])

    # Step 2 --------------------
    dyt = derivs(xh, yt, nu, itan)
    for i in range(n):
        yt.append(y[i]+hh*dyt[i])

    # Step 3 --------------------
    dym = derivs(xh, yt, nu, itan)
    for i in range(n):
        yt.append(y[i]+h*dym[i])
        dym.append(dyt[i]+dym[i])

    # Step 4 --------------------   CHECK!!
    dyt = derivs(x+h, yt, nu, itan)
    for i in range(n):
        yout.append(y[i]+h6*(dydx[i]+dyt[i]+2.0*dym[i]))

    return yout


def derivs(x, y, nu, itan):
    """
    This function contains the equations used in continuous-thrust orbital mechanics.

    :param x:
    :param y:
    :param nu:
    :param itan:
    :return:
    """
    dydx = [0] * 5
    y.insert(0, 0)

    # circumferential thrust
    if itan == 0:
        dydx[1] = y[2] * y[3] * y[3] - 1.0 / (y[2] * y[2])
        dydx[2] = y[1]
        dydx[3] = -2.0 * y[1] * y[3] / y[2] + nu / y[2]
        dydx[4] = y[3]

    # tangential thrust
    if itan == -1:
        if y[1] > 0:
            phi = np.arctan(y[2] * y[3] / y[1])
        if y[1] < 0:
            phi = np.arctan(y[2] * y[3] / y[3]) + np.pi
        if y[1] == 0:
            phi = np.pi / 2.0
        dydx[1] = y[2] * y[3] * y[3] - 1.0 / (y[2] * y[2]) + nu * np.cos(phi)
        dydx[2] = y[1]
        dydx[3] = -2.0 * y[1] * y[3] / y[2] + nu * np.sin(phi) / y[2]
        dydx[4] = y[3]

    # vectored thrust
    if itan > 0:
        phi = itan * np.pi / 180
        dydx[1] = y[2] * y[3] * y[3] - 1.0 / (y[2] * y[2]) + nu * np.cos(phi)
        dydx[2] = y[1]
        dydx[3] = -2.0 * y[1] * y[3] / y[2] + nu * np.sin(phi) / y[2]
        dydx[4] = y[3]
    dydx.remove(0)  # This is super jank, please fix!!
    y.remove(0)

    return dydx


# %% Practical Functions

def pol_to_cart(radius_list, theta_list):
    """


    :param radius_list: List of radii in any units
    :param theta_list: List of angles in radians
    :return:
    """
    length = len(radius_list)
    x_list = [0] * length
    y_list = [0] * length
    for index in range(length):
        x_list[index] = radius_list[index] * np.cos(theta_list[index])
        y_list[index] = radius_list[index] * np.sin(theta_list[index])
    return x_list, y_list


# %% Inputs

print('Starting Script\n')

# Constants
#r_s = 6378.14  # Surface Radius
#r_0 = 6378.14  # Initial orbit radius

# Thruster constants
itan = 0         # Thrust angle (0=circumferential, -1=tangential)
nu_mean = 0.0124      # T/m in terms of g (thrust)
tau_i = 0        # Initial thrust time (tau) (x1)
#tau_f = 159.8    # Final thrust time (tau) (x2)





#tau_f = 2000
#tau_f = 800
#tau_f = 400
#tau_f = 350
#tau_f = 160
tau_f = 159.8
#tau_f = 50
#tau_f = 10






tau_g = 0        # Non-thrust time (tau) (x3)
h = 0.01         # Time step
iprt = 10        # Integer time step
n = 4            # Number of equations to solve

# RK4 inputs?
rho_d = 0
rho = 1
theta_d = 1
theta = 0        # Arbitrary (for now)

# rho_list = [rho]
# theta_list = [theta]
# tau_list = [tau_i]




# Thruster parameters - OUTPUT FROM S9_distribution2
thruster_int = 1925.912807607399
thruster_std = 0.9474934936528844

#thruster_int = 1924.28
#thruster_std = 1.08225
#thruster_int = 1923.54  # Actual
#thruster_std = 0.28106  # Actual
#thruster_std = 500
#thruster_std = 50

print(' ----- Thruster Stats ----- ')
print('Integral:', thruster_int)
print('Standard Deviation:', thruster_std)
print(' ----- -------------- ----- ')


# Logic based on inputs
# y = [rho_d, rho, theta_d, theta]
Tratio = 2*thruster_std/thruster_int
nu_plus = nu_mean + (Tratio*nu_mean)
nu_minus = nu_mean - (Tratio*nu_mean)

nu_list = [nu_mean, nu_plus, nu_minus]

# For loop here!!!

# %% Thrusting code
rho_nested_list = []
theta_nested_list = []
tau_nested_list = []

for nu in nu_list:

    rho_list = [rho]
    theta_list = [theta]
    tau_list = [tau_i]
    y = [rho_d, rho, theta_d, theta]


    x = tau_i  # Declare time start
    yrow = y  # Defining values
    ii = 0  # define iterations
    nn = 0  # define number of time steps               how is this different from ii?

    # Loop for thrusting time
    while x < tau_f-h:
    #for j in range(1000):
        ii += 1
        dydx = derivs(x, y, nu, itan)  # Get diff. eq's from EOM
        yout = rk4(y, dydx, n, x, h, nu, itan)  # Solve diff. eq's using RK4 method

        # Once time step is reached, update values?
        if ii == iprt:
            yrow = yout  # Update values for equation
            xp = x+h  # not necessary?
            ii = 0  # reset counter

        # Add to time step h
        nn += 1  #
        x = tau_i+h*nn  # update time to represent steps marched so far

        # update all y values to yout values?
        for index in range(n):
            y[index] = yout[index]
        rho_list.append(y[1])
        theta_list.append(y[3])
        tau_list.append(x)
    # END WHILE LOOP 1

    # Output some stuff?

    # %% Non-thrusting code

    # Similar to above while loop

    # Loop for non-thrusting time               same as above code, create function?
    nu = 0
    while x < tau_f+tau_g:
    #for j in range(100):
        ii += 1
        dydx = derivs(x, y, nu, itan)
        yout = rk4(y, dydx, n, x, h, nu, itan)
        if ii == iprt:
            yrow = y
            xp = x+h
            ii = 0

        nn += 1
        x = tau_i+h*nn
        for index in range(n):
            y[index] = yout[index]
        rho_list.append(y[1])
        theta_list.append(y[3])
        tau_list.append(x)

    rho_nested_list.append(rho_list)
    theta_nested_list.append(theta_list)
    tau_nested_list.append(tau_list)
    # END WHILE LOOP 2

# Output some stuff
print('Calculations Complete')


# %% Plotting Parameter calculations


# Create outline for staring body
P1_r = [1]*100  # Radius
P1_th = np.linspace(0, 2*np.pi, 100)  # Angle

# Calculate SOI of body
# Earth:
r_earth = 6371  # km
r_earth_SOI = 0.929 * (10**6)
P1_SOI_r = [r_earth_SOI/r_earth]*100
#P1_SOI_th = np.linspace(0, 2*np.pi, 100)


# Re-dimensionalize radius and time
mu = 398600
# r_earth
t_list_sec = [0]*len(tau_nested_list[0])
t_list_day = [0]*len(tau_nested_list[0])



r_list_mean = [0]*len(rho_nested_list[0])
r_list_plus = [0]*len(rho_nested_list[1])
r_list_minus = [0]*len(rho_nested_list[2])


for index in range(len(tau_nested_list[0])):
    t_list_sec[index] = tau_nested_list[0][index]/((mu/r_earth**3)**.5)
    t_list_day[index] = t_list_sec[index]/86400

for index in range(len(rho_nested_list[0])):
    r_list_mean[index] = rho_nested_list[0][index]*r_earth
for index in range(len(rho_nested_list[1])):
    r_list_plus[index] = rho_nested_list[1][index] * r_earth
for index in range(len(rho_nested_list[2])):
    r_list_minus[index] = rho_nested_list[2][index] * r_earth


# %% Convert coordinates from polar to cartesian

# Move to function?

# Vehicle thrust curves
x_list_mean, y_list_mean = pol_to_cart(rho_nested_list[0], theta_nested_list[0])
x_list_plus, y_list_plus = pol_to_cart(rho_nested_list[1], theta_nested_list[1])
x_list_minus, y_list_minus = pol_to_cart(rho_nested_list[2], theta_nested_list[2])

dist_list = []
a_list = []
b_list = []
i = 0
a2 = 0
for (x_plus, x_minus, y_plus, y_minus) in zip(x_list_plus, x_list_minus, y_list_plus, y_list_minus):
    a = abs((x_plus-x_minus) * r_earth)
    b = abs((y_plus-y_minus) * r_earth)
    c = ((a**2)+(b**2))**.5

    dist_list.append(c)
    i += 1
    a2 = abs((x_plus-x_minus) * r_earth)

# Planet 1
P1_x, P1_y = pol_to_cart(P1_r, P1_th)
P1_SOI_x, P1_SOI_y = pol_to_cart(P1_SOI_r, P1_th)

# a = abs(x_list_plus[-1] - x_list_minus[-1])
# b = abs(y_list_plus[-1] - y_list_minus[-1])
# c = ((a**2)+(b**2))**.5
print('Max range between endpoints:', dist_list[-1]/r_earth, 'Earth radii')
print('Plotting...\n')

"""
What comes out of this script?

x_list, y_list



"""

# %% Plotting
# Actually plot values
#fig1, (ax1, ax2) = plt.subplots(1,2)
#fig1.set_figheight(9)
#fig1.set_figwidth(15)

fig1 = plt.figure(figsize=(10,6))
ax1 = plt.subplot(121)



ax1.plot(P1_x, P1_y, 'k', label='Planet 1 Surface')


#ax1.plot(P1_SOI_x, P1_SOI_y, 'k:', label='Planet 1 SOI')


#ax1.fill_between(x_list, y_list, label='95% Confidence Interval')
ax1.plot(x_list_plus, y_list_plus, 'r--', label='Plus')
ax1.plot(x_list_minus, y_list_minus, 'b--', label='Minus')
#ax1.plot(x_list_mean, y_list_mean, '--', label='Vehicle Trajectory')

ax1.set(title='Continuous Thrust Mission Profile', xlabel='X-axis (Earth radii)', ylabel='Y-axis (Earth radii)')
ax1.legend()
ax1.grid(True)
ax1.axis('equal')


ax2 = plt.subplot(122)
ax2.plot(t_list_day, dist_list)
ax2.set(title='Distance Between Bounds', xlabel='Time (days)', ylabel='Distance Between Bounds (km)')
ax2.grid(True)


# %%

fig1.tight_layout()


plt.savefig('cartesian1_160')


plt.show()
