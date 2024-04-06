# %% Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import time

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


def unit_conv(input, conv):
    output = [0]*len(input)
    for i in range(len(input)):
        output[i] = input[i] * conv
    return output

def unit_conv2(input1, input2, conv):
    output1 = unit_conv(input1, conv)
    output2 = unit_conv(input2, conv)
    output = (output1, output2)
    return output

def unit_conv4(input1, input2, input3, input4, conv):
    output1 = unit_conv(input1, conv)
    output2 = unit_conv(input2, conv)
    output3 = unit_conv(input3, conv)
    output4 = unit_conv(input4, conv)
    output = (output1, output2, output3, output4)
    return output

# %% Inputs


print('Starting Script\n')

# Constants
#r_s = 6378.14  # Surface Radius
#r_0 = 6378.14  # Initial orbit radius


# Thruster constants
thruster_int = 1925.912807607399  # Thruster parameters - OUTPUT FROM S9_distribution2
thruster_std = 0.9474934936528844  # Thruster parameters - OUTPUT FROM S9_distribution2
#thruster_int = 1924.28
#thruster_std = 1.08225
#thruster_int = 1923.54  # Actual
#thruster_std = 0.28106  # Actual
#thruster_std = 500
#thruster_std = 50

km_in_AU = 149597870.7

r_oort_in_AU = 2000
r_oort_out_AU = 200000
r_origin = 695700  # [km] # SUN
mu_P1 = 132712000000  # SUN

total_time_years = 1
total_time_secs = total_time_years*31536000
tau_f = total_time_secs * ((mu_P1/r_origin**3)**.5)



# Orbital Transfer Calculations
mass_sun = 1.99847*10**30  # [kg]
mass_AC = (1.0788 + 0.9092 + 0.1221) * mass_sun  # [kg]
mu_mass_ratio = mu_P1/mass_sun
mu_P2 = mass_AC * mu_mass_ratio

total_distance_ly = 4.396  # ly
total_distance_AU = total_distance_ly * 63240
mass_ratio = mass_sun / (mass_sun+mass_AC)
transition_point_AU = mass_ratio * total_distance_AU

#tau_f = 2000
#tau_f = 800
#tau_f = 400
#tau_f = 350
#tau_f = 160
#tau_f = 159.8
#tau_f = 50
#tau_f = 10
# r_origin = 6371  # [km] EARTH
# mu = 398600  # EARTH


itan = 0         # Thrust angle (0=circumferential, -1=tangential)
nu_mean = 0.0124      # T/m in terms of g (thrust)
tau_i = 0        # Initial thrust time (tau) (x1)
#tau_f = 159.8   # Final thrust time (tau) (x2)

tau_g = 0        # Non-thrust time (tau) (x3)
h = 0.01         # Time step
iprt = 10        # Integer time step
n = 4            # Number of equations to solve

# RK4 inputs
rho_d = 0
rho = 1          # Starting position
theta_d = 1
theta = 0        # Arbitrary (for now)

# %% Logic based on inputs - Initialization
print(' ----- Thruster Stats ----- ')
print('Integral:', thruster_int)
print('Standard Deviation:', thruster_std)
print(' -------------------------- ')
print()
print('Starting Calculations...')
start = time.time()

Tratio = 2*thruster_std/thruster_int
nu_plus = nu_mean + (Tratio*nu_mean)
nu_minus = nu_mean - (Tratio*nu_mean)
nu_list = [nu_mean, nu_plus, nu_minus]

# %% Thrusting code

# Initialize
rho_nested_list = []
theta_nested_list = []
tau_nested_list = []

tot_x_len = int(np.ceil(tau_f/h))
#progress_interval = (tot_x_len / 33.33)  # - 1
complete_perc = 100/len(nu_list)
num_complete = 0
counter = 0

#print('Progress: 0%')

# For loop runs RK4 to find orbital path
for nu in nu_list:


    num_save_total = 1000-1
    num_save_counter = 0
    num_save_index = 0

    #rho_list = [0]*int(np.floor(tot_x_len/num_save_total))
    #theta_list = [0]*int(np.floor(tot_x_len/num_save_total))
    #tau_list = [0]*int(np.floor(tot_x_len/num_save_total))

    rho_list = np.zeros(int(np.floor(tot_x_len/num_save_total)))
    theta_list = np.zeros_like(rho_list)
    tau_list = np.zeros_like(rho_list)
    rho_list[0] = rho
    theta_list[0] = theta
    tau_list[0] = tau_i

    y = [rho_d, rho, theta_d, theta]
    x = tau_i  # Declare time start
    yrow = y  # Defining values
    ii = 0  # define iterations
    nn = 0  # define number of time steps               how is this different from ii?

    # Loop for thrusting time
    while x < tau_f-h:
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

        #counter += 1
        percent = ((x/tau_f)*100)/3 + num_complete*complete_perc

        #if counter >= progress_interval:
        if percent >= counter:
            end = time.time()
            #print('Progress:', np.round(percent, decimals=2), '%   Runtime:',
            #      np.round((end-start)/60, decimals=2), 'min')
            print('Progress: '+str(counter)+'%  ---  Runtime:', np.round((end-start)/60, decimals=2), 'min')
            counter += 1

        # update all y values to yout values?
        for index in range(n):
            y[index] = yout[index]

        num_save_counter += 1
        if num_save_counter >= num_save_total:
            rho_list[num_save_index] = y[1]
            theta_list[num_save_index] = y[3]
            tau_list[num_save_index] = x

            num_save_index += 1
            num_save_counter = 0

        # rho_list[nn] = y[1]
        # theta_list[nn] = y[3]
        # tau_list[nn] = x
        #rho_list.append(y[1])
        #theta_list.append(y[3])
        #tau_list.append(x)

    # Output some stuff?

    # %% Non-thrusting code - Similar to above while loop

    # Loop for non-thrusting time               same as above code, create function?
    nu = 0
    while x < tau_f+tau_g-h:
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
        rho_list[nn] = y[1]
        theta_list[nn] = y[3]
        tau_list[nn] = x

    rho_nested_list.append(rho_list)
    theta_nested_list.append(theta_list)
    tau_nested_list.append(tau_list)

    num_complete += 1

# Output some stuff
end = time.time()
print('Progress: 100%  ---  Runtime:', np.round((end-start)/60, decimals=2), 'min')

#print('Calculations Complete')

# %% Plotting Parameter calculations


# Create outline for origin body
P1_r = [r_origin]*100  # Radius
P1_th = np.linspace(0, 2*np.pi, 100)  # Angle

oort_ri = [r_oort_in_AU]*100
oort_ro = [r_oort_out_AU]*100
P1_r_SOI = [transition_point_AU]*100

# Re-dimensionalize radius and time

t_list_sec = unit_conv(tau_nested_list[0], 1/((mu_P1/r_origin**3)**.5))
t_list_year = unit_conv(t_list_sec, 1/31536000)

r_list_mean = unit_conv(rho_nested_list[0], r_origin)
r_list_plus = unit_conv(rho_nested_list[1], r_origin)
r_list_minus = unit_conv(rho_nested_list[2], r_origin)


# %% Convert coordinates from polar to cartesian, convert units to AU and years

# Vehicle thrust curves
x_list_mean_radii, y_list_mean_radii = pol_to_cart(rho_nested_list[0], theta_nested_list[0])
x_list_plus_radii, y_list_plus_radii = pol_to_cart(rho_nested_list[1], theta_nested_list[1])
x_list_minus_radii, y_list_minus_radii = pol_to_cart(rho_nested_list[2], theta_nested_list[2])

dist_list = []
a_list = []
b_list = []
i = 0
a2 = 0
for (x_plus, x_minus, y_plus, y_minus) in zip(x_list_plus_radii, x_list_minus_radii, y_list_plus_radii, y_list_minus_radii):
    a = abs((x_plus-x_minus) * r_origin)
    b = abs((y_plus-y_minus) * r_origin)
    c = ((a**2)+(b**2))**.5

    dist_list.append(c)
    i += 1
    a2 = abs((x_plus-x_minus) * r_origin)

# Planet 1
P1_x_km, P1_y_km = pol_to_cart(P1_r, P1_th)
oort_xi_AU, oort_yi_AU = pol_to_cart(oort_ri, P1_th)
oort_xo_AU, oort_yo_AU = pol_to_cart(oort_ro, P1_th)
P1_x_SOI, P1_y_SOI = pol_to_cart(P1_r_SOI, P1_th)

P1_x_radii, P1_y_radii = unit_conv2(P1_x_km, P1_y_km, 1/r_origin)
P1_x_AU, P1_y_AU = unit_conv2(P1_x_km, P1_y_km, 1/km_in_AU)

dist_list_km = dist_list.copy()
dist_list_radii = unit_conv(dist_list_km, 1/r_origin)
dist_list_AU = unit_conv(dist_list_km, 1/km_in_AU)

# Convert radii to km
x_list_plus_km, y_list_plus_km, x_list_minus_km, y_list_minus_km = unit_conv4(
    x_list_plus_radii, y_list_plus_radii, x_list_minus_radii, y_list_minus_radii,  # Items being converted
    r_origin)  # Conversion Factor
# Convert km to AU
x_list_plus_AU, y_list_plus_AU, x_list_minus_AU, y_list_minus_AU = unit_conv4(
    x_list_plus_km, y_list_plus_km, x_list_minus_km, y_list_minus_km,  # Items being converted
    1/km_in_AU)  # Conversion Factor

r_list_mean_km = unit_conv(rho_nested_list[0], r_origin)
r_list_mean_AU = unit_conv(r_list_mean_km, 1/km_in_AU)

end = time.time()
print('\nCalculations Complete! Total Runtime:', np.round((end - start)/60, decimals=2), 'min')
print()
year_string = r'Year' if total_time_years <= 1 else r'Years'
print('Total Mission Time:         ', np.round(total_time_years, decimals=2), year_string)
print('Total (mean) disp travelled:', np.round(r_list_mean_AU[-1], decimals=2), 'AU')
print('Max range between endpoints:', np.round(dist_list_AU[-1], decimals=2), 'AU')
# If transition point reached, say what angle it was reached at
print('')

print()


print('Plotting...\n')


# %% Plotting - Actually plot values

# Orbital
fig1 = plt.figure(figsize=(12, 12))
ax1 = plt.subplot(111)
# ax1 = plt.subplot(121)

#ax1.plot(P1_x_AU, P1_y_AU, 'k', label='Origin Surface')  # Plot Surface of Origin
ax1.plot(x_list_plus_AU, y_list_plus_AU, 'r--', label='Plus')
ax1.plot(x_list_minus_AU, y_list_minus_AU, 'b--', label='Minus')
#ax1.plot(oort_xi_AU, oort_yi_AU, 'k.', label='Inner Oort Cloud Boundary')  # , markersize=1)
ax1.plot(oort_xo_AU, oort_yo_AU, 'k.', label='Outer Oort Cloud Boundary')  # , markersize=1)
ax1.plot(P1_x_SOI, P1_y_SOI, 'g--', label='"Sphere of Influence"')
# ax1.plot(x_list_mean, y_list_mean, '--', label='Vehicle Trajectory')

ax1.set(title=str('Total Time = '+str(total_time_years)+' '+year_string), xlabel='X-axis (AU)', ylabel='Y-axis (AU)')
ax1.legend()
ax1.grid(True)
ax1.axis('equal')

# # Distance
# fig2 = plt.figure(figsize=(8, 8))
# ax2 = plt.subplot(111)
# ax2.plot(t_list_month, dist_list_AU)
# #ax2.plot(t_list_day, dist_list_radii)
# #ax2.plot(t_list_day, dist_list)
# ax2.set(title='Distance Between Bounds', xlabel='Time (months)', ylabel='Distance Between Bounds (AU)')
# #ax2.set(title='Distance Between Bounds', xlabel='Time (days)', ylabel='Distance Between Bounds (origin radii)')
# # ax2.set(title='Distance Between Bounds', xlabel='Time (days)', ylabel='Distance Between Bounds (km)')
# ax2.grid(True)
# # ax2.axis('equal')

# %% Final Plotting

fig1.tight_layout()
# plt.savefig('cartesian1_160')
plt.show()

print('Done!')
