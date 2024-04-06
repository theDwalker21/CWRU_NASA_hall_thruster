#%% Import Libraries
import numpy as np
import matplotlib.pyplot as plt

from oem import OrbitEphemerisMessage
import pandas as pd

#import jplephem
#from jplephem.spk import SPK

#import datetime as dt
"""
from astropy.time import Time

#from matplotlib import pyplot as plt

from poliastro.bodies import Earth, Mars, Jupiter, Sun
from poliastro.frames import Planes
from poliastro.plotting import StaticOrbitPlotter
from poliastro.twobody import Orbit




import matplotlib.pyplot as plt
from astropy import units as u
from poliastro.maneuver import Maneuver
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
ss_i = Orbit.circular(Earth, alt=700 * u.km)
hoh = Maneuver.hohmann(ss_i, 36000 * u.km)
from poliastro.plotting.static import StaticOrbitPlotter
op = StaticOrbitPlotter()
ss_a,ss_f = ss_i.apply_maneuver(hoh, intermediate=True)
op.plot(ss_i)
op.plot(ss_a)
op.plot(ss_f)
plt.show()

#epoch = Time("2018-08-17 12:05:50", scale="tdb")

#plotter = StaticOrbitPlotter(plane=Planes.EARTH_ECLIPTIC)
#plotter.plot_body_orbit(Earth, epoch, label="Earth")
#plotter.plot_body_orbit(Mars, epoch, label="Mars")
#plotter.plot_body_orbit(Jupiter, epoch, label="Jupiter")
#"""

#%% Define Functions

def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def parse_ephem_oem(path):
    # Read File
    ephem_data = OrbitEphemerisMessage.open(path)
    # Parse Data
    t_list = []
    x_list = []
    y_list = []
    z_list = []
    xv_list = []
    yv_list = []
    zv_list = []
    for segment in ephem_data:
        for state in segment:
            t_list.append(state.epoch)
            x_list.append(state.position[0])
            y_list.append(state.position[1])
            z_list.append(state.position[2])
            xv_list.append(state.position[3])
            yv_list.append(state.position[4])
            zv_list.append(state.position[5])
    # Store and Return
    output = (t_list, x_list, y_list, z_list, xv_list, yv_list, zv_list)
    return output


def parse_ephem_e(path):
    t_list = []
    x_list = []
    y_list = []
    z_list = []
    xv_list = []
    yv_list = []
    zv_list = []
    ephem_data = open(path, "r")
    do_while = True
    i_end = 5000
    i = 0
    while do_while:
        try:
            ephem_line = ephem_data.readline().split()
            if i == i_end:
                do_while = False
            elif i == 3:
                i_end = int(ephem_line[-1])
            elif i >= 18:
                t_list.append(float(ephem_line[0]))
                x_list.append(float(ephem_line[1]))
                y_list.append(float(ephem_line[2]))
                z_list.append(float(ephem_line[3]))
                xv_list.append(float(ephem_line[4]))
                yv_list.append(float(ephem_line[5]))
                zv_list.append(float(ephem_line[6]))
            #else:
            #    print(i, ephem_line)
        except:
            #print('an error occurred')
            continue
        i += 1
    output = (t_list, x_list, y_list, z_list, xv_list, yv_list, zv_list)
    return output


def split_ephemeris(t, x, y, z, start, stop):
    try:
        t_good = []
        x_good = []
        y_good = []
        z_good = []
        do_while = True
        i = 0
        while do_while:
            #print(i)
            if start < t[i] < stop:
                t_good.append(t[i])
                x_good.append(x[i])
                y_good.append(y[i])
                z_good.append(z[i])
            i += 1
            if i > len(t) or t[i] > stop:
                do_while = False
        if len(t_good) > 0:
            output = (t_good, x_good, y_good, z_good)
        else:
            print('No values in range!')
            output = (t, x, y, z)
    except:
        print('Date was not found!')
        print(start, stop)
        output = (t, x, y, z)
    return output


def pos_sphere_to_cart_arr(R, T, P):
    X = np.zeros_like(R)
    Y = np.zeros_like(R)
    Z = np.zeros_like(R)
    for row in range(len(R)):
        for col in range(len(R)):
            r = R[row, col]
            t = 0.5*np.pi - T[row, col]
            p = P[row, col]
            x = r*np.sin(t)*np.cos(p)
            y = r*np.sin(t)*np.sin(p)
            z = r*np.cos(t)
            X[row, col] = x
            Y[row, col] = y
            Z[row, col] = z

    output = (X, Y, Z)
    return output

def pos_sphere_to_cart(r, T, p):
    try:
        len(T)
        t = [0]*len(T)
        for i, val in enumerate(T):
            t[i] = 0.5*np.pi - val  # Change of definition of 0 angle
    except:
        t = 0.5*np.pi - T  # Change of definition of 0 angle
    x = r*np.sin(t)*np.cos(p)
    y = r*np.sin(t)*np.sin(p)
    z = r*np.cos(t)
    output = (x, y, z)
    return output



def find_diff(t0, x0, y0, z0, t1, x1, y1, z1):
    diff_list = []
    t_list = []
    dx2_list = []
    dy2_list = []
    dz2_list = []
    #do_while = True
    # while do_while:
    #     if len(x0) == len(x1):
    #         do_while = False
    #     elif len(x0) > len(x1):
    #         x0 = x0[0:-1]
    #         y0 = y0[0:-1]
    #         z0 = z0[0:-1]
    #         print('cut first list short', x0[-1], y0[-1], z0[-1])
    #     elif len(x0) < len(x1):
    #         x1 = x1[0:-1]
    #         y1 = y1[0:-1]
    #         z1 = z1[0:-1]
    #         print('cut second list short', x1[-1], y1[-1], z1[-1])
    #     else:
    #         print('something weird happened!')
    #         return NaN
    print(len(t0))
    #i = 0
    i0 = 0
    i1 = 0
    do_while = True
    do_compare = True
    #for i in range(len(x0)):
    while do_while:
        #try:
        if i0 < 4366:
            do_compare = True
            i0 += 1
            i1 += 1
        elif 7520 < i0 < 7540:  # < (t1[i1] == 80758415.51982) or (t1[i1] == 80852745.93701):
            do_compare = False
            i0 += 1
            i1 += 1
        elif (abs(t0[i0]-t0[i0+1]) < 2300) or (abs(t1[i1]-t1[i1+1]) < 2300):
            print(i0)
            do_compare = False
            i0 += 1
            i1 += 1
        elif abs(t0[i0] - t1[i1]) < 7010:
            #print(i0)
            do_compare = True
            i0 += 1
            i1 += 1
        elif t0[i0] > t1[i1]:
            print('greater')
            i1 += 1
            do_compare = False
        elif t0[i0] < t1[i1]:
            print('less')
            i0 += 1
            do_compare = False
        else:
            print('Something BAD happened!')
            do_compare = False
            #break

        if do_compare:
            dx2 = abs(x1[i1]-x0[i0])**2
            dy2 = abs(y1[i1]-y0[i0])**2
            dz2 = abs(z1[i1]-z0[i0])**2
            r = (dx2 + dy2 + dz2)**.5  # Numpy linalg norm?
            diff_list.append(r)
            t_list.append((t0[i0]+t1[i1])/2)  # modify this? average?
            dx2_list.append(dx2)
            dy2_list.append(dy2)
            dz2_list.append(dz2)

        #except:
        #    print(i0, i1, 'exception occured!')
        #    i0 += 1
        #    i1 += 2
        #    continue
        #if r > 1.25*10**7:
        #    print(i, x1[i])

        if i0 >= len(t0)-1 or i1 >= len(t1)-1:
            do_while = False


    output = (diff_list, t_list, dx2_list, dy2_list, dz2_list)
    return output


def check_disc(l, num):
    disc_list = [0]
    for i in range(len(l)):
        if i != 0:
            diff = abs(l[i-1]-l[i])
            disc_list.append(diff)
    index_list = sorted(range(len(disc_list)), key=lambda i: disc_list[i])[-num:]
    val_list = []
    for i in index_list:
        val_list.append(disc_list[i])
    output = (val_list, index_list)
    return output

#%% Import and Data

# Inputs
#file_path = r"C:\Users\David\GMAT\gmat-win-R2022a\GMAT\output\TestEphemeris.e"
file_path = r"C:\Users\David\Desktop\Semester10\FPI_Lab\EPHEMERIS\vary_nominal.e"
file_path_earth = r"C:\Users\David\Desktop\Semester10\FPI_Lab\EPHEMERIS\vary_nominal_earth.e"

file_path_angle = r"C:\Users\David\Desktop\Semester10\FPI_Lab\EPHEMERIS\vary_angle.e"
file_path_max = r"C:\Users\David\Desktop\Semester10\FPI_Lab\EPHEMERIS\vary_max.e"
file_path_angle_max = r"C:\Users\David\Desktop\Semester10\FPI_Lab\EPHEMERIS\vary_angle_max.e"

file_path_oem = r"C:\Users\David\Desktop\Semester10\FPI_Lab\EPHEMERIS\vary_nominal.oem"
file_path_ang_max_oem = r"C:\Users\David\Desktop\Semester10\FPI_Lab\EPHEMERIS\vary_nominal.oem"

ee_start = 0
dec_start = 5219940
acc_start = 61379940
coast_start = 80834280
coast_end = 111828100

#%% Call Functions

# Nominal Trajectories
T, X, Y, Z, XV, YV, ZV = parse_ephem_e(file_path)  # Parse data
Te, Xe, Ye, Ze, XeV, YeV, ZeV = parse_ephem_e(file_path_earth)  # Parse data - Earth Frame
T_eee, X_eee, Y_eee, Z_eee = split_ephemeris(Te, Xe, Ye, Ze, ee_start, dec_start)  # Earth Exit - Earth Frame
T_ee, X_ee, Y_ee, Z_ee = split_ephemeris(T, X, Y, Z, ee_start, dec_start)  # Earth Exit
TV_ee, XV_ee, YV_ee, ZV_ee = split_ephemeris(T, XV, YV, ZV, ee_start, dec_start)  # Earth Exit Velocity
T_dec, X_dec, Y_dec, Z_dec = split_ephemeris(T, X, Y, Z, dec_start, acc_start)  # Declination
TV_dec, XV_dec, YV_dec, ZV_dec = split_ephemeris(T, XV, YV, ZV, dec_start, acc_start)  # Declination Velocity
T_acc, X_acc, Y_acc, Z_acc = split_ephemeris(T, X, Y, Z, acc_start, coast_start)  # Acceleration
TV_acc, XV_acc, YV_acc, ZV_acc = split_ephemeris(T, XV, YV, ZV, acc_start, coast_start)  # Acceleration Velocity
T_coast, X_coast, Y_coast, Z_coast = split_ephemeris(T, X, Y, Z, coast_start, coast_end)  # Coast
TV_coast, XV_coast, YV_coast, ZV_coast = split_ephemeris(T, XV, YV, ZV, coast_start, coast_end)  # Coast Velocity

#Toem, Xoem, Yoem, Zoem, XVoem, YVoem, ZVoem = parse_ephem_oem(file_path_oem)

# Angled Trajectories
Ta, Xa, Ya, Za, XaV, YaV, ZaV = parse_ephem_e(file_path_angle)  # Parse data
Ta_ee, Xa_ee, Ya_ee, Za_ee = split_ephemeris(Ta, Xa, Ya, Za, ee_start, dec_start)  # Earth Exit
Ta_dec, Xa_dec, Ya_dec, Za_dec = split_ephemeris(Ta, Xa, Ya, Za, dec_start, acc_start)  # Declination
Ta_acc, Xa_acc, Ya_acc, Za_acc = split_ephemeris(Ta, Xa, Ya, Za, acc_start, coast_start)  # Acceleration
Ta_coast, Xa_coast, Ya_coast, Za_coast = split_ephemeris(Ta, Xa, Ya, Za, coast_start, coast_end)  # Coast

# Mag Trajectories
Tm, Xm, Ym, Zm, XmV, YmV, ZmV = parse_ephem_e(file_path_max)  # Parse data
Tm_ee, Xm_ee, Ym_ee, Zm_ee = split_ephemeris(Tm, Xm, Ym, Zm, ee_start, dec_start)  # Earth Exit
Tm_dec, Xm_dec, Ym_dec, Zm_dec = split_ephemeris(Tm, Xm, Ym, Zm, dec_start, acc_start)  # Declination
Tm_acc, Xm_acc, Ym_acc, Zm_acc = split_ephemeris(Tm, Xm, Ym, Zm, acc_start, coast_start)  # Acceleration
Tm_coast, Xm_coast, Ym_coast, Zm_coast = split_ephemeris(Tm, Xm, Ym, Zm, coast_start, coast_end)  # Coast

# Worst Trajectories
Tw, Xw, Yw, Zw, XwV, YwV, ZwV = parse_ephem_e(file_path_angle_max)  # Parse data
Tw_ee, Xw_ee, Yw_ee, Zw_ee = split_ephemeris(Tw, Xw, Yw, Zw, ee_start, dec_start)  # Earth Exit
TwV_ee, XwV_ee, YwV_ee, ZwV_ee = split_ephemeris(Tw, XwV, YwV, ZwV, ee_start, dec_start)  # Earth Exit Velocity
Tw_dec, Xw_dec, Yw_dec, Zw_dec = split_ephemeris(Tw, Xw, Yw, Zw, dec_start, acc_start)  # Declination
TwV_dec, XwV_dec, YwV_dec, ZwV_dec = split_ephemeris(Tw, XwV, YwV, ZwV, dec_start, acc_start)  # Declination Velocity
Tw_acc, Xw_acc, Yw_acc, Zw_acc = split_ephemeris(Tw, Xw, Yw, Zw, acc_start, coast_start)  # Acceleration
TwV_acc, XwV_acc, YwV_acc, ZwV_acc = split_ephemeris(Tw, XwV, YwV, ZwV, acc_start, coast_start)  # Acceleration Velocity
Tw_coast, Xw_coast, Yw_coast, Zw_coast = split_ephemeris(Tw, Xw, Yw, Zw, coast_start, coast_end)  # Coast
TwV_coast, XwV_coast, YwV_coast, ZwV_coast = split_ephemeris(Tw, XwV, YwV, ZwV, coast_start, coast_end)  # Coast Velocity

#Twoem, Xwoem, Ywoem, Zwoem, XwVoem, YwVoem, ZwVoem = parse_ephem_oem(file_path)

# Dw_ee = find_diff(X_ee, Y_ee, Z_ee, Xw_ee, Yw_ee, Zw_ee)  # Position Diff
# Dw_dec = find_diff(X_dec, Y_dec, Z_dec, Xw_dec, Yw_dec, Zw_dec)  # Position Diff
# Dw_acc = find_diff(X_acc, Y_acc, Z_acc, Xw_acc, Yw_acc, Zw_acc)  # Position Diff
# Dw_coast = find_diff(X_coast, Y_coast, Z_coast, Xw_coast, Yw_coast, Zw_coast)  # Position Diff
# DwV_ee = find_diff(XV_ee, YV_ee, ZV_ee, XwV_ee, YwV_ee, ZwV_ee)  # Velocity Diff
# DwV_dec = find_diff(XV_dec, YV_dec, ZV_dec, XwV_dec, YwV_dec, ZwV_dec)  # Velocity Diff
# DwV_acc = find_diff(XV_acc, YV_acc, ZV_acc, XwV_acc, YwV_acc, ZwV_acc)  # Velocity Diff
# DwV_coast = find_diff(XV_coast, YV_coast, ZV_coast, XwV_coast, YwV_coast, ZwV_coast)  # Velocity Diff

# Dw = find_diff(X[7524:7530], Y[7524:7530], Z[7524:7530], Xw[7524:7530], Yw[7524:7530], Zw[7524:7530])
# DwV = find_diff(XV[7524:7530], YV[7524:7530], ZV[7524:7530], XwV[7524:7530], YwV[7524:7530], ZwV[7524:7530])

# Check for discontinuities
# disc_acc, disc_acc_i = check_disc(Dw_acc, 3)
# disc_accV, disc_accV_i = check_disc(DwV_acc, 3)

#check1 = find_diff(Dw[], [], [], [], [], [])
#check2 = find_diff([], [], [], [], [], [])



# Create Ecliptic Planes to plot
earth_surf_rad = 5*10**5
sun_surf_rad = 2*10**8

# Earth
plane_rE = np.linspace(0, earth_surf_rad)
plane_tE = np.linspace(0, 2*np.pi)
plane_RE, plane_PE = np.meshgrid(plane_rE, plane_tE)
plane_TE = np.zeros_like(plane_RE)
plane_XE, plane_YE, plane_ZE = pos_sphere_to_cart_arr(plane_RE, plane_TE, plane_PE)

# Sun 0
plane_rS = np.linspace(0, sun_surf_rad)
plane_tS = np.linspace(0, 2*np.pi)
plane_RS, plane_PS = np.meshgrid(plane_rS, plane_tS)
plane_TS = np.zeros_like(plane_RS)
plane_XS, plane_YS, plane_ZS = pos_sphere_to_cart_arr(plane_RS, plane_TS, plane_PS)

# Sun 1
plane_rS1 = np.linspace(0, sun_surf_rad*5)
plane_tS1 = np.linspace(0, 2*np.pi)
plane_RS1, plane_PS1 = np.meshgrid(plane_rS1, plane_tS1)
plane_TS1 = np.zeros_like(plane_RS1)
plane_XS1, plane_YS1, plane_ZS1 = pos_sphere_to_cart_arr(plane_RS1, plane_TS1, plane_PS1)


# Create orbits of planets to plot
# Moon
moon_rad = 385000
moon_rad_list = [moon_rad]*1001
moon_ang_list = np.linspace(0, 2*np.pi, 1001)
moon_zero_list = [0]*1001

moon_x = np.array(moon_rad_list)*np.cos(moon_ang_list)
moon_y = np.array(moon_rad_list)*np.sin(moon_ang_list)
moon_z = [0]*1001

#moon_x, moon_y, moon_z = pos_sphere_to_cart(moon_rad_list, moon_zero_list, moon_ang_list)

#%% Plot inputs

save_photo = True
figure_path = r'C:\Users\David\Desktop\Semester10\FPI_Lab\FIGURES'

#%% Legacy Plots

#"""
# Plot Total Trajectory
fig0 = plt.figure(figsize=(3.54*1.2, 3.54), dpi=300)
ax0 = fig0.add_subplot(1, 1, 1, projection='3d')
ax0.scatter([0], [0], [0], s=1, c='k', label="Sun")
#ax0.scatter([0], [0], [0], c='y', label="Sun")
ax0.plot(X_ee+X_dec+X_acc, Y_ee+Y_dec+Y_acc, Z_ee+Z_dec+Z_acc, linewidth=1, label="Trajectory")
#ax0.plot(Xw, Yw, Zw, linewidth=1, label="Deviated")
#ax0.plot(X_eee, Y_eee, Z_eee, linewidth=1, label='Earth Exit')
ax0.plot_surface(plane_XS1, plane_YS1, plane_ZS1, alpha=0.3)  # , label='Ecliptic Plane')
#ax0.plot(moon_x, moon_y, moon_zero_list, 'r', linewidth=1, label='Moon')
#ax0.scatter([X_ee[0]], [Y_ee[0]], [Z_ee[0]], s=1, c='g', label="Section Start")
#ax0.scatter([X_ee[-1]], [Y_ee[-1]], [Z_ee[-1]], s=1, c='k', label="Section End")
ax0.set(xlabel="X (km)", ylabel="Y (km)", zlabel="Z (km)")
ax0.legend(bbox_to_anchor=(0, .8), loc='lower left')
set_axes_equal(ax0)
#ax0.set_xticks([-300000, 0, 300000])
#ax0.set_yticks([-300000, 0, 300000])
#ax0.set_zticks([-600000, -300000, 0, 300000])
ax0.view_init(elev=15, azim=-45)

figure_name = r'\GMATentire'
if save_photo:
    plt.savefig(figure_path + figure_name)
#"""

#"""
# Plot variation?
fig01 = plt.figure(figsize=(3.54*1.2, 3.54), dpi=300)
ax01 = fig01.add_subplot(1, 1, 1, projection='3d')
#ax01.scatter([0], [0], [0], s=1, c='k', label="Sun")
#ax0.scatter([0], [0], [0], c='y', label="Sun")
ax01.plot(X_dec[int(len(X_dec)/2):-1], Y_dec[int(len(X_dec)/2):-1], Z_dec[int(len(X_dec)/2):-1], linewidth=1, alpha=.7, label="Nominal")
ax01.plot(Xw_dec[int(len(Xw_dec)/2):-1], Yw_dec[int(len(Xw_dec)/2):-1], Zw_dec[int(len(Xw_dec)/2):-1], linewidth=1, alpha=.7, label="Deviated")
#ax01.plot(X_dec, Y_dec, Z_dec, linewidth=1, alpha=.7, label="Nominal")
#ax01.plot(Xw_dec, Yw_dec, Zw_dec, linewidth=1, alpha=.7, label="Deviated")
#ax0.plot(X_eee, Y_eee, Z_eee, linewidth=1, label='Earth Exit')
#ax01.plot_surface(plane_XS, plane_YS, plane_ZS, alpha=0.3)  # , label='Ecliptic Plane')
#ax0.plot(moon_x, moon_y, moon_zero_list, 'r', linewidth=1, label='Moon')
#ax0.scatter([X_ee[0]], [Y_ee[0]], [Z_ee[0]], s=1, c='g', label="Section Start")
#ax0.scatter([X_ee[-1]], [Y_ee[-1]], [Z_ee[-1]], s=1, c='k', label="Section End")
ax01.set(xlabel="X (km)", ylabel="Y (km)", zlabel="Z (km)")
ax01.legend(bbox_to_anchor=(0, 0), loc='lower left')
#ax01.set_xlim3d([])
#ax01.set_ylim3d([])
#ax01.set_zlim3d([])
#set_axes_equal(ax0)
#ax01.set_xticks([-300000, 0, 300000])
#ax01.set_yticks([-300000, 0, 300000])
#ax01.set_zticks([-600000, -300000, 0, 300000])
ax01.view_init(elev=-10, azim=-45)

figure_name = r'\GMATdeviation'
if save_photo:
    plt.savefig(figure_path + figure_name)
#"""

"""

# Plot Earth Exit
fig1 = plt.figure(figsize=(3.54*1.2, 3.54), dpi=300)
ax1 = fig1.add_subplot(1, 1, 1, projection='3d')
ax1.scatter([0], [0], [0], s=1, c='k', label="Earth")
#ax1.scatter([0], [0], [0], c='y', label="Sun")
#ax.plot(X, Y, Z, linewidth=1, label="All Ephemeris Data")
ax1.plot(X_eee, Y_eee, Z_eee, linewidth=1, label='Earth Exit')
ax1.plot_surface(plane_XE, plane_YE, plane_ZE, alpha=0.3)  # , label='Ecliptic Plane')
ax1.plot(moon_x, moon_y, moon_zero_list, 'r', linewidth=1, label='Moon')
#ax1.scatter([X_ee[0]], [Y_ee[0]], [Z_ee[0]], s=1, c='g', label="Section Start")
#ax1.scatter([X_ee[-1]], [Y_ee[-1]], [Z_ee[-1]], s=1, c='k', label="Section End")
ax1.set(xlabel="X (km)", ylabel="Y (km)", zlabel="Z (km)")
ax1.legend(bbox_to_anchor=(0, .8), loc='lower left')
set_axes_equal(ax1)
ax1.set_xticks([-300000, 0, 300000])
ax1.set_yticks([-300000, 0, 300000])
ax1.set_zticks([-600000, -300000, 0, 300000])
ax1.view_init(elev=20, azim=40)

figure_name = r'\GMATearthExit'
if save_photo:
    plt.savefig(figure_path + figure_name)

#"""

"""

# Plot Declination
fig2 = plt.figure(figsize=(3.54*1.2, 3.54), dpi=300)
ax2 = fig2.add_subplot(1, 1, 1, projection='3d')
ax2.scatter([0], [0], [0], s=1, c='k', label="Sun")

#ax2.plot(Xa_ee+Xa_dec, Ya_ee+Ya_dec, Za_ee+Za_dec, linewidth=1, label='Angled')
#ax2.plot(Xm_ee+Xm_dec, Ym_ee+Ym_dec, Zm_ee+Zm_dec, linewidth=1, label='Max')
#ax2.plot(Xw_ee+Xa_dec, Yw_ee+Yw_dec, Zw_ee+Zw_dec, linewidth=1, label='Worst')

ax2.plot(X_ee, Y_ee, Z_ee, linewidth=1)  # , label='Earth Exit')
ax2.plot_surface(plane_XS, plane_YS, plane_ZS, alpha=0.3)
ax2.plot(X_dec, Y_dec, Z_dec, linewidth=1, label='Declination')

ax2.set(xlabel="X (km)", ylabel="Y (km)", zlabel="Z (km)")
ax2.legend(bbox_to_anchor=(0, .8), loc='lower left')
set_axes_equal(ax2)
ax2.view_init(elev=20, azim=-30)

figure_name = r'\GMATdeclination'
#figure_name = r'\GMATdeclination_ang'
#figure_name = r'\GMATdeclination_mag'
#figure_name = r'\GMATdeclination_worst'
if save_photo:
    plt.savefig(figure_path + figure_name)



# Plot Acceleration
fig3 = plt.figure(figsize=(3.54*1.2, 3.54), dpi=300)
ax3 = fig3.add_subplot(1, 1, 1, projection='3d')
ax3.scatter([0], [0], [0], s=1, c='k', label="Sun")

#ax3.plot(Xa_ee+Xa_dec+Xa_acc, Ya_ee+Ya_dec+Ya_acc, Za_ee+Za_dec+Za_acc, linewidth=1, label='Angled')
#ax3.plot(Xm_ee+Xm_dec+Xm_acc, Ym_ee+Ym_dec+Ym_acc, Zm_ee+Zm_dec+Zm_acc, linewidth=1, label='Max')
#ax3.plot(Xw_ee+Xw_dec+Xw_acc, Yw_ee+Yw_dec+Yw_acc, Zw_ee+Zw_dec+Zw_acc, linewidth=1, label='Worst')

ax3.plot(X_ee, Y_ee, Z_ee, linewidth=1)  # , label='Earth Exit')
ax3.plot_surface(plane_XS, plane_YS, plane_ZS, alpha=0.3)
ax3.plot(X_dec, Y_dec, Z_dec, linewidth=1)  # , label='Declination')
ax3.plot(X_acc, Y_acc, Z_acc, linewidth=1, label='Acceleration')
ax3.set(xlabel="X (km)", ylabel="Y (km)", zlabel="Z (km)")
ax3.legend(bbox_to_anchor=(0, .8), loc='lower left')
set_axes_equal(ax3)
ax3.view_init(elev=20, azim=-30)

figure_name = r'\GMATacceleration'
#figure_name = r'\GMATacceleration_ang'
#figure_name = r'\GMATacceleration_mag'
#figure_name = r'\GMATacceleration_worst'
if save_photo:
    plt.savefig(figure_path + figure_name)

#"""

"""
# Plot 2 Acceleration
fig23 = plt.figure(figsize=(3.54*1.2, 3.54), dpi=300)
ax23 = fig23.add_subplot(1, 1, 1, projection='3d')
#ax23.scatter([0], [0], [0], s=1, c='k', label="Sun")

#ax23.plot(Xa_acc, Ya_acc, Za_acc, linewidth=1, label='Angled')
#ax23.plot(Xm_acc, Ym_acc, Zm_acc, linewidth=1, label='Max')
ax23.plot(Xw_acc, Yw_acc, Zw_acc, linewidth=1, label='Worst - Acc')
ax23.plot(Xw_coast, Yw_coast, Zw_coast, linewidth=1, label='Worst - Coast')

#ax23.plot(X_ee, Y_ee, Z_ee, linewidth=1)  # , label='Earth Exit')
#ax23.plot_surface(plane_XS, plane_YS, plane_ZS, alpha=0.3)
#ax23.plot(X_dec, Y_dec, Z_dec, linewidth=1)  # , label='Declination')
ax23.plot(X_acc, Y_acc, Z_acc, linewidth=1, label='Acceleration')
ax23.plot(X_coast, Y_coast, Z_coast, linewidth=1, label='Coast')
ax23.set(xlabel="X", ylabel="Y", zlabel="Z")
ax23.legend(bbox_to_anchor=(0, .8), loc='lower left')
set_axes_equal(ax23)
ax23.view_init(elev=20, azim=-30)

#figure_name = r'\GMATacceleration'
#figure_name = r'\GMATacceleration_only_ang'
#figure_name = r'\GMATacceleration_only_mag'
figure_name = r'\GMATacceleration_only_worst'
if save_photo:
    plt.savefig(figure_path + figure_name)

#"""

"""

# Plot Coast
fig4 = plt.figure(figsize=(3.54*1.2, 3.54), dpi=300)
ax4 = fig4.add_subplot(1, 1, 1, projection='3d')
ax4.scatter([0], [0], [0], s=1, c='k', label="Sun")
ax4.plot(X_ee, Y_ee, Z_ee, linewidth=1)  # , label='Earth Exit')
ax4.plot_surface(plane_XS, plane_YS, plane_ZS, alpha=0.3)
ax4.plot(X_dec, Y_dec, Z_dec, linewidth=1)  # , label='Declination')
ax4.plot(X_acc, Y_acc, Z_acc, linewidth=1)  # , label='Acceleration')
ax4.plot(X_coast, Y_coast, Z_coast, linewidth=1, label='Coast')
ax4.set(xlabel="X (km)", ylabel="Y (km)", zlabel="Z (km)")
ax4.legend(bbox_to_anchor=(0, .8), loc='lower left')
set_axes_equal(ax4)
ax4.view_init(elev=20, azim=-30)

figure_name = r'\GMATcoast'
#figure_name = r'\GMATcoast_ang'
if save_photo:
    plt.savefig(figure_path + figure_name)

#"""


#%% Difference Plots

#Tw_dec_np = np.array(Tw_dec)/86400
#TwV_dec_np = np.array(TwV_dec)/86400
#Tw_acc_np = np.array(Tw_acc)/86400
#TwV_acc_np = np.array(TwV_acc)/86400
#Tw_coast_np = np.array(Tw_coast)/86400

Dw, Tw_mod, dx, dy, dz = find_diff(T, X, Y, Z, Tw, Xw, Yw, Zw)
DwV, TwV_mod, dxV, dyV, dzV = find_diff(T, XV, YV, ZV, Tw, XwV, YwV, ZwV)

#Dw_oem, Tw_mod_oem, dx_oem, dy_oem, dz_oem = find_diff(Toem, Xoem, Yoem, Zoem, Twoem, Xwoem, Ywoem, Zwoem)
#DwV_oem, TwV_mod_oem, dxV_oem, dyV_oem, dzV_oem = find_diff(Toem, XVoem, YVoem, ZVoem, Twoem, XwVoem, YwVoem, ZwVoem)

#X[7524:7530]


#TwV_coast_np = np.array(TwV_coast)/86400

#
# do_while = True
# while do_while:
#     if len(Tw_acc_np) == len(Dw_acc):
#         do_while = False
#     elif len(Tw_acc_np) > len(Dw_acc):
#         Tw_acc_np = Tw_acc_np[0:-1]
#     elif len(Tw_acc_np) < len(Dw_acc):
#         Dw_acc = Dw_acc[0:-1]
#         DwV_acc = DwV_acc[0:-1]
# do_while = True
# while do_while:
#     if len(Tw_coast_np) == len(Dw_coast):
#         do_while = False
#     elif len(Tw_coast_np) > len(Dw_coast):
#         Tw_coast_np = Tw_coast_np[0:-1]
#     elif len(Tw_coast_np) < len(Dw_coast):
#         Dw_coast = Dw_coast[0:-1]
#         DwV_coast = DwV_coast[0:-1]

"""

figA = plt.figure(figsize=(3.54*1.2, 3.54), dpi=300)
axA = figA.add_subplot(1, 1, 1)
#axA.plot(T_ee, Dw_ee, linewidth=1, label='Earth Exit')
#axA.plot(Tw_dec_np[10:-1], Dw_dec[10:-1], linewidth=1, label='Declination')
#axA.plot(Tw_acc_np, Dw_acc, linewidth=1, label='Acceleration')
#axA.plot(Tw_coast_np, Dw_coast, linewidth=1, label='Coast')
axA.plot(np.array(Tw_mod)/86400, Dw)
#axA.plot(np.array(Tw_mod_oem)/86400, Dw_oem)
axA.set(xlabel='Time (days)', ylabel='Spacecraft Deviation (km)')
#xA.legend()
axA.grid(True)
plt.tight_layout()
figure_name = r'\GMATdiff_pos'
if save_photo:
    plt.savefig(figure_path + figure_name)

figB = plt.figure(figsize=(3.54*1.2, 3.54), dpi=300)
axB = figB.add_subplot(1, 1, 1)
#axB.plot(T_ee, DwV_ee, linewidth=1, label='Earth Exit')
# axB.plot(Tw_dec_np[10:-1], DwV_dec[10:-1], linewidth=1, label='Declination')
# axB.plot(Tw_acc_np, DwV_acc, linewidth=1, label='Acceleration')
# axB.plot(Tw_coast_np, DwV_coast, linewidth=1, label='Coast')
axB.plot(np.array(TwV_mod)/86400, DwV)
#axB.plot(np.array(TwV_mod_oem)/86400, DwV_oem)
axB.set(xlabel='Time (days)', ylabel='Relative Velocity (km/s)')
#axB.set_xlim([680, 720])
#axB.legend()
axB.grid(True)
plt.tight_layout()
figure_name = r'\GMATdiff_vel'
if save_photo:
    plt.savefig(figure_path + figure_name)

#"""

"""
figC = plt.figure(figsize=(3.54*1.2, 3.54), dpi=300)
axC = figC.add_subplot(1, 1, 1)
#axC.scatter(Tw[7524:7530], Dw)
#axC.scatter(Tw[7525:7530], Dw[7525:7530])
#axC.scatter(T[7525:7530], X[7525:7530])
#axC.scatter(T[7525:7530], Y[7525:7530])
#axC.scatter(T[7525:7530], Z[7525:7530])
axC.scatter(T[7510:7540], dx[7510:7540])
axC.scatter(T[7510:7540], dy[7510:7540])
axC.scatter(T[7510:7540], dz[7510:7540])
#axC.scatter(Tw[7525:7530], Yw[7525:7530])
#axC.scatter(Tw[7525:7530], Zw[7525:7530])
#axC.scatter(Tw[7525:7530], Xw[7525:7530])
#axC.scatter(Tw[7525:7530], Yw[7525:7530])
#axC.scatter(Tw[7525:7530], Zw[7525:7530])
#axC.plot(Tw[7524:7528], Dw[7524:7528])
#axC.plot(Tw[7520:7540], Xw[7520:7540])
#axC.plot(Tw[7520:7540], Yw[7520:7540])
#axC.plot(Tw[7520:7540], Zw[7520:7540])
# axC.plot(T, X, label='X')
# axC.plot(T, Y, label='Y')
# axC.plot(T, Z, label='Z')
# axC.plot(Tw, Xw, label='Xw')
# axC.plot(Tw, Yw, label='Yw')
# axC.plot(Tw, Zw, label='Zw')
# axC.plot(T_dec, X_dec, label='X_dec')
# axC.plot(T_dec, Y_dec, label='Y_dec')
# axC.plot(T_dec, Z_dec, label='Z_dec')
# axC.plot(Tw_dec, Xw_dec, label='Xw_dec')
# axC.plot(Tw_dec, Yw_dec, label='Yw_dec')
# axC.plot(Tw_dec, Zw_dec, label='Zw_dec')
# axC.plot(T_acc, X_acc, label='X_acc')
# axC.plot(T_acc, Y_acc, label='Y_acc')
# axC.plot(T_acc, Z_acc, label='Z_acc')
# axC.plot(Tw_acc, Xw_acc, label='Xw_acc')
# axC.plot(Tw_acc, Yw_acc, label='Yw_acc')
# axC.plot(Tw_acc, Zw_acc, label='Zw_acc')
# axC.plot(T_coast, X_coast, label='X_coast')
# axC.plot(T_coast, Y_coast, label='Y_coast')
# axC.plot(T_coast, Z_coast, label='Z_coast')
# axC.plot(Tw_coast, Xw_coast, label='Xw_coast')
# axC.plot(Tw_coast, Yw_coast, label='Yw_coast')
# axC.plot(Tw_coast, Zw_coast, label='Zw_coast')
# axC.set(xlabel='Time', ylabel='Position (km)')
#axC.legend()
axC.grid(True)
plt.tight_layout()
#"""

"""
figD = plt.figure(figsize=(3.54*1.2, 3.54), dpi=300)
axD = figD.add_subplot(1, 1, 1)
#axD.scatter(T[7510:7540], X[7510:7540], s=2)
#axD.scatter(Tw[7510:7540], Xw[7510:7540], s=2)
axD.scatter(T[7510:7540], dx[7510:7540], s=2)
#axD.scatter(Tw[7510:7540], dxV[7510:7540], s=2)
#axD.scatter(T[7540:-1], X[7540:-1], s=2)
#axD.scatter(Tw[7540:-1], Xw[7540:-1], s=2)
#axD.scatter(T[7510:7540], Y[7510:7540])
#axD.scatter(T[7510:7540], Yw[7510:7540])
#axD.scatter(T[7510:7540], Z[7510:7540])
#axD.scatter(T[7510:7540], Zw[7510:7540])
axD.grid(True)
plt.tight_layout()

#"""





#%% Bug fixing

test_df = pd.DataFrame(
    {'Tw_mod': Tw_mod,
     'Dw': Dw,
     'dx': dx,
     'dy': dy,
     'dz': dz
    })

