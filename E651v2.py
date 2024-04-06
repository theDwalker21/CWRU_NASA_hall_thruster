# %% Import Libraries
import numpy as np
import matplotlib.pyplot as plt

"""
import matplotlib as mpl
import requests
import json
from zipfile import ZipFile
from io import BytesIO
import os
from time import time
import xml.etree.ElementTree as E
import datetime
import epoch
import csv
from operator import itemgetter
"""
import pandas as pd

"""
from sqlalchemy import create_engine
# import ipywidgets as widgets
import IPython
import warnings
import xlsxwriter
import openpyxl
import string
"""
# import scipy
from scipy import optimize
from scipy import stats as st
from scipy.stats import variation
import scipy.special

# from lmfit import minimize, Parameters, Parameter, report_fit
# from mpl_toolkits import mplot3d
from matplotlib import gridspec

# %% Import Functions

from E651_func_v1 import _1gauss,_1lorentz,_1voigt,_2gauss,_2lorentz,_2voigt,_2voigt_2cen,_1gauss_skew,_2gauss_skew,\
    _1lorentz_skew,_2lorentz_skew,_1voigt_skew,_2voigt_skew,_1voigt_skew_cen,_2voigt_skew_cen,_2voigt_skew_cen_mod1,\
    _2voigt_skew_cen_mod2,\
    combine_func


# %% INPUTS

# File path for local folder containing Data
file_path1 = 'C:\\Users\\David\\Desktop\\FPI Lab\\Data_csv\\'

# File name for headers
file_name_head = 'TVP File Template.xlsx'

# Sheet Name to read (per Excel file)
file_sheet_head = 'Sheet1'

# Limit Sweep Angle
limit_sweep_bool = True

# Limits for Sweep Angle
limit_sweep = 22

# Remove Center
center_remove_2D_bool = True

# Max Radius
center_remove_2D_mag = 4

# Choose plots
plot_2D = True
plot_3D = False

# Choose to do fit based on 3D data?
fit_3d = False

# Choose to auto-find center
find_center = False

# Choose number of steps to find center (each)
center_steps = 100
if not find_center:
    center_steps = 1

# Choose to do experimental limiting
test_code = False

# ---------------------- SET LABELS FOR AXES HERE? ----------------------

# Add # of file to look at?

# Add Type of fit to perform?

# Collector for 2D
col_2D = 12

# Files to actually read
file_names_data = [
    '190612 Radial scans20190612_131606_TDU3COGraph-0001-RFC5-600V-21A_1000_TVP.csv',
    '190612 Radial scans20190612_131804_TDU3COGraph-0001-RFC5-600V-21A_1000_TVP.csv',
    '190612 Radial scans20190612_132024_TDU3COGraph-0001-RFC5-600V-21A_1000_TVP.csv',
    '190612 Radial scans20190612_132211_TDU3COGraph-0001-RFC5-600V-21A_1000_TVP.csv',
    '190612 Radial scans20190612_132925_TDU3COGraph-0001-RFC5-600V-21A_1000_TVP.csv',
    '190612 Radial scans20190612_133123_TDU3COGraph-0001-RFC5-600V-21A_1000_TVP.csv',
    '190612 Radial scans20190612_134206_TDU3COGraph-0001-RFC5-600V-21A_1000_TVP.csv',
    '190612 Radial scans20190612_134348_TDU3COGraph-0001-RFC5-600V-21A_1000_TVP.csv',
    '190612 Radial scans20190612_134630_TDU3COGraph-0001-RFC5-600V-21A_1000_TVP.csv',
    '190612 Radial scans20190612_134816_TDU3COGraph-0001-RFC5-600V-21A_1000_TVP.csv',
    '190612 Radial scans20190612_135826_TDU3COGraph-0001-RFC5-600V-21A_1000_TVP.csv',
    '190612 Radial scans20190612_140254_TDU3COGraph-0001-RFC5-600V-21A_1000_TVP.csv',
    '190612 Radial scans20190612_140610_TDU3COGraph-0001-RFC5-600V-21A_1000_TVP.csv',
    '190612 Radial scans20190612_140805_TDU3COGraph-0001-RFC5-600V-21A_1000_TVP.csv',
    '190612 Radial scans20190612_141724_TDU3COGraph-0001-RFC5-600V-21A_1000_TVP.csv',
    '190612 Radial scans20190612_141915_TDU3COGraph-0001-RFC5-600V-21A_1000_TVP.csv',
    '190612 Radial scans20190612_142218_TDU3COGraph-0001-RFC5-600V-21A_1000_TVP.csv',
    '190612 Radial scans20190612_142400_TDU3COGraph-0001-RFC5-600V-21A_1000_TVP.csv',
    '190612 Radial scans20190612_143612_TDU3COGraph-0001-RFC5-600V-21A_1000_TVP.csv',
]

# %% CURVE FITTING INITIAL VALUES

# Guesses
c1, c2, c3 = 1, 3, 4
g_a1, g_s1 = 20, 2
l_a1, l_w1 = 20, 3

skew1 = 1.2
skew2 = 1.2

mc, mc1, mc2 = 0, 0, 0

"""
g_a2, g_c2, g_s2 = .6, 4, 20
l_a2, l_c2, l_w2 = .6, 4, 20
"""

# Store to Lists

guess_gauss = [c1, g_a1, g_s1]
guess_lorentz = [c1, l_a1, l_w1]
guess_voigt = [c1, g_a1, g_s1, l_a1, l_w1]
# guess_voigt_2cen = [c1,c2,  g_a1,g_s1,  l_a1,l_w1]

guess_gauss_skew = [skew1, c3, g_a1, g_s1]
guess_lorentz_skew = [skew1, c1, l_a1, l_w1]
guess_voigt_skew = [skew1, c1, g_a1, g_s1, skew2, c2, l_a1, l_w1]

guess_voigt_skew_cen = [skew1, c1, g_a1, g_s1, skew2, l_a1, l_w1]
guess_voigt_skew_cen_mod1 = [mc, skew1, c1, g_a1, g_s1, skew2, l_a1, l_w1]
guess_voigt_skew_cen_mod2 = [mc1, mc2, skew1, c1, g_a1, g_s1, skew2, l_a1, l_w1]

# %% CHOOSE TYPE OF FIT AND GUESSES

# function_fit = _1gauss
# function_fit = _1lorentz
# function_fit = _1voigt
# function_fit = _2gauss
# function_fit = _2lorentz
# function_fit = _2voigt
# function_fit = _2voigt_2cen

# function_fit = _1gauss_skew
# function_fit = _2gauss_skew
# function_fit = _1lorentz_skew
# function_fit = _2lorentz_skew
# function_fit = _1voigt_skew
# function_fit = _2voigt_skew

# function_fit = _1voigt_skew_cen
# function_fit = _2voigt_skew_cen
# function_fit = _2voigt_skew_cen_mod1
function_fit = _2voigt_skew_cen_mod2

# guesses = guess_gauss
# guesses = guess_lorentz
# guesses = guess_voigt
# guesses = guess_voigt_2cen

# guesses = guess_gauss_skew
# guesses = guess_lorentz_skew
# guesses = guess_voigt_skew

# guesses = guess_voigt_skew_cen
# guesses = guess_voigt_skew_cen_mod1
guesses = guess_voigt_skew_cen_mod2

"""
List of Well-Fitting Functions:

_2lorentz_skew
_2voigt

"""

# %% IMPORT DATA

# Create File Path from inputs
file_path_head = file_path1 + file_name_head

# Get headers from template
df_head = pd.read_excel(file_path_head, sheet_name=file_sheet_head, header=None)
col_head = df_head.values.tolist()[0]

# Gather and write Data as DataFrames, store into master list
df_list = [0] * len(file_names_data)
for i, name in enumerate(file_names_data):
    file_path_data = file_path1 + name
    df_list[i] = pd.read_csv(file_path_data, lineterminator='\n', names=col_head)

# for df_data in df_list:
df_data = df_list[6]  # Changes which iteration plots ----------------- CHANGE TO INPUT!!

# %% LIMIT SWEEP ANGLE


if limit_sweep_bool:
    # df_data_mod = df_data.drop(df_data[(-limit_sweep <= x <= limit_sweep) for x in df_data['Theta Angle (Degrees)']])
    df_1 = df_data.copy()
    df_1.drop(df_data[df_data['Theta Angle (Degrees)'] >= limit_sweep].index, inplace=True)
    df_1.drop(df_data[df_data['Theta Angle (Degrees)'] <= -limit_sweep].index, inplace=True)
# MAKE COPIES, ADD IF STATEMENT to plotting section?

# %% LIMIT AXES  (evens out scale for plot)

# X (collector angle) (variable separation, constant for each dataframe)
if limit_sweep_bool:
    angle_sweep = df_1[df_1.columns[0]].to_list()
else:
    angle_sweep = df_data[df_data.columns[0]].to_list()

angle_probe = 22  # ** Need to not explicitly state!!!!

collector_list_nested = []
angle_probe_nested = []
if limit_sweep_bool:
    collector_bias = df_1[df_1.columns[24]]  # .to_list()  #.loc(:,'Collector Bias')
else:
    collector_bias = df_data[df_data.columns[24]]  # .to_list()  #.loc(:,'Collector Bias')

for i in range(1, 24):

    # Y (Probe angle) (2 deg of separation for all)
    angle_probe_nested.append([angle_probe] * len(angle_sweep))
    angle_probe -= 2

    # Z (amplitude)
    if limit_sweep_bool:
        collector_list_raw = df_1[df_1.columns[i]]  # .reset_index()  # .to_list()
    else:
        collector_list_raw = df_data[df_data.columns[i]]  # .reset_index()  # .to_list()

    collector_list = collector_bias - collector_list_raw
    # collector_list.reset_index()
    collector_list_nested.append(collector_list)

# %% FIND "ACTUAL" CENTER OF PLOT AND COMBINE 3D DATA TO 2D POLAR - ROTATE ABOUT AXIS

# Reset indexes of lists to be able to iterate through
collector_list_nested_reset = []
for i, collector_series in enumerate(collector_list_nested):
    list_from_series = collector_series.to_list()
    collector_list_nested_reset.append(list_from_series)

# Convert data to "3D" array
for i, ang_probe in enumerate(angle_probe_nested):
    ang_p = ang_probe[0]
    for j, ang_s in enumerate(angle_sweep):
        col_val = collector_list_nested_reset[i][j]
        if i == 0 and j == 0:
            _3d_data = np.array([[ang_p, ang_s, col_val]])
        else:
            _3d_data = np.append(_3d_data, [[ang_p, ang_s, col_val]], axis=0)

# FIND CENTER - NAIVE METHOD

off_pro = 0  # angle probe
off_swp = 0  # angle sweep
off_pro_list = []
off_swp_list = []
variance_list = []

if fit_3d:
    pass


else:
    change_probe = True

    b = 0
    inc = 0
    for a in range(2 * center_steps):  # change to while loop?
        radius_list = []
        amplitude_list = []

        # if change_probe:
        if a == center_steps:
            change_probe = False

        if not change_probe:
            b += 1

        for i, ang_probe in enumerate(angle_probe_nested):
            ang_p = ang_probe[0] + off_pro

            for j, ang_s in enumerate(angle_sweep):
                ang_s += off_swp

                r = (((ang_s + 0.00) ** 2) + ((ang_p + 0.00) ** 2)) ** 0.5  # Offset each?
                amp = collector_list_nested_reset[i][j]

                radius_list.append(r)
                amplitude_list.append(amp)

        rad_amp_dict = dict(
            zip(amplitude_list.copy(), radius_list.copy()))  # map(lambda i,j : (i,j) , amplitude_list,radius_list))

        # Find x values of the 6 max values (only need to do once?) should change for completion
        max_amp_list = []
        max_rad_list = []
        amplitude_list_ref = amplitude_list.copy()
        radius_list_ref = radius_list.copy()
        for i in range(6):
            max_amp = max(amplitude_list_ref)
            max_rad = rad_amp_dict[max_amp]

            # max_amp_index = amplitude_list_ref.index(max_amp)
            # max_rad_index = radius_list_ref.index(max_rad)
            # print(a,i,max_amp,max_rad)

            amplitude_list_ref.remove(max_amp)
            radius_list_ref.remove(max_rad)
            max_amp_list.append(max_amp)
            max_rad_list.append(max_rad)

        # Find variance
        variance = variation(max_rad_list, ddof=1)  # Check what "ddof" means (in-depth)
        variance_list.append(variance)

        # Test variance with last variance
        # Set new increment OR break from loop
        if a == 0:
            inc = +0.1
        elif variance < variance_list[a - 1]:
            pass
        else:
            inc = (-inc) * 0.5

        if find_center:
            if change_probe:
                off_pro += inc
            else:
                off_swp += inc

        off_pro_list.append(off_pro)
        off_swp_list.append(off_swp)

        # end
        # (start loop again)
    x_array = radius_list
    z_array = amplitude_list

# %% SPLIT INSIDE FROM OUTSIDE VALUES (for visuals)


# Split Inside values (combined?)
x_array_out = x_array.copy()
z_array_out = z_array.copy()
x_array_in = []
z_array_in = []

x_cutoff = min(max_rad_list)
# x_cutoff = sum(max_rad_list)/len(max_rad_list)  # May want to change based on how it actually works out????

for i, x_val in reversed(list(enumerate(x_array))):
    if x_val < x_cutoff:
        x_array_in.append(x_array[i])
        z_array_in.append(z_array[i])
        del x_array_out[i]
        del z_array_out[i]

# %% SCIPY FITTING - 2D

if fit_3d:
    popt_1, pcov_1 = scipy.optimize.curve_fit(function_fit, _3d_data[:, :2], _3d_data[:, 2], p0=guesses)
    perr_1 = np.sqrt(np.diag(pcov_1))
    pars_1 = popt_1  # --------------------------------------------------- needed?

if not fit_3d:
    popt_1, pcov_1 = scipy.optimize.curve_fit(function_fit, x_array, z_array, p0=guesses)
    perr_1 = np.sqrt(np.diag(pcov_1))
    pars_1 = popt_1  # --------------------------------------------------- needed?

    x_array_lin = np.linspace(0, 30, 1001)
    peak_1 = function_fit(x_array_lin, *pars_1)

    x_testing = np.linspace(-10, 40, 1001)
    peak_test = function_fit(x_testing, *pars_1)

    # Below if statements are to print to command window the ideal values found by the above fitting
    if function_fit == _1voigt_skew or function_fit == _2voigt_skew:
        pars_1g = pars_1[0:4]
        pars_1l = pars_1[4:8]
        if function_fit == _1voigt_skew:
            peak_1g = _1gauss_skew(x_testing, *pars_1g)
            peak_1l = _1lorentz_skew(x_testing, *pars_1l)
        if function_fit == _2voigt_skew:
            peak_1g = _2gauss_skew(x_testing, *pars_1g)
            peak_1l = _2lorentz_skew(x_testing, *pars_1l)

    if function_fit == _1voigt_skew_cen or function_fit == _2voigt_skew_cen:
        pars_1g = pars_1[0:4]
        pars_1l = [pars_1[4], pars_1[1], pars_1[5], pars_1[6]]
        if function_fit == _1voigt_skew_cen:
            peak_1g = _1gauss_skew(x_testing, *pars_1g)
            peak_1l = _1lorentz_skew(x_testing, *pars_1l)
        if function_fit == _2voigt_skew_cen:
            peak_1g = _1gauss_skew(x_testing, *pars_1g)  # CHANGE TO 2??
            peak_1l = _1lorentz_skew(x_testing, *pars_1l)

    if function_fit == _2voigt_skew_cen_mod1:
        pars_1g = pars_1[1:5]
        pars_1l = [pars_1[5], pars_1[2], pars_1[6], pars_1[7]]
        peak_1g = _1gauss_skew(x_testing, *pars_1g)  # CHANGE TO 2??
        peak_1l = _1lorentz_skew(x_testing, *pars_1l)

    if function_fit == _2voigt_skew_cen_mod2:
        pars_1g = pars_1[1:5]
        pars_1l = [pars_1[5], pars_1[2], pars_1[6], pars_1[7]]
        peak_1g = _1gauss_skew(x_testing, *pars_1g)  # CHANGE TO 2??
        peak_1l = _1lorentz_skew(x_testing, *pars_1l)

    if function_fit == _1voigt or function_fit == _2voigt:
        pars_1g = pars_1[1:3]
        pars_1l = pars_1[3:5]
        peak_1g = _1gauss(x_array_lin, pars_1[0], *pars_1g)
        peak_1l = _1lorentz(x_array_lin, pars_1[0], *pars_1l)

        print("=====  Voigt  =====")
        print()
        print("WEIGHTS:")
        print("Gauss = %0.4f" % ((pars_1[1] / (abs(pars_1[1]) + abs(pars_1[3]))) * 100))
        print("Lorentz = %0.4f" % ((pars_1[3] / (abs(pars_1[1]) + abs(pars_1[3]))) * 100))

    if function_fit == _2voigt_2cen:
        pars_1g = pars_1[1:3]
        pars_1l = pars_1[4:6]
        peak_1g = _1gauss(x_array_lin, pars_1[0], *pars_1g)
        peak_1l = _1lorentz(x_array_lin, pars_1[3], *pars_1l)
        print("=====  Voigt - Free Centers  =====")
        print()
        print("WEIGHTS:")
        print("Gauss = %0.4f" % ((pars_1[1] / (abs(pars_1[1]) + abs(pars_1[4]))) * 100))
        print("Lorentz = %0.4f" % ((pars_1[4] / (abs(pars_1[4]) + abs(pars_1[4]))) * 100))

    print("Output Values for: " + function_fit.__name__)
    print(pars_1)
    print()

# %% 2D PLOT

if plot_2D:
    """
    x_array = angle_sweep
    z_array = collector_list_nested[col_2D]

    #y_array = angle_probe_nested[col_2D]
    #"""

    """
    # x_array = radius_list
    # z_array = amplitude_list
    """

    # Create Figure
    fig1 = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 0.25])
    ax1 = fig1.add_subplot(gs[0])
    ax2 = fig1.add_subplot(gs[1])
    gs.update(hspace=0)

    # Plot to Figure
    ax1.plot(x_array_out, z_array_out, "o", markersize=1)
    ax1.plot(x_array_in, z_array_in, "o", markersize=1)
    ax1.plot(x_array_lin, peak_1, "k", linewidth=2)

    ax1.plot(x_testing, peak_test, "--", alpha=.5)

    # Set Parameters of Figure
    ax2.set_xlabel("Probe Angle Relative to Centerline (deg)", family="serif", fontsize=12)
    ax1.set_ylabel("Beam Current (mA/mm2)", family="serif", fontsize=12)
    ax2.set_ylabel("Residual", family="serif", fontsize=12)
    ax1.set_title("Curve Fit for Function: " + function_fit.__name__)
    ax1.grid()
    ax2.grid()

    # ax1.plot(x_testing, peak_test)
    # ax1.set_ylim(0, 10)

    # PLOT CHARACTERISTIC LINES

    if function_fit == _1voigt or function_fit == _2voigt:
        ax1.plot(x_array_lin, peak_1g)
        ax1.plot(x_array_lin, peak_1l)

    if function_fit == _2voigt_2cen:
        ax1.plot(x_array_lin, peak_1g)
        ax1.plot(x_array_lin, peak_1l)

    # """
    if function_fit == _1voigt_skew or function_fit == _2voigt_skew or function_fit == _1voigt_skew_cen or function_fit == _2voigt_skew_cen or function_fit == _2voigt_skew_cen_mod1 or function_fit == _2voigt_skew_cen_mod2:
        ax1.plot(x_testing, peak_1g, "--", alpha=.5)
        ax1.plot(x_testing, peak_1l, "--", alpha=.5)
    # """

    # RESIDUAL
    # """
    # ----------------------- NEED TO MODIFY -----------------------
    residual = []
    residual_array = function_fit(x_array, *popt_1)
    for z, res in zip(z_array, residual_array):
        residual.append(z - res)
    RMSE = (np.square(residual).mean()) ** 0.5
    # ----------------------- NEED TO MODIFY -----------------------
    ax2.plot(x_array, residual, 'o', markersize=1)
    center_y = np.zeros(len(x_array_lin))
    ax2.plot(x_array_lin, center_y, 'k--')

    res_count = 0
    for val in residual:
        res_count += abs(val)
    res_avg = res_count / len(residual)

    print()
    print("-----  Residual  -----")
    print("Max =", max(residual))
    print("Min =", min(residual))
    print("Avg =", res_avg)
    print("RMSE =", RMSE)
    print()
    # print("Defining variables:")
    # print(pars_1)
    # """

    # %% Rotate Calculated distribution around Z-axis
    #    """
    mat_arr = np.linspace(0, 30, 1000)
    x_mat = np.tile(mat_arr, (len(mat_arr), 1))
    y_mat = x_mat.copy().T

    z_mat = np.zeros((len(mat_arr), len(mat_arr)))  # Zeros
    for i in range(len(mat_arr)):
        for j in range(len(mat_arr)):
            r = (x_mat[i, j] ** 2 + y_mat[i, j] ** 2) ** 0.5
            z_mat[i, j] = function_fit([r], *pars_1)[0]
            # z_mat[i, j] = function_fit([r], *pars_1)
    # """

# %% SPLIT DATA INTO 4 QUADRANTS
# """
collector_list_diff_nested = []

sweepQ1_list = []  # X
sweepQ2_list = []
sweepQ3_list = []
sweepQ4_list = []
probeQ1_list = []  # Y
probeQ2_list = []
probeQ3_list = []
probeQ4_list = []
radiusQ1_list = []  # R
radiusQ2_list = []
radiusQ3_list = []
radiusQ4_list = []
ampQ1_list = []  # Z
ampQ2_list = []
ampQ3_list = []
ampQ4_list = []

for i, ang_probe in enumerate(angle_probe_nested):
    collector_list_diff = []
    for j, ang_s in enumerate(angle_sweep):
        r = (((ang_s + off_swp + 0.00) ** 2) + ((ang_probe[0] + off_pro + 0.00) ** 2)) ** 0.5  # Offset each?
        diff = collector_list_nested_reset[i][j] - function_fit([r], *pars_1)[0]
        collector_list_diff.append(diff)

        if ang_probe[0] + off_pro == 0:
            pass
        elif ang_probe[0] + off_pro > 0:
            if ang_s + off_swp >= 0:
                sweepQ1_list.append(ang_s + off_swp)
                probeQ1_list.append(ang_probe[0] + off_pro)
                radiusQ1_list.append(r)
                ampQ1_list.append(collector_list_nested_reset[i][j])
            else:
                sweepQ2_list.append(ang_s + off_swp)
                probeQ2_list.append(ang_probe[0] + off_pro)
                radiusQ2_list.append(r)
                ampQ2_list.append(collector_list_nested_reset[i][j])

        elif ang_probe[0] + off_pro < 0:
            if ang_s + off_swp >= 0:
                sweepQ3_list.append(ang_s + off_swp)
                probeQ3_list.append(ang_probe[0] + off_pro)
                radiusQ3_list.append(r)
                ampQ3_list.append(collector_list_nested_reset[i][j])
            else:
                sweepQ4_list.append(ang_s + off_swp)
                probeQ4_list.append(ang_probe[0] + off_pro)
                radiusQ4_list.append(r)
                ampQ4_list.append(collector_list_nested_reset[i][j])
        else:
            print('Error, not in range:', ang_probe[0] + off_pro)

    collector_list_diff_nested.append(collector_list_diff)

# %% STATISTICS

# Student's t-test:
ttest12 = st.ttest_ind(ampQ1_list, ampQ2_list)
ttest13 = st.ttest_ind(ampQ1_list, ampQ3_list)
ttest14 = st.ttest_ind(ampQ1_list, ampQ4_list)
ttest23 = st.ttest_ind(ampQ2_list, ampQ3_list)
ttest24 = st.ttest_ind(ampQ2_list, ampQ4_list)
ttest34 = st.ttest_ind(ampQ3_list, ampQ4_list)

print()
print("T-Tests:")
print("1-2:", ttest12[1], ttest12[1] > 0.2)  # Prints "true" if p-value is within range
print("1-3:", ttest13[1], ttest13[1] > 0.2)
print("1-4:", ttest14[1], ttest14[1] > 0.2)
print("2-3:", ttest23[1], ttest23[1] > 0.2)
print("2-4:", ttest24[1], ttest24[1] > 0.2)
print("3-4:", ttest34[1], ttest34[1] > 0.2)
# """

# %% 3D plot???

if plot_3D:

    fig2 = plt.figure(figsize=(12, 8))
    ax3 = fig2.add_subplot(1, 2, 1, projection='3d')
    # ax3 = plt.axes(projection='3d')
    # ax = fig.gca(projection='3d')

    for i in range(len(collector_list_nested)):
        # ax.plot3D(angle_sweep, angle_probe_nested[i], collector_list_nested[i])
        ax3.plot3D(angle_sweep, angle_probe_nested[i], collector_list_nested[i])
        ax3.scatter(angle_sweep, angle_probe_nested[i], collector_list_nested[i], alpha=0.5, s=8)

    if plot_2D:
        ax3.plot_surface(x_mat, y_mat, z_mat, alpha=0.5)
        ax3.plot_surface(-x_mat, y_mat, z_mat, alpha=0.5)
        ax3.plot_surface(x_mat, -y_mat, z_mat, alpha=0.5)
        ax3.plot_surface(-x_mat, -y_mat, z_mat, alpha=0.5)

    ax3.set_xlabel('Sweep Angle ($\\theta$)', fontsize=20)
    ax3.set_ylabel('Probe Angle ($\phi$)', fontsize=20)
    ax3.set_zlabel('Amplitude', fontsize=20)

    ax4 = fig2.add_subplot(1, 2, 2, projection='3d')
    # fig3 = plt.axes(projection='3d')
    # ax4 = plt.axes(projection='3d')

    for i in range(len(collector_list_diff_nested)):
        ax4.plot3D(angle_sweep, angle_probe_nested[i], collector_list_diff_nested[i])
        ax4.scatter(angle_sweep, angle_probe_nested[i], collector_list_diff_nested[i], alpha=0.5, s=2)

    ax4.set_xlabel('Sweep Angle ($\\theta$)', fontsize=20)
    ax4.set_ylabel('Probe Angle ($\phi$)', fontsize=20)
    ax4.set_zlabel('Amplitude', fontsize=20)

    # space factor
    # ax.yaxis._axinfo['label']['space_factor'] = 20
    # ax.xaxis._axinfo['label']['space_factor'] = 20

    # set zticks
    # change font size

    # ax.zaxis.set_rotate_label(False)

    # Check if actually doing anything
    # fig2.tight_layout()

# %% SHOW PLOT(S)
plt.show()

