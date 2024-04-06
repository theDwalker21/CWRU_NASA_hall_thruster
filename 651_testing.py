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

#from random import random
import random

# %% Import Functions

from E651_func_v1 import _1gauss, _2gauss_skew, _2lorentz_skew, _2voigt_skew_cen_mod1, _2voigt_skew_cen_mod2,\
    _2voigt_skew_cen_mod2_rand, _2voigt_skew_cen_mod2a, _2voigt_skew_cen_mod2a_rand, _2voigt_skew_cen_mod3


# %% Inputs

_2d = False

start, end, steps = 0, 50, 101

# _1gauss
cen, amp, sigma = 10, 5, 5

# _2voigt_skew_cen_mod1
mas_cen = .94
mas_col, mas_swe, = -20, -30
aG,cG,ampG,sG = 2.25, 3.70, 7.91, 4.34
aL,ampL,wL = 3.98,9.10,0.90

# %% Math

# X, Y, one array
x_array = np.linspace(start,end,steps)
y_array = x_array.copy()



for i, x_val in enumerate(x_array):
    for j, y_val in enumerate(y_array):
        if i == 0 and j == 0:
            arr_in = np.array([[x_val,y_val]])
        else:
            arr_in = np.append(arr_in, [[x_val,y_val]], axis=0)

# X arr, Y arr (separate)

X_mat, Y_mat = np.meshgrid(x_array,y_array)


# %% Run Functions

# z_array = _1gauss(x_array,cen,  amp,sigma)
# z_array = _2voigt_skew_cen_mod1(x_array, mas_cen, aG, cG, ampG, sG, aL, ampL, wL)

z_array = _2voigt_skew_cen_mod2(arr_in, mas_col, mas_swe, aG, cG, ampG, sG, aL, ampL, wL)
z_array_rand = _2voigt_skew_cen_mod2_rand(arr_in, mas_col, mas_swe, aG, cG, ampG, sG, aL, ampL, wL)
# Z_mat = _2voigt_skew_cen_mod3( [X_mat, Y_mat], mas_col, mas_swe, aG, cG, ampG, sG, aL, ampL, wL)

# z_array1 = _2gauss_skew(x_array, aG, cG, ampG, sG)
# z_array2 = _2lorentz_skew(x_array, aL, cG, ampL, wL)


# %% Create noise

#random.seed(10)
#rand_list = np.zeros_like(z_array)
#for i in range(0,len(z_array)):
#    rand_list[i] = random.randrange(0, 11)
#z_array_rand = z_array + (rand_list/10)

#arr_in_rand = np.zeros_like(arr_in)
#arr_in_rand[:,0:2] = arr_in[:,0:2]
#arr_in_rand2 = np.concatenate([arr_in_rand, column_to_be_added], axis=1)
#arr_in_rand2 = np.append(arr_in_rand, z_array_rand, axis=1)
# arr_in_rand[:,2] = z_array_rand

# %% Integrate

#integral = scipy.integrate.dblquad(_2voigt_skew_cen_mod2a, -22, 22, -22, 22, args=(mas_col, mas_swe, aG, cG, ampG, sG, aL, ampL, wL))
#print("Integral:", integral[0])

#integral_rand = scipy.integrate.dblquad(_2voigt_skew_cen_mod2a_rand, -22, 22, -22, 22, args=(mas_col, mas_swe, aG, cG, ampG, sG, aL, ampL, wL))
#print("Integral rand:", integral_rand[0])

options = {'limit':100}

integral = scipy.integrate.nquad(_2voigt_skew_cen_mod2a,[[-22,22],[-22,22]], args=(mas_col, mas_swe, aG, cG, ampG, sG, aL, ampL, wL),opts=[options,options])
print("Integral:", integral)
integral_rand = scipy.integrate.nquad(_2voigt_skew_cen_mod2a_rand,[[-22,22],[-22,22]], args=(mas_col, mas_swe, aG, cG, ampG, sG, aL, ampL, wL),opts=[options,options])
print("Integral rand:", integral_rand)


# %% Plot

if _2d:

    fig = plt.figure(figsize=(8,8))
    plt.plot(x_array, [0]*len(x_array), 'k--')
    plt.plot(x_array, [max(z_array)]*len(x_array), 'k--')
    try:
        plt.plot(x_array,z_array)
    except:
        pass
    try:
        plt.plot(x_array,z_array1, '--')
    except:
        pass
    try:
        plt.plot(x_array,z_array2, '--')
    except:
        pass

    plt.xlabel('Location')
    plt.ylabel('Amplitude')
    plt.title('TEST PLOT')
else:
    fig = plt.figure(figsize=(32, 16))
    ax3 = fig.add_subplot(1,2,1, projection='3d')


    #ax3.plot_surface(X_mat,Y_mat,Z_mat, cmap=plt.cm.plasma, linewidth=0, antialiased=False)

    ax3.plot_trisurf(arr_in[:,0],arr_in[:,1],np.array(z_array), cmap=plt.cm.coolwarm, alpha=.7)
    #ax3.plot_trisurf(arr_in[:,0],arr_in[:,1],np.array(z_array_rand), cmap=plt.cm.coolwarm, alpha=.7)

    ax4 = fig.add_subplot(1, 2, 2, projection='3d')
    ax4.plot_trisurf(arr_in[:, 0], arr_in[:, 1], np.array(z_array_rand), cmap=plt.cm.coolwarm, alpha=.7)


plt.tight_layout()
plt.show()

# %%

