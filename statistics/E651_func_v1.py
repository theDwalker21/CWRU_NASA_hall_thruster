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

import random

from math import log10, floor

# %% Math Functions


# ==========  Single  ==========

def _1gauss(x_array, cen, amp, sigma):
    return amp * (1 / (sigma * (np.sqrt(2 * np.pi)))) * (np.exp((-1 / 2) * (((x_array - cen) / sigma) ** 2)))


def _1lorentz(x, cen, amp, wid):
    return amp * wid ** 2 / ((x - cen) ** 2 + wid ** 2)


def _1voigt(x, cen, g_amp, g_std, l_amp, l_wid):
    return (g_amp * (1 / (g_std * (np.sqrt(2 * np.pi)))) * (np.exp(-((x - cen) ** 2) / ((2 * g_std) ** 2)))) + \
           (l_amp * l_wid ** 2 / ((x - cen) ** 2 + l_wid ** 2))


# ==========  Double  ========== (bimodal)

def _2gauss(x_array, cen, amp, sigma):
    return amp * (1 / (sigma * (np.sqrt(2 * np.pi)))) * (np.exp((-1 / 2) * (((x_array - cen) / sigma) ** 2))) + \
           amp * (1 / (sigma * (np.sqrt(2 * np.pi)))) * (np.exp((-1 / 2) * (((x_array + cen) / sigma) ** 2)))


def _2lorentz(x, cen, amp, wid):
    return amp * wid ** 2 / ((x - cen) ** 2 + wid ** 2) + \
           amp * wid ** 2 / ((x + cen) ** 2 + wid ** 2)


def _2voigt(x, cen, g_amp, g_std, l_amp, l_wid):
    return (g_amp * (1 / (g_std * (np.sqrt(2 * np.pi)))) * (np.exp(-((x - cen) ** 2) / ((2 * g_std) ** 2)))) + \
           (l_amp * l_wid ** 2 / ((x - cen) ** 2 + l_wid ** 2)) + \
           (g_amp * (1 / (g_std * (np.sqrt(2 * np.pi)))) * (np.exp(-((x + cen) ** 2) / ((2 * g_std) ** 2)))) + \
           (l_amp * l_wid ** 2 / ((x + cen) ** 2 + l_wid ** 2))


# ==========  Varied Centers  ==========

def _2voigt_2cen(x, g_cen, g_amp, g_std, l_cen, l_amp, l_wid):
    return (g_amp * (1 / (g_std * (np.sqrt(2 * np.pi)))) * (np.exp(-((x - g_cen) ** 2) / ((2 * g_std) ** 2)))) + \
           (l_amp * l_wid ** 2 / ((x - l_cen) ** 2 + l_wid ** 2)) + \
           (g_amp * (1 / (g_std * (np.sqrt(2 * np.pi)))) * (np.exp(-((x + g_cen) ** 2) / ((2 * g_std) ** 2)))) + \
           (l_amp * l_wid ** 2 / ((x + l_cen) ** 2 + l_wid ** 2))


# ==========  Skewed Distributions  ==========
"""
def _1gauss_skew(x_array,a,cen,amp,sigma):
    output = []
    for val_in in x_array:
        val_out = _1gauss(val_in,cen,amp,sigma) * (1+scipy.special.erf( (a*val_in)/(2**0.5) ))
        output.append(val_out)
    return output

def _2gauss_skew(x_array,a,cen,amp,sigma):
    output = []
    for val_in in x_array:
        val_out1 = _1gauss(val_in, cen, amp, sigma) * (1 + scipy.special.erf(( a * val_in) / (2 ** 0.5)))

        val_out2 = _1gauss(val_in,-cen, amp, sigma) * (1 + scipy.special.erf((-a * val_in) / (2 ** 0.5)))
        output.append(val_out1+val_out2)
    return output
#"""


# Gauss Skewed
def _1gauss_skew(x_list, a, c, amp, s):
    output = []
    v = ((2 * np.pi) / (np.pi - (8 * (a ** 2)) + (3 * np.pi * (a ** 2)))) ** .5
    m = (2 * v * s * a) / (np.pi ** .5)  # Needed?
    try:
        for x in x_list:
            num = np.exp(-(abs(x - c) / (v * s * (1 + (a * np.sign(x - c))))) ** 2)
            den = v * s * (np.pi ** .5)
            output.append(amp * (num / den))
    except:
        num = np.exp(-(abs(x_list - c) / (v * s * (1 + (a * np.sign(x_list - c))))) ** 2)
        den = v * s * (np.pi ** .5)
        output.append(amp * (num / den))

    return output

# Lorentz Skewed
def _1lorentz_skew(x_list, a, c, amp, s):
    output = []
    try:
        for x in x_list:
            num = abs(x - c) ** 2
            den = (s ** 2) * ((a * np.sign(x - c) + 1) ** 2)
            output.append(amp / (s * np.pi * ((num / den) + 1)))
    except:
        num = abs(x_list - c) ** 2
        den = (s ** 2) * ((a * np.sign(x_list - c) + 1) ** 2)
        output.append(amp / (s * np.pi * ((num / den) + 1)))
    return output


# ==========  Partial Derivatives  ==========



# %% Combined Functions


def _2gauss_skew(x_list, a, c, amp, s):
    output = []
    output1 = _1gauss_skew(x_list, a, c, amp, s)
    output2 = _1gauss_skew(x_list, -a, -c, amp, s)
    for val1, val2 in zip(output1, output2):
        output.append(val1 + val2)

    return output


def _2lorentz_skew(x_list, a, c, amp, s):
    output = []
    output1 = _1lorentz_skew(x_list, a, c, amp, s)
    output2 = _1lorentz_skew(x_list, -a, -c, amp, s)
    for val1, val2 in zip(output1, output2):
        output.append(val1 + val2)
    return output


# Voigt Skewed
def _1voigt_skew(x_list, aG, cG, ampG, sG, aL, cL, ampL, wL):
    output = []
    output1 = _1gauss_skew(x_list, aG, cG, ampG, sG)
    output2 = _1lorentz_skew(x_list, aL, cL, ampL, wL)
    for val1, val2 in zip(output1, output2):
        output.append(val1 + val2)
    return output


def _2voigt_skew(x_list, aG, cG, ampG, sG, aL, cL, ampL, wL):
    output = []
    output1 = _2gauss_skew(x_list, aG, cG, ampG, sG)
    output2 = _2lorentz_skew(x_list, aL, cL, ampL, wL)
    for val1, val2 in zip(output1, output2):
        output.append(val1 + val2)
    return output


# Voigt Skewed centered
def _1voigt_skew_cen(x_list, aG, cG, ampG, sG, aL, ampL, wL):
    output = []
    output1 = _1gauss_skew(x_list, aG, cG, ampG, sG)
    output2 = _1lorentz_skew(x_list, aL, cG, ampL, wL)  # Uses same center, cG
    for val1, val2 in zip(output1, output2):
        output.append(val1 + val2)
    return output


def _2voigt_skew_cen(x_list, aG, cG, ampG, sG, aL, ampL, wL):
    output = []
    output1 = _2gauss_skew(x_list, aG, cG, ampG, sG)
    output2 = _2lorentz_skew(x_list, aL, cG, ampL, wL)  # Uses same center, cG
    for val1, val2 in zip(output1, output2):
        output.append(val1 + val2)
    return output


# Modified Skewed centered
def _2voigt_skew_cen_mod1(x_list, mas_cen, aG, cG, ampG, sG, aL, ampL, wL):
    output = []
    if mas_cen == 0:
        output1 = _2gauss_skew(x_list, aG, cG, ampG, sG)
        output2 = _2lorentz_skew(x_list, aL, cG, ampL, wL)  # Uses same center, cG
    else:
        x_list_mod = []
        for x in x_list:
            x_list_mod.append(x + mas_cen)
        output1 = _2gauss_skew(x_list_mod, aG, cG, ampG, sG)
        output2 = _2lorentz_skew(x_list_mod, aL, cG, ampL, wL)  # Uses same center, cG
    for val1, val2 in zip(output1, output2):
        output.append(val1 + val2)
    return output

# For lists
def _2voigt_skew_cen_mod2(arr_in, mas_col, mas_swe, aG, cG, ampG, sG, aL, ampL, wL):
    output = []

    r_list = []
    for i, coord in enumerate(arr_in):
        ang_col = coord[0]
        ang_swe = coord[1]
        r = np.sqrt((ang_col+mas_col)**2 + (ang_swe+mas_swe)**2)
        r_list.append(r)

    output1 = _2gauss_skew(r_list,   aG,cG,ampG,sG)
    output2 = _2lorentz_skew(r_list, aL,cG,ampL,wL)
    for val1, val2 in zip(output1,output2):
        output.append(val1+val2)

    return output


def _2voigt_skew_cen_mod2single(r_list, mas_col, mas_swe, aG, cG, ampG, sG, aL, ampL, wL):
    output = []

    output1 = _2gauss_skew(r_list,   aG,cG,ampG,sG)
    output2 = _2lorentz_skew(r_list, aL,cG,ampL,wL)
    for val1, val2 in zip(output1,output2):
        output.append(val1+val2)

    return output


def _2voigt_skew_cen_mod2_rand(arr_in, mas_col, mas_swe, aG, cG, ampG, sG, aL, ampL, wL):
    output = []

    r_list = []
    for i, coord in enumerate(arr_in):
        ang_col = coord[0]
        ang_swe = coord[1]
        r = np.sqrt((ang_col+mas_col)**2 + (ang_swe+mas_swe)**2)
        r_list.append(r)

    output1 = _2gauss_skew(r_list,   aG,cG,ampG,sG)
    output2 = _2lorentz_skew(r_list, aL,cG,ampL,wL)

    rand_list = np.random.uniform(-.5,.5,len(output1))

    for val1, val2, val_rand in zip(output1,output2,rand_list):
        output.append(val1+val2+val_rand)

    return output


# For individual numbers (inputs)
def _2voigt_skew_cen_mod2a(x, y, mas_col, mas_swe, aG, cG, ampG, sG, aL, ampL, wL):
    ang_col = x
    ang_swe = y
    r = np.sqrt((ang_col+mas_col)**2 + (ang_swe+mas_swe)**2)

    output1 = _2gauss_skew(r,   aG,cG,ampG,sG)
    output2 = _2lorentz_skew(r, aL,cG,ampL,wL)
    output = output1[0] + output2[0]

    return output

def _2voigt_skew_cen_mod2a_rand(x, y, mas_col, mas_swe, aG, cG, ampG, sG, aL, ampL, wL):
    ang_col = x
    ang_swe = y
    r = np.sqrt((ang_col+mas_col)**2 + (ang_swe+mas_swe)**2)

    output1 = _2gauss_skew(r,   aG,cG,ampG,sG)
    output2 = _2lorentz_skew(r, aL,cG,ampL,wL)

    rand_list = np.random.uniform(-.5,.5,len(output1))

    output = output1[0] + output2[0] + rand_list
    return output

def integrate_func(x,y,z):
    return z


# Broken
def _2voigt_skew_cen_mod3(arr_in, mas_col, mas_swe, aG, cG, ampG, sG, aL, ampL, wL):
    arr_X = arr_in[0]
    arr_Y = arr_in[1]
    output = np.zeros_like(arr_X)
    r_arr = np.zeros_like(arr_X)

    for i, ang_col in enumerate(arr_Y[0]):
        for j, ang_swe in enumerate(arr_X):
            print("\n\nIteration values: ", ang_col, ang_swe[0])
            r = np.sqrt((ang_col+mas_col)**2 + (ang_swe[0]+mas_swe)**2)
            print(r)
            r_arr[j, i] = r
            # print(r_arr)
            out1 = _2gauss_skew(r,   aG,cG,ampG,sG)[0]
            out2 = _2lorentz_skew(r, aL,cG,ampL,wL)[0]
            print("out1 = ", out1)
            print("out2 = ", out2)
            # print(output)
            output[j,i] = out1 + out2
            print("Output:", output[j,i])

    """
    r_list = []
    for i, coord in enumerate(arr_in):
        ang_col = coord[0]
        ang_swe = coord[1]
        r = np.sqrt((ang_col+mas_col)**2 + (ang_swe+mas_swe)**2)
        r_list.append(r)

    output1 = _2gauss_skew(r_list,   aG,cG,ampG,sG)
    output2 = _2lorentz_skew(r_list, aL,cG,ampL,wL)
    for val1, val2 in zip(output1,output2):
        output.append(val1+val2)
    # """
    print(output)
    return output


# %% Other Functions

def combine_func(list1, list2):
    output = []
    for i, val1 in enumerate(list1):
        output.append(val1 + list2[i])
    return output


def round_to_1(x):
   return round(x, -int(floor(log10(abs(x)))))





