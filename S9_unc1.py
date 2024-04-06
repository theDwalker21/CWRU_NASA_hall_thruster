# %% Import Libraries
# import ad
import numpy as np
# import matplotlib
import matplotlib.pyplot as plt
import pandas as pd



# import scipy
from scipy import optimize
from scipy import integrate
# from scipy import stats as st
# from scipy.stats import variation
import scipy.special

# from lmfit import minimize, Parameters, Parameter, report_fit
# from mpl_toolkits import mplot3d
from matplotlib import gridspec
import matplotlib.tri as mtri

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

from sqlalchemy import create_engine
# import ipywidgets as widgets
import IPython
import warnings
import xlsxwriter
import openpyxl
import string
"""

import uncertainties as u
from uncertainties import ufloat
from uncertainties.umath import *

import pydoc

import sympy as sym

# %% Import Functions

from E651_func_v1 import _2voigt_skew_cen_mod2


# %% Inputs




# %% Main Code

# Define Symbolic Parameters
x = sym.Symbol('x')  # Independent Variable
c = sym.Symbol('c')  # center

aG = sym.Symbol('aG')  # Gauss
sG = sym.Symbol('sG')
mG = sym.Symbol('mG')

aC = sym.Symbol('aC')  # Cauchy
sC = sym.Symbol('sC')
mC = sym.Symbol('mC')

# Define constants
pi = np.pi
v = ((2*pi)/(pi-(8*aG**2)+(3*pi*aG**2)))**.5

# Integrate
Gint = sym.integrate((mG*sym.exp(-(((abs(x+c))/(v*sG*(1+aG*sym.sign(x+c))**2))**2)))/(v*sG*(np.pi**.5)), x)

# Differentiate
Gint_a = sym.diff(Gint, aG)


# %% Outputs

print("Gint:", Gint)
print("Gint_a:", Gint_a)


# %% Plots




# end
