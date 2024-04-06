
#%% Import Libraries
import numpy as np
import matplotlib.pyplot as plt

# %% Functions
from E651_func_v1 import _1gauss, _1lorentz, _1voigt, \
    _1gauss_skew, _1lorentz_skew, \
    _2gauss, _2lorentz, _2voigt, \
    _2gauss_skew, _2lorentz_skew, _2voigt_skew


# %% Inputs
n = 0
dist0 = 8
dist1 = 12
num = 1000 + 1

cen0 = 0
cen1 = 3

g_amp = 5
sigma = 2
g_a = .5

l_amp = 1
wid = 2
l_a = .5

# %% Calculations
x_reg = np.linspace(-dist0, dist0, num)
x_skew = np.linspace(0, dist1, num)

y_gauss = _1gauss(x_reg, cen0, g_amp, sigma)
y_lorentz = _1lorentz(x_reg, cen0, l_amp, wid)
y_voigt = _1voigt(x_reg, cen0, g_amp, sigma, l_amp, wid)/2

y_gauss_skew = _1gauss_skew(x_skew, g_a, cen1, g_amp, sigma)
y_lorentz_skew = _1lorentz_skew(x_skew, l_a, cen1, l_amp*6.7, wid)
y_voigt_skew = (np.array(y_gauss_skew) + np.array(y_lorentz_skew))/2

y_gauss_bim = _2gauss(x_reg, cen1, g_amp, sigma)
y_lorentz_bim = _2lorentz(x_reg, cen1, l_amp*.91, wid)
y_voigt_bim = _2voigt(x_reg, cen1, g_amp*.95, sigma, l_amp*.91*.95, wid)/2

y_gauss_bs = _2gauss_skew(x_reg, g_a, cen1/2, g_amp, sigma)
y_lorentz_bs = _2lorentz_skew(x_reg, l_a, cen1/2, l_amp*6, wid)
y_voigt_bs = np.array(_2voigt_skew(x_reg, g_a, cen1/2, g_amp*.95, sigma, l_a, cen1/2, l_amp*6, wid))/2
# x_list, aG, cG, ampG, sG, aL, cL, ampL, wL

# %% Plot

save_photo = True
figure_path = r'C:\Users\David\Desktop\Semester10\FPI_Lab\FIGURES'

n += 1
fig1 = plt.figure(n, figsize=(3.54, 3.54), dpi=300)
ax1 = fig1.add_subplot(1, 1, 1)
ax1.plot(x_reg, y_gauss, alpha=0.7, label='Gauss')
ax1.plot(x_reg, y_lorentz, alpha=0.7, label='Cauchy')
ax1.plot(x_reg, y_voigt, alpha=0.7, label='P-Voigt')
ax1.legend()
ax1.grid()
plt.tight_layout()
figure_name = r'\curves_reg'
if save_photo:
    plt.savefig(figure_path + figure_name)

n += 1
fig2 = plt.figure(n, figsize=(3.54, 3.54), dpi=300)
ax2 = fig2.add_subplot(1, 1, 1)
ax2.plot(x_skew, y_gauss_skew, alpha=0.7, label='Gauss (skew)')
ax2.plot(x_skew, y_lorentz_skew, alpha=0.7, label='Cauchy (skew)')
ax2.plot(x_skew, y_voigt_skew, alpha=0.7, label='P-Voigt (skew)')
ax2.legend()
ax2.grid()
plt.tight_layout()
figure_name = r'\curves_skew'
if save_photo:
    plt.savefig(figure_path + figure_name)

n += 1
fig3 = plt.figure(n, figsize=(3.54, 3.54), dpi=300)
ax3 = fig3.add_subplot(1, 1, 1)
ax3.plot(x_reg, y_gauss_bim, alpha=0.7, label='Gauss (bim.)')
ax3.plot(x_reg, y_lorentz_bim, alpha=0.7, label='Cauchy (bim.)')
ax3.plot(x_reg, y_voigt_bim, alpha=0.7, label='P-Voigt (bim.)')
ax3.legend()
ax3.grid()
plt.tight_layout()
figure_name = r'\curves_bimodal'
if save_photo:
    plt.savefig(figure_path + figure_name)

n += 1
fig4 = plt.figure(n, figsize=(3.54, 3.54), dpi=300)
ax4 = fig4.add_subplot(1, 1, 1)
ax4.plot(x_reg, y_gauss_bs, alpha=0.7, label='Gauss (b.s.)')
ax4.plot(x_reg, y_lorentz_bs, alpha=0.7, label='Cauchy (b.s.)')
ax4.plot(x_reg, y_voigt_bs, alpha=0.7, label='P-Voigt (b.s.)')
ax4.legend()
ax4.grid()
plt.tight_layout()
figure_name = r'\curves_bimodal_skew'
if save_photo:
    plt.savefig(figure_path + figure_name)
