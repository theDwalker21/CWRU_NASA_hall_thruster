# %% Import Libraries
import numpy as np
import matplotlib.pyplot as plt

# %% Velocity Functions

# Don't have to call constants because the variables are global!
def u_r(radius):
    output = -alpha * radius
    return output

def u_t(radius):
    output = (gam/(2*np.pi*radius)) * (1 - np.exp((-alpha * (radius**2))/(2*nu)))
    return output

def u_z(zee):
    output = 2 * alpha * zee
    return output

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


# %% Main Code

# Define Constants
alpha = 10
gam = 5
nu = 1

# Create Independent Variables
#radii = np.linspace(0.1, 1, 12)
#thetas = np.linspace(0, 2*np.pi, 16)

#t, r = np.meshgrid(thetas, radii)

r = np.linspace(0.1, 1, 100)
t = np.linspace(0, 3*2*np.pi, 100)
z = np.linspace(-1, 1, 100)

#x, y = pol2cart(r,t)


#zees = np.linspace(0, 200, 12)
#rz, z = np.meshgrid(radii, zees)

# Get velocity vector
U_r = u_r(r)
U_t = u_t(r)
U_z = u_z(z)

#U_rz = u_r(rz)
#U_z = u_z(z)
#U_r = u_r(radii)
#U_z = u_z(z)

# Convert cartesian to polar coords
#U_r_polar = cart2pol()
#U_z_polar
#U_r_pol, U_t_pol = cart2pol(U_r, U_t)
x,y = pol2cart(r,t)
U_tc, U_rc = pol2cart(U_r, U_t)

# %% Plotting

fig1 = plt.figure(figsize=(8, 8))

# Quiver Plot (vector field)
ax1 = fig1.add_subplot(1, 1, 1, polar=True)
#ax1 = fig1.add_subplot(1, 1, 1)
#ax1 = plt.subplots(polar=True)
ax1.quiver(t, r, U_tc, U_rc)
#ax1.quiver(x, y, U_tc, U_rc)
ax1.set(title='3c: Quiver Plot')# xlabel='x', )
#ax1.axis('equal')
ax1.grid(True)

# Streamlines
#ax2 = fig1.add_subplot(1, 1, 1)
#ax2.quiver(r, z, U_r, U_z)
#ax2.quiver(x, z, U_r, U_z)
#ax2.quiver(rz, z, U_rz, U_z)
#ax2.quiver(r, z, U_r, U_z)
#ax2.quiver(r,z,)
#ax2.set(title='3c: Streamline Plot', xlabel='r', ylabel='z')
#ax2.axis('equal')
#ax2.grid(True)

plt.tight_layout()
plt.show()
#plt.savefig('HW4_P3c')
