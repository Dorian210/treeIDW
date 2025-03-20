# %%
import numpy as np
import matplotlib.pyplot as plt
from treeIDW import treeIDW

def plot_vector_fields(xi, yi, ui, vi, x, y, u, v):
    norm_max = max(np.sqrt(u*u + v*v).max(), np.sqrt(ui*ui + vi*vi).max())
    scale = 1/(n*norm_max)
    fig, ax = plt.subplots()
    ax.quiver(x, y, u*scale, v*scale, color='#1b9e77', label='Boundary nodes', scale=1, scale_units='xy')
    ax.quiver(xi, yi, ui*scale, vi*scale, color='#d95f02', label='Internal nodes', scale=1, scale_units='xy')
    ax.set_xlim(-1/n, 1 + 1/n)
    ax.set_ylim(-1/n, 1 + 1/n)
    ax.set_aspect(1)
    plt.legend(loc='center')
    plt.axis('off')
    plt.show()

np.random.seed(3)
n = 15
mag = 1
noise = 0.2
x_tmp = np.linspace(0, 1, n)
y_tmp = np.linspace(0, 1, n)
zero = np.zeros(n)
one = np.ones(n)
ramp = np.linspace(0, mag, n)
magnitude = mag*one
perimeter = lambda a, b, c, d: np.hstack((a[:-1], b[:-1], c[:0:-1], d[:0:-1]))
x = perimeter(x_tmp, one, x_tmp, zero)
y = perimeter(zero, y_tmp, one, y_tmp)
# u = perimeter(zero, ramp, magnitude, ramp)
u = perimeter(one, zero, -one, zero)
u += np.random.randn(u.size)*noise
# v = perimeter(zero, zero, zero, zero)
v = perimeter(zero, one, zero, -one)
v += np.random.randn(v.size)*noise
xi, yi = np.meshgrid(x_tmp[1:-1], y_tmp[1:-1])
xi, yi = xi.flatten(), yi.flatten()

boundary_nodes = np.stack((x, y)).T # shape (N_boundary, 2) 2 for the 2D space
boundary_field = np.stack((u, v)).T # shape (N_boundary, 2) 2 beacause the field is a vector field
internal_nodes = np.stack((xi, yi)).T # shape (N_internal, 2) 2 for the 2D space

uivi = treeIDW(boundary_nodes, boundary_field, internal_nodes, parallel=True) # shape (N_internal, 2) 2 beacause the field is a vector field
ui, vi = uivi.T

plot_vector_fields(xi, yi, ui, vi, x, y, u, v)
# %%
