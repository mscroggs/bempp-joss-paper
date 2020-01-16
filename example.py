import bempp.api
import numpy as np
from bempp.api.operators.boundary import sparse, helmholtz
from bempp.api.operators.potential import helmholtz as helmholtz_pot
from matplotlib import pyplot as plt

k = 3

grid = bempp.api.shapes.sphere(h=0.1)
space = bempp.api.function_space(grid, "DP", 0)

id = sparse.identity(space, space, space)
dlp = helmholtz.double_layer(space, space, space, k)
slp = helmholtz.single_layer(space, space, space, k)


@bempp.api.complex_callable
def u_inc(x, n, domain_index, result):
    result[0] = np.exp(1j * k * x[0])


u_inc = bempp.api.GridFunction(space, fun=u_inc)

rhs = (dlp - 0.5 * id) * u_inc

l, info = bempp.api.linalg.gmres(slp, rhs)

Nx = 200
Ny = 200
xmin, xmax, ymin, ymax = [-3, 3, -3, 3]
plot_grid = np.mgrid[xmin:xmax:Nx * 1j, ymin:ymax:Ny * 1j]
points = np.vstack((plot_grid[0].ravel(),
                    plot_grid[1].ravel(),
                    np.zeros(plot_grid[0].size)))
u_evaluated = np.zeros(points.shape[1], dtype=np.complex128)
u_evaluated[:] = np.nan

x, y, z = points
idx = np.sqrt(x**2 + y**2) > 1.0

slp_pot = helmholtz_pot.single_layer(space, points[:, idx], k)
dlp_pot = helmholtz_pot.double_layer(space, points[:, idx], k)
res = np.real(np.exp(1j * k * points[0, idx])
              - dlp_pot.evaluate(u_inc) - slp_pot.evaluate(l))
u_evaluated[idx] = res.flat
u_evaluated = u_evaluated.reshape((Nx, Ny))

fig = plt.figure(figsize=(10, 8))
plt.imshow(np.real(u_evaluated.T), extent=[-3, 3, -3, 3])
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.savefig("solution.png")
