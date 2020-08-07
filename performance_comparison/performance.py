import numpy as np
import sys
import bempp.api

h = float(sys.argv[1])

if sys.argv[2] == "fmm":
    prefix = "fmm"
    assembler = "fmm"
    device = None
elif sys.argv[2] == "opencl":
    prefix = "opencl"
    assembler = "dense"
    device = "opencl"
elif sys.argv[2] == "numba":
    prefix = "numba"
    assembler = "dense"
    device = "numba"
else:
    raise ValueError(f"Unknown type: {sys.argv[2]}")

grid = bempp.api.shapes.sphere(h=h)
space = bempp.api.function_space(grid, "P", 1)

with bempp.api.Timer() as t:
    op = bempp.api.operators.boundary.helmholtz.single_layer(
        space, space, space, 3, assembler=assembler, device_interface=device)
    mat = op.weak_form()

with open(f"output/{prefix}_assembly_{h}", "a") as f:
    f.write(f"{t.interval}\n")

total = 0
for i in range(20):
    vec = np.random.rand(space.global_dof_count)
    with bempp.api.Timer() as t:
        mat @ vec
    total += t.interval

bempp.api.clear_fmm_cache()

with open(f"output/{prefix}_matvec_{h}", "a") as f:
    f.write(f"{total}\n")
