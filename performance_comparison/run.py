import os
import sys

a_type = sys.argv[1]
assert a_type in ["fmm", "opencl", "numba"]

N = 5

for h in [2 ** -i for i in range(1, 7)]:
    print(f"Starting {a_type} with h={h}")
    with open(f"output/{a_type}_assembly_{h}", "w") as f:
        pass
    with open(f"output/{a_type}_matvec_{h}", "w") as f:
        pass

    for i in range(N):
        print(f"Run {i+1}/{N}")
        assert os.system(f"python performance.py {h} {a_type}") == 0

    with open(f"output/{a_type}_assembly_{h}") as f:
        print("# Assembly")
        print(f.read())
    with open(f"output/{a_type}_matvec_{h}") as f:
        print("# Matvec")
        print(f.read())
