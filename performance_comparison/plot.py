
import os

data = {i: {"fmm": {}, "opencl": {}, "numba": {}} for i in ["assembly", "matvec"]}

for file in os.listdir("output"):
    print(file)
    a_type, result, h = file.split("_")
    h = float(h)
    count = 0
    total = 0
    with open(f"output/{file}") as f:
        for line in f:
            if line.strip() != "":
                total += float(line.strip())
                count += 1
    data[result][a_type][h] = total / count

print(data)

import matplotlib
matplotlib.rcParams.update({
    'font.size': 22, 'lines.markersize': 15, 'lines.linewidth': 3})
import matplotlib.pylab as plt

plt.figure(figsize=(10.0, 10.0), dpi=50)

#plt.subplot("121")

fmm_x = sorted(list(data["assembly"]["fmm"].keys()))
fmm_y = [data["assembly"]["fmm"][i] for i in fmm_x]
plt.plot(fmm_x, fmm_y, "o-")

opencl_x = sorted(list(data["assembly"]["opencl"].keys()))
opencl_y = [data["assembly"]["opencl"][i] for i in opencl_x]
plt.plot(opencl_x, opencl_y, "^-")

numba_x = sorted(list(data["assembly"]["numba"].keys()))
numba_y = [data["assembly"]["numba"][i] for i in numba_x]
plt.plot(numba_x, numba_y, "s-")

plt.legend(["ExaFMM", "OpenCL", "Numba"])
plt.xscale("log")
plt.xlabel("$h$")
plt.xlim(plt.xlim()[::-1])
plt.ylabel("Time for assembly (s)")

plt.savefig("../performance_assembly.png")

plt.clf()


plt.figure(figsize=(10.0, 10.0), dpi=50)
#plt.subplot("122")
fmm_x = sorted(list(data["matvec"]["fmm"].keys()))
fmm_y = [data["matvec"]["fmm"][i] for i in fmm_x]
plt.plot(fmm_x, fmm_y, "o-")

opencl_x = sorted(list(data["matvec"]["opencl"].keys()))
opencl_y = [data["matvec"]["opencl"][i] for i in opencl_x]
plt.plot(opencl_x, opencl_y, "^-")

numba_x = sorted(list(data["matvec"]["numba"].keys()))
numba_y = [data["matvec"]["numba"][i] for i in numba_x]
plt.plot(numba_x, numba_y, "s-")

plt.legend(["ExaFMM", "OpenCL", "Numba"])
plt.xscale("log")
plt.xlabel("$h$")
plt.xlim(plt.xlim()[::-1])
plt.ylabel("Time for 20 matvecs (s)")

plt.savefig("../performance_matvec.png")

plt.clf()
