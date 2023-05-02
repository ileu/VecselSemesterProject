import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import glob
import re
import os

from helperFunctions import power_curve

paths = [
    r"P:\Semesterarbeit\SV167-b2-14C-2050mm",
    r"P:\Semesterarbeit\SV167-b2-10C-2070nm",
    r"P:\Semesterarbeit\SV167-b2-0C-2070nm",
    r"P:\Semesterarbeit\SV167-b2-n10C-2070nm",
    r"P:\Semesterarbeit\SV167-b5-10C-2070nm",
    r"P:\Semesterarbeit\SV167-b5-0C-2070nm",
    r"P:\Semesterarbeit\SV167-b5-n10C-2070nm",
    r"P:\Semesterarbeit\SV166-a4-10C-2070nm",
    r"P:\Semesterarbeit\SV166-a4-0C-2070nm",
    r"P:\Semesterarbeit\SV166-a4-n10C-2070nm",
    r"P:\Semesterarbeit\SV165-CD2-RT-2070nm",
    r"P:\Semesterarbeit\SV165-CD2-10C-2070nm",
    r"P:\Semesterarbeit\SV165-CD2-0C-2070nm",
    r"P:\Semesterarbeit\SV165-CD2-n10C-2070nm",
]

chip1 = [1,2,3]
chip2 = [4,5,6]
currents = [0, 12, 36]

fig, axs = plt.subplot_mosaic([['a)', 'b)', 'c)'], ['a)', 'b)', 'd)']],
                              layout='constrained', figsize=(15, 3))

axa = axs['a)']
axa.set_xscale("log")
axa.set_ylim(89, 110)
axa.set_xlabel("Fluence [$\mu J / cm^2$]")
axa.set_ylabel("Reflectivity [%]")
axa.set_title('SV167-b2 temperature dependence of gain saturation', fontstyle='italic')
    

for chip in chip1:
    path = paths[chip]
    
    mat_files = glob.glob(path + "\*.mat")

    mat_files = sorted(mat_files, key=lambda fn: int(re.findall(r"\d+", fn[-9:])[0]))
    cmap = matplotlib.colormaps.get_cmap("rainbow")
    

    for i, file in enumerate(mat_files):
        fname = os.path.splitext(file)[0]
        (title, temp, wavelength, current) = fname.split("_")[-4:]
        title = "-".join(title.split("-")[-2:])
        current = int(current[:-1])
        if current not in currents:
            continue
        power = f"{power_curve(current):.1f} W"
        
        data = sio.loadmat(file)
        refl = data["R"][0]
        fluence = data["fluence"][0]
        error = data["R_err2"][0]
        
        if temp[0] == 'n':
            temp = '-' + temp[1:-1] + "°C"
        
        axa.errorbar(fluence, refl, yerr=error, label=f"{temp} & {power}", ls="", color = cmap((2 * chip) / 6))

for label, ax in axs.items():
    print(label)
    ax.set_title(label, fontfamily='serif', loc='left', fontsize='medium')

axa.legend()
plt.show()