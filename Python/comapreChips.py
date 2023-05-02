import glob
import os
import matplotlib
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd

from helperFunctions import power_curve

location = r"D:\polybox"
paths = [
    r"\Semesterarbeit\SV167-b2-14C-2050mm",
    r"\Semesterarbeit\SV167-b2-10C-2070nm",
    r"\Semesterarbeit\SV167-b2-0C-2070nm",
    r"\Semesterarbeit\SV167-b2-n10C-2070nm",
    r"\Semesterarbeit\SV167-b5-10C-2070nm",
    r"\Semesterarbeit\SV167-b5-0C-2070nm",
    r"\Semesterarbeit\SV167-b5-n10C-2070nm",
    r"\Semesterarbeit\SV166-a4-10C-2070nm",
    r"\Semesterarbeit\SV166-a4-0C-2070nm",
    r"\Semesterarbeit\SV166-a4-n10C-2070nm",
    r"\Semesterarbeit\SV165-CD2-10C-2070nm",
    r"\Semesterarbeit\SV165-CD2-0C-2070nm",
    r"\Semesterarbeit\SV165-CD2-n10C-2070nm",
    r"\Semesterarbeit\SV165-CD2-RT-2070nm",
]


chip1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]
# currents = [0, 12, 36]

# fig, axs = plt.subplot_mosaic([['a)'], ['b)']],
#                               layout='constrained', figsize=(8, 3))

# axa = axs['a)']
# axb = axs['b)']
# axa.set_xscale("log")
# axa.set_ylim(89, 110)
# axa.set_ylabel("Saturation fluence [$\mu J/cm^2$]")
# axa.set_xlabel("Pump power [W]")
# axa.set_title('Saturation fluence of SV167 vs temp', fontstyle='italic')

# axb.set_ylabel("Rollover [$\mu J/cm^2$]")
# axb.set_xlabel("Pump power [W]")
# axb.set_title('Saturation fluence of SV167 vs temp', fontstyle='italic')
# handles =  []

fig: Figure
ax: Axes
fig, ax = plt.subplots()
cind = 4
sname = f"Comp_b2_b5_{cind}"

cmap = matplotlib.colormaps.get_cmap("brg")


for chip in chip1:
    path = location + paths[chip]

    file = glob.glob(path + r"\*.csv")[0]

    fname = os.path.basename(file)
    (title, temp, wavelength) = fname.split("_")[:-2]

    fit_p: pd.DataFrame = pd.read_csv(file)
    if "n10C" in fname:
        # print(fname)
        color = cmap(0)
    elif "10C" in fname:
        color = cmap(0.5)
    else:
        # print(fname)
        color = cmap(1.0)

    ax.set_xlabel("Pump power [$W$]")
    ax.set_ylabel(fit_p.columns[cind])

    if chip in [4, 5, 6]:
        setting = {"ls": "--", "marker": "o", "mfc": color}
    elif chip in [1, 2, 3]:
        setting = {"ls": "--", "marker": "d", "mfc": color}
    elif chip in [7, 8, 9]:
        setting = {"ls": "--", "marker": "o", "mfc": "None", "markeredgecolor": color, "markeredgewidth": 2}
    elif chip in [10, 11, 12]:
        setting = {"ls": "--", "marker": "+", "mfc": color, "markeredgewidth": 7}
        
    plt.plot(
        power_curve(fit_p.iloc[:, 1].to_numpy()),
        fit_p.iloc[:, cind],
        label=f"{title} @ {temp}",
        color=color,
        markersize=12,
        **setting,
    )

# plt.ylim(-0.6, 7.5)
# plt.legend(loc=1, framealpha=1)
plt.tight_layout()
# plt.savefig(rf"Images\{sname}.png")
plt.show()
