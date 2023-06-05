# %%
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.optimize import curve_fit

from helperFunctions import gain_sat_r_gauss, power_curve_digit


def fit_func(fl, fsat, rss, rns, f2):
    return gain_sat_r_gauss(fl, fsat, rss, rns, f2)


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
    r"\Semesterarbeit\SV165-CD2-RT-2070nm",
    r"\Semesterarbeit\SV165-CD2-10C-2070nm",
    r"\Semesterarbeit\SV165-CD2-0C-2070nm",
    r"\Semesterarbeit\SV165-CD2-n10C-2070nm",
    r"\Semesterarbeit\SV172-Dia-15C-2070nm",
    r"\Semesterarbeit\SV176-a3-15C-2070nm",
    # r"P:\Semesterarbeit\SV167-b5-2-10C-2070nm_bs",
]

path = location + paths[5]
files = glob.glob(path + r"\*.mat")

files = sorted(files, key=lambda fn: int(re.findall(r"\d+", fn[-9:])[0]))

fit_range = (2e-2, 150)

maxR = 0
curvFig: Figure
curvAx: Axes
curvFig, curvAx = plt.subplots(1, figsize=(7, 4))

paramShorts = ["$F_{sat}$", "$R_{ss}$", "$R_{ns}$", "$F_2$"]
paramNames = [
    r"Saturation fluence [$\mu J/cm^2$]",
    r"Small signal reflectivity [%]",
    r"Nonsaturable Reflectivity [%]",
    r"Rollover [$\mu J / cm^2$]",
]

file = files[6]
fname = os.path.splitext(file)[0]
(title, temp, wavelength, current) = fname.split("_")[-4:]
title = "-".join(title.split("-")[-2:])
print(title, temp, current, wavelength)
current = int(current[:-1])

if temp[0] == "n":
    temp = "$-10$ °C"
elif temp[0] == "0":
    temp = "$0$ °C"
elif temp[0] == "R":
    temp = "Room temperature"
else:
    temp = "$10$ °C"

data = sio.loadmat(file)
refl = data["R"][0]
fluence = data["fluence"][0]
error = data["R_err2"][0]

mask = (fluence >= fit_range[0]) & (fluence <= fit_range[1])

fit_r = refl[mask]
fit_fl = fluence[mask]
fit_err = error[mask]

maxRi = max(refl)
if maxRi > maxR:
    maxR = maxRi

power = f"{power_curve_digit(current):.1f} W"

fit = curve_fit(
    fit_func,
    fit_fl,
    fit_r,
    p0=(4.7, 102, 100, 2200),
    bounds=([0, 80, 90, 0], [20, 120, 110, 1e4]),
    sigma=fit_err,
    absolute_sigma=True,
)
fitp = fit[0]
# fitp = np.insert(fitp, 2, rns)
print(fitp)
curvAx.errorbar(
    fluence,
    refl,
    yerr=error,
    ls="",
    color="b",
    marker="o",
    markerfacecolor="red",
    markeredgecolor="None",
    markersize=6,
)
curvAx.plot(fluence, gain_sat_r_gauss(fluence, *fitp), color="b", label="VECSEL")
curvAx.plot(
    fluence,
    gain_sat_r_gauss(fluence, *fitp[:-1], f2=np.inf),
    color="b",
    ls="--",
    label="VECSEL without $F_2$",
)
curvAx.vlines(fitp[0], 93, 110, ls="dotted")
curvAx.text(0.5, fitp[1] + 0.3, "$R_{ss}$", fontsize=20)
curvAx.text(0.5, fitp[2] + 0.3, "$R_{ns}$", fontsize=20)
curvAx.text(fitp[0] + 0.2, fitp[2] + 0.7, "$F_{sat}$", fontsize=20)
curvAx.hlines(fitp[1:3], 1e-2, 1e3, ls="dotted")

curvAx.set_xscale("log")
curvAx.set_ylim(fitp[2] - 1, fitp[1] + 1)
curvAx.set_xlim(0.08, 90)
curvAx.set_xlabel(r"Probe fluence  / µJ cm$^{-2}$", fontsize=16)
curvAx.set_ylabel("Reflectivity / %", fontsize=16)
curvAx.set_title(f"Nonlinear reflectivity for {title} at {temp}", fontsize=18)
curvAx.tick_params(axis="both", which="major", labelsize=12)
curvAx.legend(ncol=2, loc=1)

locs = [1, 4, 4, 1]
curvFig.tight_layout()
curvFig.savefig(r"VecselSemesterProject\Python\Images\gainSat.png")
plt.show()
