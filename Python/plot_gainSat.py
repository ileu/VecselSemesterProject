import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.optimize import curve_fit

from helperFunctions import gain_sat_r_gauss, power_curve_digit, fit_func

""" 
    Plots the gain saturation parameters for a single measurement with and without the roll over parameter.
"""

# load files and sort by measurement order
path = r"D:\polybox\Semesterarbeit\SV167-b5-0C-2070nm"

files = glob.glob(path + r"\*.mat")
files = sorted(files, key=lambda fn: int(re.findall(r"\d+", fn[-9:])[0]))

# set fitting range
fit_range = (2e-2, 150)

# prepare plotting
fig: Figure
ax: Axes
fig, ax = plt.subplots(1, figsize=(7, 4))

paramShorts = ["$F_{sat}$", "$R_{ss}$", "$R_{ns}$", "$F_2$"]
paramNames = [
    r"Saturation fluence [$\mu J/cm^2$]",
    r"Small signal reflectivity [%]",
    r"Nonsaturable Reflectivity [%]",
    r"Rollover [$\mu J / cm^2$]",
]

file = files[6]
# split the name to get info of measurements
fname = os.path.splitext(file)[0]
(title, temp, wavelength, current) = fname.split("_")[-4:]
title = "-".join(title.split("-")[-2:])
print(title, temp, current, wavelength)
current = int(current[:-1])

# make temperature more readable
if temp[0] == "n":
    temp = "$-10$ °C"
elif temp[0] == "0":
    temp = "$0$ °C"
elif temp[0] == "R":
    temp = "Room temperature"
else:
    temp = "$10$ °C"

# load data
data = sio.loadmat(file)
refl = data["R"][0]
fluence = data["fluence"][0]
error = data["R_err2"][0]

# make mask for fitting
mask = (fluence >= fit_range[0]) & (fluence <= fit_range[1])

fit_r = refl[mask]
fit_fl = fluence[mask]
fit_err = error[mask]

power = f"{power_curve_digit(current):.1f} W"

# fit the data
fit = curve_fit(
    fit_func,
    fit_fl,
    fit_r,
    p0=(4.7, 102, 100, 2200),
    bounds=([0, 80, 90, 0], [20, 120, 110, 1e4]),
    sigma=fit_err,
    absolute_sigma=True,
)
# assign fitting parameters to fit_params
fit_params = fit[0]

# plotting
ax.errorbar(
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
ax.plot(fluence, gain_sat_r_gauss(fluence, *fit_params), color="b", label="VECSEL")
ax.plot(
    fluence,
    gain_sat_r_gauss(fluence, *fit_params[:-1], f2=np.inf),
    color="b",
    ls="--",
    label="VECSEL without $F_2$",
)
ax.vlines(fit_params[0], 93, 110, ls="dotted")
ax.text(0.5, fit_params[1] + 0.3, "$R_{ss}$", fontsize=20)
ax.text(0.5, fit_params[2] + 0.3, "$R_{ns}$", fontsize=20)
ax.text(fit_params[0] + 0.2, fit_params[2] + 0.7, "$F_{sat}$", fontsize=20)
ax.hlines(fit_params[1:3], 1e-2, 1e3, ls="dotted")

ax.set_xscale("log")
ax.set_ylim(fit_params[2] - 1, fit_params[1] + 1)
ax.set_xlim(0.08, 90)
ax.set_xlabel(r"Probe fluence  / µJ cm$^{-2}$", fontsize=16)
ax.set_ylabel("Reflectivity / %", fontsize=16)
ax.set_title(f"Nonlinear reflectivity for {title} at {temp}", fontsize=18)
ax.tick_params(axis="both", which="major", labelsize=12)
ax.legend(ncol=2, loc=1)

fig.tight_layout()
fig.savefig(r"Python\Images\gainSat.png")
plt.show()
