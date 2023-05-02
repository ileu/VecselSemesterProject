import matplotlib.pyplot as plt

# %%
import scipy.io as sio
import matplotlib.pyplot as plt
import glob
import os
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import re
import numpy as np
from scipy.optimize import curve_fit
from helperFunctions import gain_sat_r_gauss, power_curve, power_curve_digit




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

path = location + paths[2]
files = glob.glob(path + r"\*.mat")

files = sorted(files, key=lambda fn: int(re.findall(r"\d+", fn[-9:])[0]))

fit_range = (2e-2, 150)

maxR = 0
curvFig: Figure
curvAx: Axes
curvFig, curvAx = plt.subplots(1, figsize=(7, 4))

paramShorts = ["$F_{sat}$", "$R_{ss}$", "$R_{rns}$", "$F_2$"]
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
curvAx.errorbar(fluence, refl, yerr=error, ls="", color="b")
curvAx.plot(fluence, gain_sat_r_gauss(fluence, *fitp), color="b",label="VECSEL")
curvAx.plot(fluence, gain_sat_r_gauss(fluence, *fitp[:-1], f2=np.inf), color="b", ls='--', label="VECSEL without $F_2$")
curvAx.vlines(fitp[0], 93, 110, ls='dotted')
curvAx.text(.5, 99.2, "$R_{ns}$", fontsize=20)
curvAx.text(.5, 106.5, "$R_{SS}$", fontsize=20)
curvAx.text(fitp[0]+0.2, 100.7, "$F_{sat}$", fontsize=20)
curvAx.hlines(fitp[1:3], 1e-2,1e3,ls="dotted")

curvAx.set_xscale("log")
curvAx.set_ylim(99, 107.3)
curvAx.set_xlim(0.08,90)
curvAx.set_xlabel(r"Fluence [$\mu J / cm^2$]")
curvAx.set_ylabel("Reflectivity [%]")
curvAx.set_title(
    f"Gain saturation curve for {title} @ {temp} & {wavelength}"
)
curvAx.legend(ncol=2, loc=1)

locs = [1, 4, 4, 1]
curvFig.tight_layout()
curvFig.savefig(r"Images\gainSat.png")
plt.show()
