import scipy.io as sio
import matplotlib.pyplot as plt
import glob
import os
from matplotlib import colormaps as cmp
import re
import numpy as np
from scipy.optimize import curve_fit
from helperFunctions import gain_sat_r_gauss, power_curve, power_curve_digit
import pandas as pd

location = r"D:\polybox"
paths = [
    r"\Semesterarbeit\SV167-b2-14C-2050mm",
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
    r"\Semesterarbeit\SV167-b2-10C-2070nm",
    r"\Semesterarbeit\SV172-Dia-15C-2070nm",
    r"\Semesterarbeit\SV176-a3-15C-2070nm",
    # r"P:\Semesterarbeit\SV167-b5-2-10C-2070nm_bs",
]

for path in paths:
    path = location + path
    files = glob.glob(path + r"\*.mat")

    files = sorted(files, key=lambda fn: int(re.findall(r"\d+", fn[-9:])[0]))

    fit_range = (2e-2, 150)

    maxR = 0

    curvFig, curvAx = plt.subplots(1)
    maxFig, maxAx = plt.subplots(1)

    paramFig, paramAx = plt.subplots(2, 2, squeeze=False, figsize=(6.4 * 2, 4.8 * 2))
    paramAx = paramAx.flatten()
    paramShorts = ["$F_{sat}$", "$R_{ss}$", "$R_{rns}$", "$F_2$"]
    paramNames = [
        r"Saturation fluence [$\mu J/cm^2$]",
        r"Small signal reflectivity [%]",
        r"Nonsaturable Reflectivity [%]",
        r"Rollover [$\mu J / cm^2$]",
    ]
    fit_params = []

    for i, file in enumerate(files):
        fname = os.path.splitext(file)[0]
        (title, temp, wavelength, current) = fname.split("_")[-4:]
        title = "-".join(title.split("-")[-2:])
        print(i, title, temp, current, wavelength)
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
        cmap = cmp.get_cmap("rainbow")
        color = cmap(i / len(files))

        power = f"{power_curve_digit(current):.1f} W"

        if i == 0:
            fit = curve_fit(
                gain_sat_r_gauss,
                fit_fl,
                fit_r,
                p0=(4.7, 102, 99.5, 2200),
                bounds=([0, 80, 80, 0], [20, 120, 101, 1e4]),
                sigma=fit_err,
                absolute_sigma=True,
            )
            fitp = fit[0]
            rns = fitp[2]
        else:

            def fit_func(fl, fsat, rss, f2):
                return gain_sat_r_gauss(fl, fsat, rss, rns, f2)

            fit = curve_fit(
                fit_func,
                fit_fl,
                fit_r,
                p0=(4.7, 102, 2200),
                bounds=([0, 80, 0], [20, 120, 1e4]),
                sigma=fit_err,
                absolute_sigma=True,
            )
            fitp = fit[0]
            fitp = np.insert(fitp, 2, rns)
        print(fitp)
        fit_params.append([current, *fitp])
        curvAx.errorbar(fluence, refl, yerr=error, label=power, c=color, ls="")
        curvAx.plot(fluence, gain_sat_r_gauss(fluence, *fitp), color=color)
        maxAx.scatter(power_curve(current), maxRi, color=color)
        # maxAx.scatter(power, refl[3], color=color)
        # np.savetxt(fname + "_fitp.csv", fitp, delimiter=',', header= ','.join(paramShorts))
        for p, par in enumerate(fitp):
            paramAx[p].scatter(power_curve(current), par, color=color, label=power)

    curvAx.set_xscale("log")
    curvAx.set_ylim(89, 110)
    curvAx.set_xlabel(r"Fluence [$\mu J / cm^2$]")
    curvAx.set_ylabel("Reflectivity [%]")
    curvAx.set_title(
        f"Gain saturation curves for {title} @ {temp} & {wavelength} \n max Reflectivity of {maxR:.1f}"
    )
    curvAx.legend(ncol=2, loc=1)

    maxAx.set_xlabel("Pump power")
    maxAx.set_ylabel("max Reflectivity")
    maxAx.set_title(
        f"Maximum reflectivity vs pump power for {title} @ {temp} & {wavelength}"
    )
    locs = [1, 4, 4, 1]
    for ind, pAx in enumerate(paramAx):
        pmean = np.mean(fitp[ind])
        pAx.set_xlabel("Pump power [W]")
        pAx.set_ylabel(paramNames[ind])
        pAx.set_title(
            f"Fit parameter {paramShorts[ind]} for {title} @ {temp} & {wavelength} \n mean value {pmean:.2f}"
        )
        pAx.legend(ncol=2, loc=locs[ind])

    curvFig.tight_layout()
    maxFig.tight_layout()
    paramFig.tight_layout()

    curvFig.savefig(path + r"\gainsat.png", bbox_inches="tight")
    maxFig.savefig(path + r"\maxfig.png", bbox_inches="tight")
    paramFig.savefig(path + r"\paramfig.png", bbox_inches="tight")
    df = pd.DataFrame(fit_params, columns=["Current [$A$]", *paramNames])
    df.to_csv(path + rf"\{title}_{temp}_{wavelength}_fit_parameters.csv", sep=",")
    print(f"Saved {title}")
