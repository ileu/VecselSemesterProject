import scipy.io as sio
import matplotlib.pyplot as plt
import glob
import os
from matplotlib import colormaps as cmp
import re
import numpy as np
from scipy.optimize import curve_fit
from helperFunctions import gain_sat_r_gauss, paths, power_curve_digit, save_figure
import pandas as pd

"""
    processes the data
    creates & saves the plot and the fitting parameter in the datafile location
"""

location = r"D:\polybox"

for path in paths:
    path = location + path
    files = glob.glob(path + r"\*.mat")

    files = sorted(files, key=lambda fn: int(re.findall(r"\d+", fn[-9:])[0]))

    fit_range = (2e-2, 150)

    maxR = 0
    minR = 120

    curvFig, curvAx = plt.subplots(1)
    maxFig, maxAx = plt.subplots(1)

    paramFig, paramAx = plt.subplots(2, 2, squeeze=False, figsize=(6.4 * 2, 4.8 * 2))
    paramAx = paramAx.flatten()
    paramShorts = ["$F_{sat}$", "$R_{ss}$", "$R_{ns}$", "$F_2$"]
    paramNames = [
        r"Saturation fluence [$\mu J/cm^2$]",
        r"Small signal reflectivity [%]",
        r"Nonsaturable Reflectivity [%]",
        r"Rollover [$\mu J / cm^2$]",
    ]
    fit_params = []

    for i, file in enumerate(files):
        fname = os.path.splitext(file)[0]
        (title, temp_name, wavelength, current) = fname.split("_")[-4:]
        title = "-".join(title.split("-")[-2:])
        print(i, title, temp_name, current, wavelength)
        current = int(current[:-1])
        data = sio.loadmat(file)
        refl = data["R"][0]
        fluence = data["fluence"][0]
        error = data["R_err2"][0]

        mask = (fluence >= fit_range[0]) & (fluence <= fit_range[1])

        fit_r = refl[mask]
        fit_fl = fluence[mask]
        fit_err = error[mask]

        minRi = refl[0]
        if minRi < minR:
            minR = minRi

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

        fit_params.append([current, *fitp])
        curvAx.errorbar(fluence, refl, yerr=error, label=power, c=color, ls="")
        curvAx.plot(fluence, gain_sat_r_gauss(fluence, *fitp), color=color)
        maxAx.scatter(power_curve_digit(current), maxRi, color=color)
        # maxAx.scatter(power, refl[3], color=color)
        # np.savetxt(fname + "_fitp.csv", fitp, delimiter=',', header= ','.join(paramShorts))
        for p, par in enumerate(fitp):
            paramAx[p].scatter(
                power_curve_digit(current), par, color=color, label=power
            )

    if temp_name[0] == "n":
        temp = "$-10$ °C"
    elif temp_name[0] == "0":
        temp = "$0$ °C"
    elif temp_name[0] == "R":
        temp = "Room temperature"
    else:
        temp = "$10$ °C"

    curvAx.set_xscale("log")
    curvAx.set_xlim(curvAx.get_xlim()[0], 75)
    curvAx.set_ylim(minR * 0.98, maxR * 1.02)
    # curvAx.set_ylim(98.7, 102.2)
    curvAx.tick_params(axis="both", which="major", labelsize=10)
    curvAx.set_xlabel(r"Probe fluence  / µJ cm$^{-2}$", fontsize=14)
    curvAx.set_ylabel("Reflectivity / %", fontsize=14)
    # curvAx.set_title(f"Nonlinear reflectivity for {title} at {temp}", fontsize=16)
    curvAx.legend(ncol=2, loc=1, fontsize=10)

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

    save_figure(curvFig, path + r"\gainsat.png")
    save_figure(maxFig, path + r"\maxfig.png")
    save_figure(paramFig, path + r"\paramfig.png")

    df = pd.DataFrame(fit_params, columns=["Current [$A$]", *paramNames])
    df.to_csv(path + rf"\{title}_{temp_name}_{wavelength}_fit_parameters.csv", sep=",")
    print(f"Saved {title}")
