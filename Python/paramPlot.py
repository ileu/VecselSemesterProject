import matplotlib
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
    # r"\Semesterarbeit\SV167-b2-14C-2050mm",
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
    # r"\Semesterarbeit\SV172-Dia-15C-2070nm",
    # r"\Semesterarbeit\SV176-a3-15C-2070nm",
    # r"P:\Semesterarbeit\SV167-b5-2-10C-2070nm_bs",
]

paramFig: matplotlib.figure.Figure
paramAx: matplotlib.axes.Axes
paramFig, paramAx = plt.subplots(2, 2, squeeze=False, figsize=(2 * 6.4, 2 * 4.8))
paramAx = paramAx.flatten()
paramShorts = ["$F_{sat}$", "$R_{ss}$", "$R_{ns}$", "$F_2$"]
paramNames = [
    r"Saturation fluence / µJ cm$^{-2}$",
    r"Small signal reflectivity / % ",
    r"Nonsaturable Reflectivity / %",
    r"Rollover / mJ cm$^{-2}$",
]

std = 0
mn = 0


for ind, path in enumerate(paths):
    path = location + path
    file = glob.glob(path + r"\*fit_parameters.csv")[0]

    fname = os.path.splitext(file)[0]
    fname = fname.split("\\")[-1]
    (title, temp, wavelength) = fname.split("_")[:-2]

    # title = "diamond, pump DBR"

    df = pd.read_csv(file)
    df["Power / $W$"] = power_curve(df["Current [$A$]"])
    df.rename(
        columns={
            r"Saturation fluence [$\mu J/cm^2$]": r"$F_{sat}$ / $\mu Jcm^{-2}$",
            r"Nonsaturable Reflectivity [%]": r"$R_{ns}$ / \%",
            r"Small signal reflectivity [%]": r"$R_{ss}$ / \%",
            r"Rollover [$\mu J / cm^2$]": r"$F_2$ / $\mu Jcm^{-2}$",
        },
        inplace=True
    )
    cols = df.columns.to_list()
    df = df[cols[-1:] + cols[:-1]]
    old_temp = temp
    old_title = title

    if temp[0] == "n":
        temp = "$-10$ °C"
        color = "blue"
    elif temp[0] == "0":
        temp = "$0$ °C"
        color = "green"
    elif temp[0] == "R":
        temp = "Room temperature"
        color = "orange"
    else:
        temp = "$10$ °C"
        color = "red"

    if title == "SV165-CD2":
        title = "hybrid"
    elif title == "SV166-a4":
        title = "no pump DBR"
    elif title == "SV167-b5":
        title = "copper, pump DBR"
    elif title == "SV167-b2":
        title = "diamond, pump DBR"
    
    # print(df.columns)
    df.to_latex(
        rf"VecselSemesterProject\Latex\tables\{old_title}-{old_temp}-table.tex",
        columns=[
            "Power / $W$",
            *df.columns[3:7]
        ],
        index=False,
        float_format="{:.7}".format,
        formatters={"Power / $W$": "{:.1f}".format},
        caption=rf"Fitting parameters for {title} at {temp}",
        position="H",
    )

    print()
    print(title, temp)

    # print(df)
    for n, pAx in enumerate(paramAx):
        x_data = df["Power / $W$"]
        if n == 3:
            data = df.iloc[:, 3 + n] * 1e-3
            lin = np.polyfit(x_data[3:-1], data[3:-1], 1)
            print(f"F: {lin[0]*1e3:.2f}\t{lin[1]:.2f}")
            pAx.plot(
                x_data,
                np.poly1d(lin)(x_data),
                color="gray",
                ls="--",
                alpha=0.8,
                zorder=2,
            )
            # print(np.polyfit(x_data, data, 1))
        else:
            data = df.iloc[:, 3 + n]
            print(np.max(data))
        pAx.plot(
            x_data,
            data,
            marker="o",
            ls="--",
            label=f"{temp}",
            color=color,
            linewidth=1.75,
            markersize=10,
            zorder=5,
        )
        if n == 1:
            print("Rmax", np.max(data))
            pAx.set_ylim(97.9, 106.8)
            lin, cov = np.polyfit(x_data[:4], data[:4], 1, cov=True)
            # print(lin)
            pAx.plot(
                x_data,
                np.poly1d(lin)(x_data),
                color="gray",
                ls="--",
                alpha=0.8,
                zorder=2,
            )
            # pAx.plot(x_data[:4], data[:4], color="black", zorder=200)

            # pAx.fill_between(
            #     x_data,
            #     np.poly1d(lin + np.sqrt(np.diag(cov)))(x_data),
            #     np.poly1d(lin - np.sqrt(np.diag(cov)))(x_data),
            #     color="red",
            #     alpha=0.5,
            # )
            mark_prep = (
                np.poly1d(lin)(x_data) - data - np.poly1d(np.sqrt(np.diag(cov)))(x_data)
            )

            def calc_distance(m, b, p3):
                p1 = np.array([0, b])

                p2 = np.array([10, 10 * m + b])
                temp = np.abs(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
                return temp

            mark_prep = [calc_distance(*lin, [x_data[x], y]) for x, y in enumerate(data)]

            # print(mark_prep)
            mark = len(x_data) - next(
                x for x, val in enumerate(mark_prep[::-1]) if val < 0.4
            )

            mark = max(mark - 1, 0)
            print("MARK", x_data[mark])
            pAx.scatter(
                x_data[mark],
                data[mark],
                facecolors="none",
                edgecolors=color,
                s=600,
                zorder=15,
                linewidths=4,
            )

        if n == 0:
            lin = np.mean(data[4:-1])
            std = np.std(data[4:-1])
            print("Mean Fsat", lin, std)
        elif n==2:
            print("Non Sat", data[1])
    # plt.show()
    # print(df.iloc[:, 4])
    if (ind % 3 == 2 and ind + 2 != len(paths)) or ind + 1 == len(paths):
        for k, pAx in enumerate(paramAx):
            if k == 0:
                if ind + 1 != len(paths):
                    pAx.set_ylim(pAx.get_ylim()[0], 5.35)
                else:
                    pAx.set_ylim(pAx.get_ylim()[0], 6.35)

            pAx.tick_params(axis="both", which="major", labelsize=14)
            # pAx.tick_params(axis='both', which='minor', labelsize=11)
            pAx.set_xlabel("Pump power / W", fontsize=16)
            pAx.set_ylabel(paramNames[k], fontsize=16)
            # pAx.set_title(f"Fit parameter {paramShorts[k]} for {title}", fontsize=20)
            if ind + 1 != len(paths):
                pAx.legend(fontsize=16)
            else:
                pAx.legend(fontsize=16, ncols=2)
            paramFig.tight_layout()
            if ind == 5:
                extent = pAx.get_tightbbox().transformed(
                    paramFig.dpi_scale_trans.inverted()
                )

                paramFig.savefig(
                    rf"VecselSemesterProject\Latex\images\param{k}.png", bbox_inches=extent.expanded(1.03, 1.03), dpi=200
                )
        # plt.show()
        # break
        paramFig.savefig(rf"VecselSemesterProject\Latex\images\{title.replace(' ', '-').replace(',','')}_paramfig.png", bbox_inches="tight", dpi=200)
        print("Saved")

        std = 0
        mean = 0

        for pAx in paramAx:
            pAx.clear()
