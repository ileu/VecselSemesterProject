import numpy as np
import scipy.io as sio
import glob

import matplotlib.pyplot as plt

from helperFunctions import make_gif

level = 0.978
rep = 20


def eformat(f, prec, exp_digits):
    s = "%.*e" % (prec, f)
    mantissa, exp = s.split("e")
    # add 1 to digits as 1 is taken by sign +/-
    return "%se%+0*d" % (mantissa, exp_digits + 1, int(exp))


def plot_traces():
    dat_files = sorted(glob.glob(r"datafiles\*.mat"), key=len)
    log = np.logspace(-3.5, 0, 30)

    for i, file in enumerate(dat_files):
        plot_trace(file, i, log[i])
        print(i, file)


def plot_trace(file, i, fluence):
    dat = sio.loadmat(file)["data0"].flatten()

    # steps = three_step(x_steps) + sin_noise(x_steps, 2, 2e-3, 2e-3)
    # plt.plot(x_steps, steps)

    dat = (dat - np.min(dat)) / (np.max(dat) - np.min(dat))
    mask = dat > level
    diff_mask = np.flatnonzero((mask[:-1]) & (np.invert(mask[1:])))
    masked_steps = dat.copy()
    masked_steps[np.invert(mask)] = np.nan
    # slope_mask = np.arange(diff_mask[-1] - 10, diff_mask[-1] + 10)
    x_steps = np.linspace(0, 10, 800)
    trace = np.zeros_like(dat, dtype=bool)
    test_mask = diff_mask - diff_mask[rep]
    tt_mask = np.logical_and((test_mask >= -250), (test_mask <= 550))
    trace[diff_mask[rep] - 250 : diff_mask[rep] + 550] = True
    plt.plot(x_steps, dat[trace], label="Data trace")
    plt.hlines(level, -1, 11, "gray", "--", alpha=0.6, label="Trigger level")
    # plt.plot(x_steps, masked_steps[trace], "r")
    plt.scatter(
        x_steps[diff_mask[tt_mask] - diff_mask[rep] + 250],
        dat[diff_mask[tt_mask]],
        marker="o",
        color="r",
        s=30,
        zorder=5,
        label="Trigger point",
    )
    plt.xlabel("Time")
    plt.ylabel("Normalised signal")
    plt.title(
        "Fluence "
        + rf"${fluence * 1e3:.2g}".replace("e", "\cdot10^{")
        + "}$[$mJ/cm^2$]"
    )
    plt.xticks([])
    plt.xlim(-0.5, 10.5)
    plt.ylim(-0.05, 1.04)
    plt.legend(framealpha=1, loc=1)
    plt.tight_layout()
    # plt.show()
    plt.savefig(rf"Images\gif\trace{i}.png")
    plt.gcf().clf()


if __name__ == "__main__":
    plot_traces()
    make_gif(r"Images\gif", "my_awesome.gif")
