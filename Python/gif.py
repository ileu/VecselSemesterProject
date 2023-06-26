import matplotlib
import numpy as np
import scipy.io as sio
import glob

import matplotlib.pyplot as plt


from helperFunctions import make_gif

level = 0.978
rep = 20
gif_mode = True


"""
    file for make the gif and the complete signal trace with the colorbar
"""

def plot_traces(gif=False):
    """ Iterates over all the data traces for plotting the individaul tracces
    """
    dat_files = sorted(
        glob.glob(r"Python\datafiles\*.mat"), key=len
    )
    log = np.logspace(-3.5, 0, 30)
    print(len(dat_files))
    for i, file in enumerate(dat_files):
        plot_trace(file, i, log[i], gif)
        print(i, file)


def plot_trace(file, i, fluence, gif=False):
    """ Plots on of the cycles of the given data trace

    Parameters
    ----------
    file : string
        file path to the data trace
    i : int
        index of the datatrace
    fluence : float
        fluence value for the data trace
    gif : bool, optional
        if the plot is in gif mode or not, by default False
    """
    dat = sio.loadmat(file)["data0"].flatten()


    cmap = matplotlib.colormaps.get_cmap("Blues")
    c = cmap((i + 10) / 40)
    if gif:
        c = "blue"

    dat = (dat - np.min(dat)) / (np.max(dat) - np.min(dat))
    mask = dat > level
    diff_mask = np.flatnonzero((mask[:-1]) & (np.invert(mask[1:])))
    masked_steps = dat.copy()
    masked_steps[np.invert(mask)] = np.nan

    x_steps = np.linspace(0, 10, 800)
    trace = np.zeros_like(dat, dtype=bool)

    test_mask = diff_mask - diff_mask[rep]
    tt_mask = np.logical_and((test_mask >= -250), (test_mask <= 550))

    trace[diff_mask[rep] - 250 : diff_mask[rep] + 550] = True
    plt.plot(x_steps, dat[trace], label="Data trace", color=c)

    if gif:
        plt.hlines(level, -1, 11, "gray", "--", alpha=0.6, label="Trigger level")
        plt.plot(x_steps, masked_steps[trace], "r")
        plt.scatter(
            x_steps[diff_mask[tt_mask] - diff_mask[rep] + 250],
            dat[diff_mask[tt_mask]],
            marker="o",
            color="r",
            s=30,
            zorder=5,
            label="Trigger point",
        )

    plt.xlabel("Time", fontsize=16)
    plt.ylabel("Normalised signal", fontsize=16)

    if gif:
        plt.title(
            "Fluence "
            + rf"${fluence * 1e3:.2g}".replace("+0", "").replace("e", "\cdot10^{")
            + "}$ / µJ cm$^{-2}$"
        )

    plt.xticks([])
    plt.xlim(-0.5, 10.5)
    plt.ylim(-0.05, 1.04)

    if gif:
        plt.legend(framealpha=1, loc=1)
        plt.tight_layout()
        
    # plt.show() # for inspections
    plt.savefig(rf"Python\Images\gif\trace{i}.png")

    if gif:
        plt.gcf().clf()


if __name__ == "__main__":
    plot_traces(gif_mode)

    if gif_mode:
        make_gif(r"Python\Images\gif", "signal.gif")
    else:
        # adds a colorbar to the plot
        norm = matplotlib.colors.LogNorm(vmin=10 ** (-3.5), vmax=1)
        sm = plt.cm.ScalarMappable(cmap="Blues", norm=norm)
        cb = plt.colorbar(sm, ax=plt.gca(), label="Probe fluence / µJ cm$^{-2}$")
        cb.ax.yaxis.label.set(fontsize=16)
        plt.tight_layout
        plt.savefig(
            r"C:\Users\Ueli\Dropbox\ETH\Semsterarbeit\VecselSemesterProject\Python\Images\trace_complete.png"
        )

    print("DONE")
