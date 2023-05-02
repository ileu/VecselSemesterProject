import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from helperFunctions import gain_sat_r_gauss


def sin_noise(x, w, amp, noise):
    return np.sin(w * x) * amp + np.random.normal(0, noise, x.size)


def step_up(x, k, a):
    return 1 / (1 + np.exp(-k * x)) ** a


def step(x, k, a, dx):
    return step_up(x, k, a) - step_up(x - dx, k, a)


def step_moved(x, dx, k, a, dx2):
    return step(x - dx, k, a, dx2 - dx)


def three_step(x):
    k = 20
    a = 2
    return (
        0.7 * step_moved(x, 0.65, k, a, 3.4)
        + 0.3 * step_moved(x, 2.6, k, a, 5.4)
        + 0.4 * step_moved(x, 6.4, k, a, 9)
    )


dat = sio.loadmat(f"datafiles\data_13.mat")["data0"]

x_steps = np.linspace(0, 10, 800)
# steps = three_step(x_steps) + sin_noise(x_steps, 2, 2e-3, 2e-3)
# plt.plot(x_steps, steps)

dat = dat[200:1000] / np.max(dat)
mask = dat > 0.97
diff_mask = np.flatnonzero((mask[:-1]) & (np.invert(mask[1:])))
masked_steps = dat.copy()
masked_steps[np.invert(mask)] = np.nan


plt.plot(x_steps, dat, label="Data trace")

plt.xlabel("Time")
plt.ylabel("Normalised signal")
plt.xticks([])
plt.xlim(-.5,10.5)
plt.legend(framealpha=1)
plt.tight_layout()
plt.savefig(r"Images\trace1.png")

plt.hlines(0.97, -1, 11, "gray", "--", alpha=0.3, label="Trigger level")

plt.legend(framealpha=1)
plt.tight_layout()
plt.savefig(r"Images\trace2.png")

plt.plot(x_steps, masked_steps, "r")

plt.legend(framealpha=1)
plt.tight_layout()
plt.savefig(r"Images\trace3.png")

plt.scatter(x_steps[diff_mask], dat[diff_mask], marker="o", color="r", s=30, zorder=5, label="Trigger point")


# log = np.logspace(-3.5, 0, 30)
# plt.plot(log, gain_sat_r_gauss(log, 4, 103, 99, np.inf))

plt.legend(framealpha=1)
plt.tight_layout()
plt.savefig(r"Images\old_trace.png")
plt.show()
