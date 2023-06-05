import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


def sin_noise(x, w, amp, noise):
    return np.sin(w * x) * amp + np.random.normal(0, noise, x.size)


def step_up(x, k, a):
    return 1 / (1 + np.exp(-k * x)) ** a


def step(x, k, a, dx):
    return step_up(x, k, a) - step_up(x - dx, k, a)


def step_moved(x, dx, k, a, dx2):
    return step(x - dx, k, a, dx2)


def three_step(x):
    return (
        0.7 * step(x, 5, 4, 10)
        + 0.3 * step_moved(x, 5, 5, 4, 10)
        + 0.4 * step_moved(x, 20, 5, 4, 5)
    )


dat = sio.loadmat(f"VecselSemesterProject\Python\datafiles\data_13.mat")["data0"]

x_steps = np.linspace(0, 10, 800)
level = 0.8
# steps = three_step(x_steps) + sin_noise(x_steps, 2, 2e-3, 2e-3)
# plt.plot(x_steps, steps)

dat = dat[200:1000] / np.max(dat)
mask = dat > level
diff_mask = np.flatnonzero((mask[:-1]) & (np.invert(mask[1:])))
masked_steps = dat.copy()
masked_steps[np.invert(mask)] = np.nan

slope_mask = np.arange(diff_mask[-1] - 10, diff_mask[-1] + 10)
# plt.plot(x_test, masked_steps, "r")


plt.plot(x_steps, dat, label="Data trace")

plt.hlines(level, -1, 11, "gray", "--", alpha=0.8, label="Trigger level")

plt.scatter(
    x_steps[diff_mask],
    dat[diff_mask],
    marker="o",
    color="r",
    s=30,
    zorder=5,
    label="Trigger point",
)

plt.xlabel("Time")
plt.ylabel("Normalised signal")
plt.xticks([])
plt.xlim(-0.5, 10.5)
plt.ylim(-0.053217850471577025, 1.0410437465530953)

plt.legend()
plt.tight_layout()
plt.savefig("VecselSemesterProject\Python\Images\corrected_signal_1.png")

plt.scatter(
    x_steps[slope_mask],
    dat[slope_mask],
    marker=".",
    color="red",
    s=30,
    zorder=5,
    label="Slope detection",
)

plt.legend()
plt.tight_layout()
plt.savefig("VecselSemesterProject\Python\Images\corrected_signal_2.png")

plt.fill_between(
    x_steps[(diff_mask[-1] - 50) : (diff_mask[-1])],
    -1,
    1.1,
    color="orange",
    alpha=0.3,
    edgecolor="None",
    label="Peak detection",
)

plt.legend()
plt.tight_layout()

plt.savefig("VecselSemesterProject\Python\Images\corrected_signal_3.png")
plt.show()
