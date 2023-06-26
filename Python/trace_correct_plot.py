import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


"""
    This file plots the new triggering method for the signal trace
"""

# load the data of a measurement containing about 200 cycles
data = sio.loadmat(f"Python\datafiles\data_13.mat")["data0"]

# set triggering level
level = 0.8

# select a single cycle from the measurement, which is about 800 data points, and normalize
x_steps = np.linspace(0, 10, 800)

data = data[200:1000]
data = (data - np.min(data)) / (np.max(data) - np.min(data))
data = data.flatten()

# prepare mask for triggering
mask = data > level
diff_mask = np.flatnonzero((mask[:-1]) & (np.invert(mask[1:])))

# points above triggerlevel
masked_steps = data.copy()
masked_steps[np.invert(mask)] = np.nan

# take the last triggering point and select the 20 points around it for the slope detection
slope_mask = np.arange(diff_mask[-1] - 10, diff_mask[-1] + 10)


# plotting

plt.plot(x_steps, data, label="Data trace")

plt.hlines(level, -1, 11, "gray", "--", alpha=0.8, label="Trigger level")

plt.scatter(
    x_steps[diff_mask],
    data[diff_mask],
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
plt.savefig("Python\Images\corrected_signal_1.png")

plt.scatter(
    x_steps[slope_mask],
    data[slope_mask],
    marker=".",
    color="red",
    s=30,
    zorder=5,
    label="Slope detection",
)

plt.legend()
plt.tight_layout()
plt.savefig("Python\Images\corrected_signal_2.png")

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

plt.savefig("Python\Images\corrected_signal_3.png")
plt.show()
