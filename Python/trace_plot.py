import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


"""
    This file plots the old triggering method for the signal trace
"""

# load the data of a measurement containing about 200 cycles
data = sio.loadmat(f"Python\datafiles\data_13.mat")["data0"]

# select a single cycle from the measurement, which is about 800 data points, and normalize
x_steps = np.linspace(0, 10, 800)

data = data[200:1000]
data = (data - np.min(data)) / (np.max(data) - np.min(data))
data = data.flatten()

# prepare mask for triggering
mask = data > 0.97
diff_mask = np.flatnonzero((mask[:-1]) & (np.invert(mask[1:])))

# points above triggerlevel
masked_steps = data.copy()
masked_steps[np.invert(mask)] = np.nan


# plotting
plt.plot(x_steps, data, label="Data trace")

plt.xlabel("Time")
plt.ylabel("Normalised signal")
plt.xticks([])
plt.xlim(-0.5, 10.5)
plt.legend(framealpha=1)
plt.tight_layout()
plt.savefig(r"Python\Images\trace_signal.png")

plt.hlines(0.97, -1, 11, "gray", "--", alpha=0.8, label="Trigger level")

plt.legend(framealpha=1)
plt.tight_layout()
plt.savefig(r"Python\Images\trace_level.png")

plt.plot(x_steps, masked_steps, "r")

plt.legend(framealpha=1)
plt.tight_layout()
plt.savefig(r"Python\Images\trace_trigger_level.png")

plt.scatter(
    x_steps[diff_mask],
    data[diff_mask],
    marker="o",
    color="r",
    s=30,
    zorder=5,
    label="Trigger point",
)


# Color different trigger levels differently (not used)

# atol = 1e-3

# level = np.isclose(np.diff(data), 0, atol=atol)

# levelDat = level * data[:-1]
# levelDat[levelDat == 0] = np.nan

# for s in np.ma.clump_unmasked(np.ma.masked_invalid(levelDat))[1:-1]:
#     nan = np.full_like(levelDat, np.nan)
#     nan[s] = levelDat[s]
#     plt.plot(x_steps[:-1], nan, linewidth=4)


plt.legend(framealpha=1)
plt.tight_layout()
plt.savefig(r"Python\Images\trace_trigger.png")
plt.show()
