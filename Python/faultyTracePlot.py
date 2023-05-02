import numpy as np
import matplotlib.pyplot as plt


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
        0.9 * step(x_test, 5, 4, 10)
        + 0.15 * step_moved(x_test, 5, 5, 4, 10)
        + 0.8 * step_moved(x_test, 20, 5, 4, 5) -0.05
    )


x_test = np.linspace(-1, 27, 10000)
steps =  three_step(x_test) + sin_noise(x_test, 6, 1e-2, 2e-4)
steps = steps/max(steps)
mask = steps > 0.99
diff_mask = np.flatnonzero((mask[:-1]) & (np.invert(mask[1:])))+1
masked_steps = steps.copy()
masked_steps[np.invert(mask)] = np.nan

mask2 = steps > 0.7
diff_mask2 = np.flatnonzero((mask2[:-1]) & (np.invert(mask2[1:])))+1
masked_steps2 = steps.copy()
masked_steps2[np.invert(mask2)] = np.nan

plt.plot(
    x_test,
   steps,
)


plt.plot(x_test, masked_steps, "orange")
plt.scatter([x_test[diff_mask]], [steps[diff_mask]], marker="o", color="orange", s=30, zorder=5, label="Case 1")
plt.hlines(0.99, -1,28, "orange", "--", "Case 1 trigger level", alpha=0.3)

plt.plot(x_test, masked_steps2, "green")
plt.scatter([x_test[diff_mask2]], [steps[diff_mask2]], marker="o", color="green", s=30, zorder=5, label="Case 2")
plt.hlines(0.7, -1,28, "green", "--", "Case 2 trigger level", alpha=0.3)

plt.xlabel("Time")
plt.ylabel("Normalised signal")
plt.xticks([])

# log = np.logspace(-3.5, 0, 30)
# plt.plot(log, gain_sat_r_gauss(log, 4, 103, 99, np.inf))
plt.tight_layout()
plt.legend()
# plt.savefig("Images\old_faults.png")
plt.show()
