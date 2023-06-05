import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter


def animate(i):
    ax.cla()
    i = 10 * i
    shift = i**3 / 46875000 - i**2 / 31250 + 4 * i / 375
    x = np.linspace(-1.3, 1.7, 2000) - shift * 0.8 - 0.2
    f2 = np.exp(-(x**2) * 3e3)

    ax.plot(x_fixed, f2, label="$I_{\omega}$")
    ax.plot(t_pulse[0], t_pulse[1] / max(t_pulse[1]), label="$E_{THz}$")

    signal_y.append(
        np.sum(
            t_pulse[1]
            / max(t_pulse[1])
            * np.exp(-((t_pulse[0] - shift * 0.8 - 0.2) ** 2) * 3e3)
        )
    )
    signal_x.append(shift * 0.8 + 0.2)

    ax.set(xlim=(-1.05, 1.56), ylim=(-1.05, 1.05))
    ax.set_xlabel("Time / ps")
    ax.set_ylabel("Normalised amplitude")
    fig.tight_layout()
    ax.legend(fontsize="large", loc=1)
    # ax.grid()


fig: matplotlib.figure.Figure
ax: matplotlib.axes.Axes
x_fixed = np.linspace(-1.3, 1.7, 2000)
signal_y = []
signal_x = []

fig, ax = plt.subplots(figsize=(5, 3))
fig2, ax2 = plt.subplots(figsize=(5, 3))

t_pulse = pd.read_csv(
    r"C:\Users\ueli\Dropbox\ETH\Ultrafast Methods in Solid State Physics\Presentation\dataset.csv",
    delimiter=";",
    decimal=",",
    header=None,
)

anim = FuncAnimation(fig, animate, interval=100, frames=100, repeat=False)

anim.save(
    r"C:\Users\ueli\Dropbox\ETH\Semsterarbeit\VecselSemesterProject\Pythontest.gif",
    writer=PillowWriter(fps=60),
)

animate(0)

fig.savefig(
    r"C:\Users\ueli\Dropbox\ETH\Semsterarbeit\VecselSemesterProject\Starfig.png",
)

# ax2.plot(signal_x, signal_y / max(signal_y))
#
# ax2.set(xlim=(-1.05, 1.56), ylim=(-1.05, 1.05))
# ax2.set_xlabel("Time / ps")
# ax2.set_ylabel("Normalised amplitude")
# fig2.tight_layout()


fig2.show()
