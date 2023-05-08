import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

x_fixed = np.linspace(-2, 2, 20000)


def animate(i):
    ax.cla()

    x = np.linspace(-0.5, 0.5, 20000) - i / 10
    f, fenv = signal.gausspulse(x, fc=5, bw=2, retquad=False, retenv=True)
    f2, fenv2 = signal.gausspulse(x, fc=120, bw=2, retenv=True)

    fenv = fenv * np.exp(1j * 2 * (x - 4) ** 3 + 1j * 3 * (x - 4) ** 2 + 1j * 0.04 * (x - 4) ** 4)

    ax.plot(x_fixed, np.real(fenv2 * np.exp(1j * 20000 * x)))
    ax.plot(x_fixed, np.real(fenv)/np.max(np.abs(np.real(fenv))))

    ax.set(xlim=(-.5, .5), ylim=(-1, 1))
    ax.grid()


fig, ax = plt.subplots(figsize=(5, 3))


animate(0)
plt.show()
#
# anim = FuncAnimation(fig, animate, interval=1000, frames=1000, repeat=False)
# writer = PillowWriter(fps=60, metadata=dict(artist="Ileu"), bitrate=1800)
#
# anim.save(
#     r"C:\Users\ueli\Dropbox\ETH\Semsterarbeit\VecselSemesterProject\Pythontest.gif",
#     writer=writer,
# )
