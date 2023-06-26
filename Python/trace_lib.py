import numpy as np


def sin_noise(x, w, amp, noise):
    """
    Generates sinusoidal noise to add to data.

    Parameters:
    -----------
    x : np.array
        x_coordinates for noise.
    w : float
        frequency of sin wave.
    amp : float
        amplitude of sine wave.
    noise : float
        standard deviation of added random noise.

    Returns:
    --------
    np.array
        Noise.
    """
    return np.sin(w * x) * amp + np.random.normal(0, noise, x.size)


def step_up(x, k, a):
    """
    Computes the step function for the given parameters.

    Parameters:
    -----------
    x : np.array
        Input values.
    k : float
        Controls the steepness of the step function.
    a : float
        Controls the shape of the step function.

    Returns:
    --------
    np.array
        Step function values.
    """
    return 1 / (1 + np.exp(-k * x)) ** a


def step(x, k, a, width):
    """
    Computes a step function shifted of a certai width.

    Parameters:
    -----------
    x : np.array
        Input values.
    k : float
        Controls the steepness of the step function.
    a : float
        Controls the shape of the step function.
    width : float
        Amount to shift the step function.

    Returns:
    --------
    np.array
        step function values.
    """
    return step_up(x, k, a) - step_up(x - width, k, a)


def step_moved(x, dx, k, a, width):
    """
    Computes the step function of a certain width and moved by a certain distance dx.

    Parameters:
    -----------
    x : np.array
        Input values.
    dx : float
        Amount to move the step function.
    k : float
        Constant that controls the steepness of the step function.
    a : float
        Controls the shape of the step function.
    width : float
        Amount to shift the step function.

    Returns:
    --------
    np.array
        step function values.
    """
    return step(x - dx, k, a, width)


def three_step(x, k=20, a=2, dx=None, widths=None, heights=None):
    """
    Computes a combination of three step functions.

    Parameters:
    -----------
    x : np.array
        Input values.
    k : float, optional
        Constant that controls the steepness of the step function (default is 20).
    a : float, optional
        Controls the shape of the step function (default is 2).
    dx : list or None, optional
        List of three amounts to shift the step functions horizontally (default is None).
        If None, default values [0.65, 2.6, 6.4] are used.
    dx2 : list or None, optional
        List of three widths for the step functions (default is None).
        If None, default values [3.4, 5.4, 9] are used.
    heights : list or None, optional
        List of three heights to scale the step functions (default is None).
        If None, default values [0.7, 0.3, 0.4] are used.

    Returns:
    --------
    np.array
        Combined step function values.
    """
    if not dx:
        dx = [0.65, 2.6, 6.4]

    if not widths:
        widths = [3.4, 5.4, 9]

    if not heights:
        heights = [0.7, 0.3, 0.4]

    return (
        heights[0] * step_moved(x, dx[0], k, a, widths[0])
        + heights[1] * step_moved(x, dx[1], k, a, widths[1])
        + heights[2] * step_moved(x, dx[2], k, a, widths[2])
    )
