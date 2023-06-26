import glob

import numpy as np
from PIL import Image
from scipy.integrate import quad_vec

threshold = 7.06  # in ampere
slope = 0.41  # in watt per ampere

# relative paths to data locations
paths = [
    r"\Semesterarbeit\SV167-b2-10C-2070nm",
    r"\Semesterarbeit\SV167-b2-0C-2070nm",
    r"\Semesterarbeit\SV167-b2-n10C-2070nm",
    r"\Semesterarbeit\SV167-b5-10C-2070nm",
    r"\Semesterarbeit\SV167-b5-0C-2070nm",
    r"\Semesterarbeit\SV167-b5-n10C-2070nm",
    r"\Semesterarbeit\SV166-a4-10C-2070nm",
    r"\Semesterarbeit\SV166-a4-0C-2070nm",
    r"\Semesterarbeit\SV166-a4-n10C-2070nm",
    r"\Semesterarbeit\SV165-CD2-RT-2070nm",
    r"\Semesterarbeit\SV165-CD2-10C-2070nm",
    r"\Semesterarbeit\SV165-CD2-0C-2070nm",
    r"\Semesterarbeit\SV165-CD2-n10C-2070nm",
    # r"\Semesterarbeit\SV172-Dia-15C-2070nm",
    # r"\Semesterarbeit\SV176-a3-15C-2070nm",
    # r"P:\Semesterarbeit\SV167-b5-2-10C-2070nm_bs",
    # r"\Semesterarbeit\SV167-b2-14C-2050mm",
]


def power_curve(current):
    """ calcualte the power of the pump diode laser 
        used for the measurement for a list of current values

    Parameters
    ----------
    current : np.array
        list of currents to calcualte the powers for

    Returns
    -------
    np.array
        list of corresponding pump powers
    """
    power = (current - threshold) * slope
    power[power < 0] = 0
    return power


def power_curve_digit(current):
    """ calcualte the power of the pump diode laser 
        used for the measurement for a single current

    Parameters
    ----------
    current : np.array
        list of currents to calcualte the powers for

    Returns
    -------
    np.array
        list of corresponding pump powers
    """
    power = (current - threshold) * slope
    if power < 0:
        return 0
    return power


def gain_sat_r(fluence, fsat, rss, rns, f2):
    """ Calculate the reflectivity for a given fluence of a VECSEL
        based on a model for the saturable absorption of a SESAM

    Parameters
    ----------
    fluence : np.array
        list of fluence values
    fsat : float
        saturation fluence
    rss : float
        small signal reflectivity
    rns : float
        non-saturable reflectivity
    f2 : float
        roll over parameter

    Returns
    -------
    np.array
        calculated reflectivity
    """
    temp = 1 + rss / rns * (np.exp(fluence / fsat) - 1)

    return rns * fsat / fluence * np.log(temp) * np.exp(-fluence / f2)


def gain_sat_r_gauss(fluence, fsat, rss, rns, f2):
    """ Calculate the reflectivity for a given fluence of a VECSEL
        based on a model for the saturable absorption of a SESAM
        Takes also the gaussion beam profile into account

    Parameters
    ----------
    fluence : np.array
        list of fluence values
    fsat : float
        saturation fluence
    rss : float
        small signal reflectivity
    rns : float
        non-saturable reflectivity
    f2 : float
        roll over parameter

    Returns
    -------
    np.array
        calculated reflectivity
    """

    def r_gauss(x):
        result = gain_sat_r(2 * fluence * x, fsat, rss, rns, f2)
        return result

    return quad_vec(r_gauss, min(fluence) / max(fluence), 1)[0]


def make_gif(frame_folder, filename, suffix=".png", duration=300):
    """ makes a gif out of picture in a folder

    Parameters
    ----------
    frame_folder : string
        path to the folder containg the picture
    filename : string
        path containing the gif name for saveing the gif
    suffix : str, optional
        file extension, by default ".png"
    duration : int, optional
        Display time in milliseconds of one frame in the gif, by default 300
    """
    frames = [
        Image.open(image)
        for image in sorted(glob.glob(f"{frame_folder}/*{suffix}"), key=len)
    ]
    frame_one = frames[0]
    frame_one.save(
        filename,
        format="GIF",
        append_images=frames,
        save_all=True,
        duration=duration,
    )


def eformat(f, prec, exp_digits):
    """Scientific formatter  numbers for the pulse fluences

    Parameters
    ----------
    f : flat
        the number
    prec : int
        how many digits after decimal
    exp_digits : int
        how many digits in the exponent

    Returns
    -------
    string
        formatted number
    """
    s = "%.*e" % (prec, f)
    mantissa, exp = s.split("e")
    # add 1 to digits as 1 is taken by sign +/-
    return "%se%+0*d" % (mantissa, exp_digits + 1, int(exp))


def save_figure(figure, location, show=False, blocked=True):
    figure.tight_layout()

    figure.savefig(location, bbox_inches="tight", dpi=200)
    if show:
        figure.show(block=blocked)
