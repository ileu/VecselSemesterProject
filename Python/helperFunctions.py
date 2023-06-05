import glob

import numpy as np
from PIL import Image
from scipy.integrate import quad_vec

threshold = 7.06  # in ampere
slope = 0.41  # in watt per ampere


def power_curve(cur):
    power = (cur - threshold) * slope
    power[power < 0] = 0
    return power


def power_curve_digit(cur):
    power = (cur - threshold) * slope
    if power < 0:
        return 0
    return power


def gain_sat_r(fluence, fsat, rss, rns, f2):
    """Calculate the  reflectivity for a given fluence of a VECSEL

    Args:
        fluence (flat): fluence
        fsat (float): saturation fluence
        rss (float): reflection without fluence
        rns (float): non-saturable reflection
        f2 (float): walk off

    Returns:
        float: reflection
    """
    temp = 1 + rss / rns * (np.exp(fluence / fsat) - 1)

    return rns * fsat / fluence * np.log(temp) * np.exp(-fluence / f2)


def gain_sat_r_gauss(fluence, fsat, rss, rns, f2):
    """Calculate the  reflectivity for a given fluence of a VECSEL
        Takes the Gaussion profile into account

    Args:
        fluence (flat): fluence
        fsat (float): saturation fluence
        rss (float): reflection without fluence
        rns (float): non-saturable reflection
        f2 (float): walk off

    Returns:
        float: reflection
    """

    def r_gauss(x):
        result = gain_sat_r(2 * fluence * x, fsat, rss, rns, f2)
        return result

    return quad_vec(r_gauss, min(fluence) / max(fluence), 1)[0]


def gain_sat_cor(fluence, fsat, rss, rns):
    def r_gauss(x):
        result = gain_sat_r(2 * fluence * x, fsat, rss, rns, 1) * np.exp(fluence)
        return result

    return quad_vec(r_gauss, min(fluence) / max(fluence), 1)[0]


def make_gif(frame_folder, filename, suffix=".png", duration=300):
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
