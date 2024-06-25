"""
Place for all helping functions and constants in this module.
"""

import numpy as np

MU0 = 4 * np.pi * 1e-7  # Magnetic permeability

# ### add __all__ ? ###


def wavenumber2wavelength(wavenumber):
    """Convert wavelength to wavenumber.
    lambda = 2*pi/k

    Parameters
    ----------
    wavenumber : float or array_like
        (rad/m) wavenumber of the wave.

    Returns
    -------
    wavelength : float or ndarray
        (m) corresponding wavelength.
    """
    return 2 * np.pi / np.array(wavenumber)


def wavelength2wavenumber(wavelength):
    """Convert wavenumber to wavelength.
    k = 2*pi/lambda

    Parameters
    ----------
    wavelength : float or ndarray
        (m) wavelength of the wave.

    Returns
    -------
    wavenumber : float or ndarray
        (rad/m) wavenumber of the corresponding wavelength.
    """
    return 2 * np.pi / np.array(wavelength)


def wrapAngle(angle):
    """Wrap angle in radians to range [0, 2*np.pi).

    Parameters
    ----------
    angle : float or array_like
        (rad) angle to wrap.

    Returns
    -------
    wrapped_angle : float or ndarray
        (rad) angle wrapped to [0, 2*np.pi).
    """
    # return np.mod(angle + np.pi, 2 * np.pi) - np.pi
    return np.mod(angle, 2 * np.pi)

