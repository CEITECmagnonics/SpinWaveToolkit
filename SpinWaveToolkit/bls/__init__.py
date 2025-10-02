"""
This submodule focuses on modelling the Brillouin light scattering
signal.

.. currentmodule:: SpinWaveToolkit.bls

Classes
-------
:class:`ObjectiveLens`
    Class for calculation of the focal electric fields of given lens.

Functions
---------
:func:`fresnel_coefficients`
    Compute Fresnel reflection and transmission coefficients.
htp
    Compute p-polarized Fresnel coefficients for a given lateral
    wavevector q.  Returned by :func:`fresnel_coefficients`.
hts
    Compute s-polarized Fresnel coefficients for a given lateral
    wavevector q.  Returned by :func:`fresnel_coefficients`.
:func:`sph_green_function`
    Compute the spherical Green's functions for p- and s-polarized
    fields.
:func:`getBLSsignal`
    Compute the Brillouin light scattering signal using Green's
    functions formalism.

Example
-------
The usage of this submodule is not as straightforward as for the rest of
the :mod:`SpinWaveToolkit`, and therefore the reader is referred to the
relevant :doc:`/examples`.

"""

from .greenAndFresnel import *
from .BLSmodel import *
from .core._class_ObjectiveLens import *


__all__ = [
    "ObjectiveLens",
    "fresnel_coefficients",
    "getBLSsignal",
]
