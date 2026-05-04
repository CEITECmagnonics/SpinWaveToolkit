"""
This submodule focuses on modelling the Brillouin light scattering
signal.

To get more insight, consult the relevant documentation pages.
For example, see :doc:`/api_reference/bls/functions` for the explanation
of differences between the available functions that calculate BLS
spectra.

.. currentmodule:: SpinWaveToolkit.bls

Modules
-------
:mod:`~SpinWaveToolkit.bls.susceptibilities`
    Module for computing the dynamic magneto-optic susceptibility tensor
    components for a given dynamic magnetization vector.

Classes
-------
:class:`ObjectiveLens`
    Class for calculation of the focal electric fields of given lens.

Functions
---------

.. autosummary::
    get_signal_GF_focal
    getBLSsignal
    get_signal_RT_pupil
    get_signal_RT_focal
    get_signal_RT_focal_3d
    fresnel_coefficients
    sph_green_function

Example
-------
The usage of this submodule is not as straightforward as for the rest of
the :mod:`SpinWaveToolkit`, and therefore the reader is referred to the
relevant :doc:`/examples`.

"""

from .greenAndFresnel import *
from .BLSmodel import *
from .core._class_ObjectiveLens import *
from . import susceptibilities


__all__ = [
    "ObjectiveLens",
    "get_signal_GF_focal",
    "getBLSsignal",
    "get_signal_RT_pupil",
    "get_signal_RT_focal",
    "get_signal_RT_focal_3d",
    "fresnel_coefficients",
    "sph_green_function",
    "susceptibilities",
]
