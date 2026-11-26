"""
Core (private) file for the `Material` class.
"""

from copy import deepcopy

import numpy as np

__all__ = ["Material", "NiFe", "CoFeB", "FeNi", "YIG"]


class Material:
    """Class for magnetic materials used in spin wave research.

    To define a custom material, type

    .. code-block:: python

        MyNewMaterial = Material(Ms=MyMS, Aex=MyAex, alpha=MyAlpha, gamma=MyGamma)

    The predefined materials (e.g. ``NiFe``) are shared instances, so
    changing their attributes changes them for the rest of the session.
    To use a modified material, e.g. in a parameter sweep, make a copy

    .. code-block:: python

        MyNewMaterial = NiFe.copy(Ms=900e3)


    Parameters
    ----------
    Ms : float
        (A/m) saturation magnetization.
    Aex : float
        (J/m) exchange stiffness constant.
    alpha : float
        () Gilbert damping.
    gamma : float, optional
        (rad*Hz/T) gyromagnetic ratio.  Default ``28.1e9*2*np.pi``.
    mu0dH0 : float, optional
        (T ) inhomogeneous broadening.  Default is 0.
    Ku : float, optional
        (J/m^2) surface anisotropy strength.  Default is 0.
        (Currently unused in any dispersion calculations.)

    Attributes
    ----------
    same as Parameters

    Methods
    -------
    copy
    get_pinning

    Predefined materials (as module constants)
    ------------------------------------------
    NiFe (Permalloy)
    CoFeB
    FeNi (Metastable iron)
    YIG

    See also
    --------
    SingleLayer, SingleLayerNumeric, DoubleLayerNumeric
    """

    def __init__(self, Ms, Aex, alpha, mu0dH0=0, gamma=28.1 * 2 * np.pi * 1e9, Ku=0):
        self.Ms = Ms
        self.Aex = Aex
        self.alpha = alpha
        self.gamma = gamma
        self.mu0dH0 = mu0dH0
        self.Ku = Ku

    def copy(self, **changes):
        """Return a copy of the material, optionally with changed
        parameters.

        The copy is independent of the original (also when the
        parameters are arrays), so it can be modified freely without
        changing the original, e.g. one of the predefined materials.

        Note that the dispersion classes read the parameters of the
        material when they are created, so a material changed afterwards
        (or its copy) does not affect existing objects.

        Parameters
        ----------
        **changes : float or array_like, optional
            New values of the parameters of the copy, given by their
            names (e.g. ``Ms=900e3``).  The remaining parameters are
            those of the original.

        Returns
        -------
        material : Material
            Copy of the material with the changed parameters.

        Raises
        ------
        TypeError
            If a name in `changes` is not an attribute of the material.

        Examples
        --------
        Sweep of the saturation magnetization, leaving the predefined
        ``NiFe`` unchanged.

        .. code-block:: python

            import numpy as np
            import SpinWaveToolkit as SWT

            mat = SWT.NiFe.copy()
            for Ms in np.linspace(700e3, 900e3, 5):
                mat.Ms = Ms
                sl = SWT.SingleLayer(50e-3, mat, 30e-9)
                ...
        """
        material = deepcopy(self)
        for name, value in changes.items():
            if name not in vars(self):
                raise TypeError(
                    f"'{name}' is not an attribute of the material "
                    f"(choose from {list(vars(self))})."
                )
            setattr(material, name, value)
        return material

    def get_pinning(self):
        """Calculates the symmetric pinning parameter on a surface
        with a given surface anisotropy. The result is in rad/m.
        ``p=-2*np.pi*Ku/Aex``

            ### Is this correct?

        https://doi.org/10.1103/PhysRevB.83.174417
        """
        return -2 * np.pi * self.Ku / self.Aex


# Predefined materials
NiFe = Material(Ms=800e3, Aex=16e-12, alpha=70e-4, gamma=28.8 * 2 * np.pi * 1e9)
CoFeB = Material(Ms=1250e3, Aex=15e-12, alpha=40e-4, gamma=30 * 2 * np.pi * 1e9)
FeNi = Material(Ms=1410e3, Aex=11e-12, alpha=80e-4)
YIG = Material(
    Ms=140e3, Aex=3.6e-12, alpha=1.5e-4, gamma=28 * 2 * np.pi * 1e9, mu0dH0=0.18e-3
)
