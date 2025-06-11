"""
Core (private) file for the `SingleLayerSCcoupled` class.
"""

import numpy as np
from SpinWaveToolkit.helpers import *

__all__ = ["SingleLayerSCcoupled"]


class SingleLayerSCcoupled:
    """Compute spin wave characteristic in dependance to k-vector
    (wavenumber) such as frequency, group velocity, lifetime and
    propagation length.

    This model describes the spin-wave behaviour in a thin ferromagnetic 
    film in contact with a superconductor (neglecting any proximity 
    effects).

    The class uses the model of Zhou et al. from
    https://doi.org/10.1103/PhysRevB.110.L020404
    slightly extended as described in this thesis
    https://www.vut.cz/en/students/final-thesis/detail/166558

    Most parameters can be specified as vectors (1d numpy arrays)
    of the same shape. This functionality is not guaranteed.

    Note: Right now only the DE mode is implemented (phi=pi/2, 
    theta=pi/2) for the zeroth order mode in totally unpinned conditions
    (this also assumes no hybridizations).  See original paper for 
    possible limitations.

    # ### update the docstring after implementing the model 
    #   (also check other models)

    # ### try to compare numeric calculation of the lifetime to the one
    #   derived by Zhou et al.

    # ### todo: add also possibility to calculate approximate dispersion 
    #   with finite SC and IS

    Parameters
    ----------
    Bext : float
        (T) external magnetic field.
    material : Material
        Instance of `Material` describing the magnetic layer material.
        Its properties are saved as attributes, but this object is not.
    d : float
        (m) magnetic layer thickness (in z direction).
    kxi : float or ndarray, default np.linspace(1e-12, 25e6, 200)
        (rad/m) k-vector (wavenumber), usually a vector.
    theta : float, default np.pi/2
        (rad) out of plane angle of static M, pi/2 is totally
        in-plane magnetization.
    phi : float or ndarray, default np.pi/2
        (rad) in-plane angle of kxi from M, pi/2 is DE geometry.
    weff : float, optional
        (m) effective width of the waveguide (not used for zeroth
        order width modes).
    boundary_cond : {1, 2, 3, 4}, default 1
        boundary conditions (BCs), 1 is totally unpinned and 2 is
        totally pinned BC, 3 is a long wave limit, 4 is partially
        pinned BC.
    dp : float, optional
        (rad/m) pinning parameter for 4 BC, ranges from 0 to inf,
        0 means totally unpinned. Can be calculated as `dp=Ks/Aex`, 
        see https://doi.org/10.1103/PhysRev.131.594.

    Attributes (same as Parameters, plus these)
    -------------------------------------------
    Ms : float
        (A/m) saturation magnetization.
    gamma : float
        (rad*Hz/T) gyromagnetic ratio (positive convention).
    Aex : float
        (J/m) exchange stiffness constant.
    alpha : float
        () Gilbert damping.
    mu0dH0 : float
        (T) inhomogeneous broadening.
    w0 : float
        (rad*Hz) parameter in Slavin-Kalinikos equation.
        `w0 = MU0*gamma*Hext`
    wM : float
        (rad*Hz) parameter in Slavin-Kalinikos equation.
        `wM = MU0*gamma*Ms`
    A : float
        (m^2) parameter in Slavin-Kalinikos equation.
        `A = Aex*2/(Ms**2*MU0)`

    Methods
    -------
    GetPartiallyPinnedKappa
    GetDisperison
    GetGroupVelocity
    GetLifetime
    GetDecLen
    GetSecondPerturbation
    GetDensityOfStates
    GetBlochFunction
    GetExchangeLen
    GetEllipticity
    GetCouplingParam
    GetThresholdField

    Private methods
    ---------------
    __GetPropagationVector
    __GetPropagationQVector
    __GetAk
    __GetBk

    Code example
    ------------
    Example of calculation of the dispersion relation `f(k_xi)`, and
    other important quantities, for the lowest-order mode in a 30 nm
    thick NiFe (Permalloy) layer.
    .. code-block:: python
        kxi = np.linspace(1e-6, 150e6, 150)

        PyChar = SingleLayer(Bext=20e-3, kxi=kxi, theta=np.pi/2,
                             phi=np.pi/2, d=30e-9, weff=2e-6,
                             boundary_cond=2, material=SWT.NiFe)
        DispPy = PyChar.GetDispersion()*1e-9/(2*np.pi)  # GHz
        vgPy = PyChar.GetGroupVelocity()*1e-3  # km/s
        lifetimePy = PyChar.GetLifetime()*1e9  # ns
        decLen = PyChar.GetDecLen()*1e6  # um

    See also
    --------
    SingleLayer, SingleLayerNumeric, DoubleLayerNumeric, Material

    """
    
    def __init__(
        self,
        Bext,
        material,
        d,
        kxi=np.linspace(1e-12, 25e6, 200),
        lam=1e-7,
        d_sc=np.inf,
        d_is=0,
    ):
        self._Bext = Bext
        self._Ms = material.Ms
        self._gamma = material.gamma
        self._Aex = material.Aex

        self.kxi = np.array(kxi)
        self.d = d
        self.alpha = material.alpha
        self.mu0dH0 = material.mu0dH0
        self.lam = lam

