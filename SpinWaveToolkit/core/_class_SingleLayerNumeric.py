"""
Core (private) file for the `SingleLayerNumeric` class.
"""

import numpy as np
from numpy import linalg
from SpinWaveToolkit.helpers import MU0, roots, distBE

__all__ = ["SingleLayerNumeric"]


class SingleLayerNumeric:
    """Compute spin wave characteristic in dependance to k-vector
    (wavenumber) such as frequency, group velocity, lifetime and
    propagation length for desired number of modes.

    The dispersion model uses the approach of Tacchi et al., see:
    https://doi.org/10.1103/PhysRevB.100.104406

    The laboratory coordinate frame of reference is
    *z || to film normal* and *x || to in-plane wavevector*.

    The wavenumber `kxi`, the angles `theta` and `phi`, the external
    field `Bext`, the thickness `d` and the material parameters can be
    given either as scalars or as 1D arrays of the same length, and
    the results are calculated elementwise.  This allows, e.g., to
    calculate the dispersion on a 2D grid of wavevectors at once (with
    flattened arrays of wavenumbers and angles `phi`), or a field sweep
    for a single wavenumber.  The system matrices for all elements are
    built and diagonalized at once, so there is no loop over the
    elements.  Exceptions: `d` and `dp` must be scalars for
    ``boundary_cond=4``, and the group velocity (and the quantities
    derived from it) needs `kxi` to be the varying 1D array.

    Parameters
    ----------
    Bext : float or 1D array
        (T ) external magnetic field.
    material : Material
        instance of `Material` describing the magnetic layer material.
        Its properties are saved as attributes, but this object is not.
    d : float or 1D array
        (m ) layer thickness (in z direction)
    kxi : float or 1D array, optional
        (rad/m) k-vector (wavenumber), usually a vector.
    theta : float or 1D array, optional
        (rad) out of plane angle static M, pi/2 is totally
        in-plane magnetization.  Other values than pi/2 multiples
        might give wrong results, as the model currently does not
        describe these situations.
    phi : float or 1D array, optional
        (rad) in-plane angle of M from kxi, pi/2 is DE geometry.
    weff : float, optional
        (m ) effective width of the waveguide (not used for zeroth
        order width modes).
    boundary_cond : {1, 2, 3, 4}, optional
        boundary conditions (BCs), 1 is totally unpinned and 2 is
        totally pinned BC, 3 is a long wave limit, 4 is partially
        pinned BC.  Default is 1.
            ### The only working BCs are 1 right now, some functions
            implement 2 and 4, but it is not complete!
    dp : float, optional
        pinning parameter for 4 BC, ranges from 0 to inf,
        0 means totally unpinned.
    KuOOP : float, optional
        (J/m^3) OOP anisotropy strength used in the Tacchi model.
            ### Should this be calculated from the surface anisotropy
            strength as `KuOOP = 2*Ks/d + OOP_comp_of_bulk_anis`?,
            where `d` is film thickness and `Ks` is the surface
            anisotropy strength (same as material.Ku)
    N : int, optional
        Number of modes to calculate, default is 3.
        This parameter defines the size of the system matrix
        and the number of modes to calculate.  The size of the
        system matrix is `2*N x 2*N`, so for N=3 it is 6x6.
        The modes are sorted from low to high frequencies.

    Attributes
    ----------
    [same as Parameters (except `material`), plus these]
    Ms : float
        (A/m) saturation magnetization.
    gamma : float
        (rad*Hz/T) gyromagnetic ratio (positive convention).
    Aex : float
        (J/m) exchange stiffness constant.
    alpha : float
        () Gilbert damping.
    mu0dH0 : float
        (T ) inhomogeneous broadening.
    w0 : float
        (rad*Hz) parameter in Kalinikos-Slavin equation.
        ``w0 = MU0*gamma*Hext``
    wM : float
        (rad*Hz) parameter in Kalinikos-Slavin equation.
        ``wM = MU0*gamma*Ms``
    A : float
        (m^2) parameter in Kalinikos-Slavin equation.
        ``A = Aex*2/(Ms**2*MU0)``
    wU : float
        (rad*Hz) circular frequency of OOP anisotropy field,
        used in the Tacchi model.

    Methods
    -------
    GetDispersion
    GetGroupVelocity
    GetLifetime
    GetDecLen
    GetDensityOfStates
    GetBlochFunction
    GetExchangeLen
    set_DE
    set_BV

    Private methods
    ---------------
    __CnncTacchi
    __pnncTacchi
    __qnncTacchi
    __OmegankTacchi
    __ankTacchi
    __bTacchi
    __PnncTacchi
    __QnncTacchi
    __Ck

    Examples
    --------
    Example of calculation of the dispersion relation `f(k_xi)`, and
    other important quantities, for the lowest-order mode in a 30 nm
    thick NiFe (Permalloy) layer.

    .. code-block:: python

        kxi = np.linspace(1e-6, 150e6, 150)

        PyChar = SingleLayerNumeric(Bext=20e-3, kxi=kxi, theta=np.pi/2,
                                    phi=np.pi/2, d=30e-9, weff=2e-6,
                                    boundary_cond=2, material=SWT.NiFe)
        DispPy = PyChar.GetDispersion()[0][0]*1e-9/(2*np.pi)  # GHz
        vgPy = PyChar.GetGroupVelocity()*1e-3  # km/s
        lifetimePy = PyChar.GetLifetime()*1e9  # ns
        decLen = PyChar.GetDecLen()*1e6  # um

    See also
    --------
    SingleLayer, DoubleLayerNumeric, Material

    """

    def __init__(
        self,
        Bext,
        material,
        d,
        kxi=np.linspace(1e-12, 25e6, 200),
        theta=np.pi / 2,
        phi=np.pi / 2,
        weff=3e-6,
        boundary_cond=1,
        dp=0,
        KuOOP=0,
        N=3,
    ):
        self._Bext = Bext
        self._Ms = material.Ms
        self._gamma = material.gamma
        self._Aex = material.Aex
        self._KuOOP = KuOOP
        self.N = N
        self.kxi = np.array(kxi)
        if np.any((np.asarray(theta) + 1e-6) % (np.pi / 2) > 1e-3):
            print(
                "WARNING: theta is not a multiple of pi/2. The results might be misleading!"
            )
        self.theta = theta
        self.phi = phi
        self.d = d
        self.weff = weff
        self.boundary_cond = boundary_cond
        self.dp = dp
        self.alpha = material.alpha
        self.mu0dH0 = material.mu0dH0
        # Compute Kalinikos-Slavin parameters wM, w0, A
        self.wM = self.Ms * self.gamma * MU0
        self.w0 = self.gamma * Bext
        self.wU = self.gamma * 2 * self.KuOOP / self.Ms  # only for Tacchi
        self.A = self.Aex * 2 / (self.Ms**2 * MU0)

    @property
    def Bext(self):
        """External field value (T)."""
        return self._Bext

    @Bext.setter
    def Bext(self, val):
        self._Bext = val
        self.w0 = self.gamma * val

    @property
    def Ms(self):
        """Saturation magnetization (A/m)."""
        return self._Ms

    @Ms.setter
    def Ms(self, val):
        self._Ms = val
        self.wM = val * self.gamma * MU0
        self.A = self.Aex * 2 / (val**2 * MU0)
        self.wU = self.gamma * 2 * self.KuOOP / val

    @property
    def gamma(self):
        """Gyromagnetic ratio (rad*Hz/T)."""
        return self._gamma

    @gamma.setter
    def gamma(self, val):
        self._gamma = val
        self.wM = self.Ms * val * MU0
        self.w0 = val * self.Bext
        self.wU = val * 2 * self.KuOOP / self.Ms

    @property
    def Aex(self):
        """Exchange stiffness constant (J/m)."""
        return self._Aex

    @Aex.setter
    def Aex(self, val):
        self._Aex = val
        self.A = val * 2 / (self.Ms**2 * MU0)

    @property
    def KuOOP(self):
        """OOP uniaxial anisotropy constant (J/m^3)."""
        return self._KuOOP

    @KuOOP.setter
    def KuOOP(self, val):
        self._KuOOP = val
        self.wU = self.gamma * 2 * val / self.Ms

    @property
    def N(self):
        """Number of modes to calculate."""
        return self._N

    @N.setter
    def N(self, val):
        self._N = val

    def __ang_coeff(self):
        """Compute A, B, C, D, E as in Kalinikos-Slavin 1986."""
        th = self.theta
        ph = self.phi
        s = np.sin
        c = np.cos
        A = -s(th) ** 2
        E = s(ph) ** 2 + c(th) ** 2 * c(ph) ** 2
        D = -2.0 * s(th) * s(ph)
        # B and C not used in the mapping below; included for completeness:
        B = -np.sin(2.0 * th) * np.cos(ph)
        C = 0.5 * np.sin(2.0 * th) * np.cos(ph)
        return A, B, C, D, E

    def __CnncTacchi(self, n, nc, k):
        """Calculate the C_{n,nc}."""
        A, B, C, D, E = self.__ang_coeff()
        return 0.5 * self.wM * (A + E) * self.__PnncTacchi(n, nc, k)

    def __pnncTacchi(self, n, nc, k):
        """Calculate the p_{n,nc}."""
        A, B, C, D, E = self.__ang_coeff()
        return -0.5 * self.wM * (E - A) * self.__PnncTacchi(n, nc, k)

    def __qnncTacchi(self, n, nc, k):
        """Calculate the q_{n,nc}."""
        A, B, C, D, E = self.__ang_coeff()
        return self.wM * D * self.__QnncTacchi(n, nc, k)

    def __OmegankTacchi(self, n, k):
        """Calculate the w_{n,k}."""
        return self.w0 + self.wM * self.A * (k**2 + (n * np.pi / self.d) ** 2)

    def __ankTacchi(self, n, k):
        """Calculate the a_{n,k}."""
        return (
            self.__OmegankTacchi(n, k)
            + 0.5 * self.wM * np.sin(self.theta) ** 2
            - 0.5 * self.wU
        )

    def __bTacchi(self):
        """Calculate the b."""
        return 0.5 * self.wM * np.sin(self.theta) ** 2 - 0.5 * self.wU

    def __PnncTacchi(self, n, nc, kxi):
        """Gives dimensionless propagation vector.
        The boundary condition is chosen based on the object property.

        Parameters
        ----------
        n : int
            Quantization number.
        nc : int
            Second quantization number, used for hybridization.
        kxi : float
            (rad/m) wavenumber.
        """
        kappa = n * np.pi / self.d
        kappac = nc * np.pi / self.d
        k = np.sqrt(np.power(kxi, 2) + kappa**2)
        kc = np.sqrt(np.power(kxi, 2) + kappac**2)
        # Totally unpinned boundary condition
        if self.boundary_cond == 1:
            Fn = 2 / (kxi * self.d) * (1 - (-1) ** n * np.exp(-kxi * self.d))
            if n == 0 and nc == 0:
                Pnn = (kxi**2) / (kc**2) - (kxi**4) / (
                    k**2 * kc**2
                ) * 1 / 2 * ((1 + (-1) ** (n + nc)) / 2) * Fn
            elif n == 0 and nc != 0 or nc == 0 and n != 0:
                Pnn = (
                    -(kxi**4)
                    / (k**2 * kc**2)
                    * 1
                    / np.sqrt(2)
                    * ((1 + (-1) ** (n + nc)) / 2)
                    * Fn
                )
            elif n == nc:
                Pnn = (kxi**2) / (kc**2) - (kxi**4) / (k**2 * kc**2) * (
                    (1 + (-1) ** (n + nc)) / 2
                ) * Fn
            else:
                Pnn = -(kxi**4) / (k**2 * kc**2) * ((1 + (-1) ** (n + nc)) / 2) * Fn
        # Totally pinned boundary condition
        elif self.boundary_cond == 2:
            if n == nc:
                Pnn = (kxi**2) / (kc**2) + (kxi**2) / (k**2) * (
                    kappa * kappac
                ) / (kc**2) * (1 + (-1) ** (n + nc) / 2) * 2 / (kxi * self.d) * (
                    1 - (-1) ** n * np.exp(-kxi * self.d)
                )
            else:
                Pnn = (
                    (kxi**2)
                    / (k**2)
                    * (kappa * kappac)
                    / (kc**2)
                    * (1 + (-1) ** (n + nc) / 2)
                    * 2
                    / (kxi * self.d)
                    * (1 - (-1) ** n * np.exp(-kxi * self.d))
                )
        else:
            raise ValueError(
                "Sorry, there is no boundary condition with this number for"
                + "the Tacchi numeric solution."
            )

        return Pnn

    def __QnncTacchi(self, n, nc, kxi):
        """Gives dimensionless propagation vector Q.
        This vector accounts for interaction between odd and even
        spin wave modes.

        Parameters
        ----------
        n : int
            Quantization number.
        nc : int
            Second quantization number, used for hybridization.
        kxi : float
            (rad/m) wavenumber.
        """
        # ### The totally pinned BC should be added
        kappa = n * np.pi / self.d if n != 0 else 1
        kappac = nc * np.pi / self.d if nc != 0 else 1
        k = np.sqrt(np.power(kxi, 2) + kappa**2)
        kc = np.sqrt(np.power(kxi, 2) + kappac**2)
        # Totally unpinned boundary condition
        if self.boundary_cond == 1:
            norm = 1.0 / np.sqrt(
                (1 + (1 if n == 0 else 0)) * (1 + (1 if nc == 0 else 0))
            )
            Fn = 2 / (kxi * self.d) * (1 - (-1) ** n * np.exp(-kxi * self.d))
            Qnn = (
                (kxi**2 / kc**2)
                * (
                    (kappac**2 / (kappac**2 - kappa**2)) * 2 / (kxi * self.d)
                    - (kxi**2) / (2 * k**2) * Fn
                )
                * ((1 - (-1) ** (n + nc)) / 2)
                * norm
            )

        elif self.boundary_cond == 4:
            dp = self.dp
            kappa = self.GetPartiallyPinnedKappa(n)
            kappac = self.GetPartiallyPinnedKappa(nc)
            if kappa == 0:
                kappa = 1
            if kappac == 0:
                kappac = 1
            An = np.sqrt(
                2
                * (
                    (kappa**2 + dp**2) / kappa**2
                    + np.sin(kappa * self.d)
                    / (kappa * self.d)
                    * (
                        (kappa**2 - dp**2) / kappa**2 * np.cos(kappa * self.d)
                        + 2 * dp / kappa * np.sin(kappa * self.d)
                    )
                )
                ** -1
            )
            Anc = np.sqrt(
                2
                * (
                    (kappac**2 + dp**2) / kappac**2
                    + np.sin(kappac * self.d)
                    / (kappac * self.d)
                    * (
                        (kappac**2 - dp**2) / kappac**2 * np.cos(kappac * self.d)
                        + 2 * dp / kappac * np.sin(kappac * self.d)
                    )
                )
                ** -1
            )
            Qnn = (
                kxi
                * An
                * Anc
                / (2 * self.d * k**2 * kc**2)
                * (
                    (kxi**2 - dp**2)
                    * np.exp(-kxi * self.d)
                    * (np.cos(kappa * self.d) - np.cos(kappac * self.d))
                    + (kxi - dp)
                    * np.exp(-kxi * self.d)
                    * (
                        (dp * kxi - kappa**2) * np.sin(kappa * self.d) / kappa
                        - (dp * kxi - kappac**2) * np.sin(kappac * self.d) / kappac
                    )
                    + (kxi - dp)
                    * (
                        (dp * kxi - kappac**2)
                        * np.cos(kappa * self.d)
                        * np.sin(kappac * self.d)
                        / kappac
                        - (dp * kxi - kappa**2)
                        * np.cos(kappac * self.d)
                        * np.sin(kappa * self.d)
                        / kappa
                    )
                    + (
                        1
                        - np.cos(kappac * self.d)
                        * np.cos(kappa * self.d)
                        * 2
                        * (
                            kxi**2 * dp**2
                            + kappa**2 * kappac**2
                            + (kappac**2 + kappa**2) * (kxi**2 + dp**2)
                        )
                        / (kappac**2 - kappa**2)
                        - np.sin(kappa * self.d)
                        * np.sin(kappac**2 * self.d)
                        / (kappa * kappac * (kappac**2 - kappa**2))
                        * (
                            dp * kxi * (kappa**4 + kappac**4)
                            + (dp**2 * kxi**2 - kappa**2 * kappac**2)
                            * (kappa**2 + kappac**2)
                            - 2 * kappa**2 * kappac**2 * (dp**2 + kxi**2 - dp * kxi)
                        )
                    )
                )
            )
        else:
            raise ValueError(
                "Sorry, there is no boundary condition with this number for "
                + "the Tacchi numeric solution."
            )
        return Qnn

    def GetPartiallyPinnedKappa(self, n):
        """Gives kappa from the transverse equation (in rad/m).

        Parameters
        ----------
        n : int
            Quantization number.
        """

        if np.ndim(self.d) > 0 or np.ndim(self.dp) > 0:
            raise ValueError(
                "`d` and `dp` must be scalars for the partially pinned boundary "
                + "condition (boundary_cond=4)."
            )

        def trans_eq(kappa, d, dp):
            e = (kappa**2 - dp**2) * np.tan(kappa * d) - kappa * dp * 2
            return e

        kappa0 = roots(
            trans_eq,
            n * np.pi / self.d,
            (n + 1) * np.pi / self.d,
            np.pi / self.d * 4e-4,
            # try decreasing dx if an error occurs
            np.pi / self.d * 1e-9,
            args=(self.d, self.dp),
        )
        for i in range(n + 1):
            # omit singularities at tan(kappa*d) when kappa*d = (n+0.5)pi
            kappa0[np.isclose(kappa0, np.pi / self.d * (i + 0.5))] = np.nan
            kappa0[kappa0 == 0.0] = np.nan  # omit 0 (probably only first is 0)
        kappa0 = kappa0[~np.isnan(kappa0)]  # remove NaNs
        return kappa0[0]

    def _Ck(self, k, N=3):
        """
        Build Tacchi/Kalinikos `C_k` with `N` thickness modes (size `2N x 2N`).
        Default ``N=3``, which gives 6x6 matrix.

        `k` and the parameters of the model can be scalars or 1D arrays
        of the same length ``M``.  The matrix has a shape of
        ``(2*N, 2*N)`` if all of them are scalars, otherwise
        ``(M, 2*N, 2*N)``.
        """
        b = self.__bTacchi()
        entries = []  # (row, column, value)

        # Diagonal 2x2 blocks for each mode n
        for n in range(N):
            ann = self.__ankTacchi(n, k) + self.__CnncTacchi(n, n, k)
            pnn = self.__pnncTacchi(n, n, k)
            i = 2 * n
            entries += [
                (i, i, -ann),
                (i, i + 1, -(b + pnn)),
                (i + 1, i, b + pnn),
                (i + 1, i + 1, ann),
            ]

        # Off-diagonal couplings between modes n != m
        # Same parity (both even or both odd): P-couplings => C,p blocks
        # Opposite parity (one even, one odd): Q-couplings => q blocks
        for n in range(N):
            for m in range(N):
                if n == m:
                    continue
                i, j = 2 * n, 2 * m
                if (n - m) % 2 == 0:
                    # same parity -> C,p block; keep your (col_mode, row_mode) call order
                    Cmn = self.__CnncTacchi(m, n, k)
                    pmn = self.__pnncTacchi(m, n, k)
                    entries += [
                        (i, j, -Cmn),
                        (i, j + 1, -pmn),
                        (i + 1, j, pmn),
                        (i + 1, j + 1, Cmn),
                    ]
                else:
                    # opposite parity -> q block
                    qmn = self.__qnncTacchi(
                        m, n, k
                    )  # note antisymmetry is inside your function
                    entries += [(i, j + 1, -qmn), (i + 1, j, -qmn)]

        # All entries are evaluated elementwise for array parameters
        shape = np.broadcast_shapes(*(np.shape(val) for _, _, val in entries))
        C = np.zeros(shape + (2 * N, 2 * N), dtype=float)
        for i, j, val in entries:
            C[..., i, j] += val
        return C

    def GetDispersion(self):
        """Gives frequencies for defined k (dispersion relation).

        Based on the model in:
        https://doi.org/10.1103/PhysRevB.100.104406

        The model formulates a system matrix and then numerically solves
        its eigenvalues and eigenvectors. The eigenvalues represent the
        dispersion relation (as the matrix is `2*N x 2*N` it has `2*N`
        eigenvalues).
        The eigen values represent `N` lowest spin-wave modes
        (`N` with negative and positive frequency).  The eigenvectors
        represent the amplitude of the individual spin-wave modes and
        can be used to calculate spin-wave profile (see the example
        :doc:`/_example_nbs/mode_profiles_numeric`).

        The returned modes are sorted from low to high frequencies,
        omitting the negative-frequency modes.

        Returns
        -------
        wV : ndarray
            (rad*Hz) frequencies of the `N` lowest spin-wave modes.
            Has a shape of ``(N, M)``, where ``M`` is the length of
            `kxi` (or of the array parameters, see the class
            docstring).
        vV : ndarray
            Mode profiles of corresponding eigenfrequencies,
            given as Fourier coefficients for IP and OOP profiles.
            Has a shape of ``(2*N, N, M)``.  ``vV[2*n]`` and
            ``vV[2*n+1]`` are the OOP and IP amplitudes of the thickness
            mode ``cos(n*pi*z/d)`` (weighted by ``1/sqrt(2)`` for
            ``n = 0``) for unpinned boundary conditions.
        """
        ks = np.abs(self.kxi)
        # System matrices for all elements, shape (M, 2N, 2N)
        Ck = self._Ck(k=ks, N=self.N).reshape(-1, 2 * self.N, 2 * self.N)
        w, v = linalg.eig(Ck)  # all matrices at once
        indi = np.argsort(w, axis=-1)[:, self.N :]  # sort low-to-high, keep positive
        wV = np.real(np.take_along_axis(w, indi, axis=-1)).T  # eigenvalues (dispersion)
        # eigenvectors (mode profiles), shape (2N, N, M)
        vV = np.real(np.take_along_axis(v, indi[:, np.newaxis, :], axis=-1))
        return wV, vV.transpose(1, 2, 0)

    def GetGroupVelocity(self, n=0):
        """Gives (tangential) group velocities for defined k.
        The group velocity is computed as vg = dw/dk.
        The result is given in m/s.

        .. warning::
            Works only when ``kxi.shape[0] >= 2`` and the elements of
            `kxi` follow each other along a line in k-space (not e.g.
            for flattened 2D grids of wavevectors).

        Parameters
        ----------
        n : {-1, 0, 1, 2, ..., N-1}, optional
            Quantization number.  If -1, data for all (positive)
            calculated modes are returned.  Default is 0.

        Returns
        -------
        vg : ndarray
            (m/s) tangential group velocity.
        """
        w, _ = self.GetDispersion()
        if n == -1:
            vg = np.zeros(w.shape)
            for i in range(w.shape[0]):
                vg[i] = np.gradient(w[i]) / np.gradient(self.kxi)
        else:
            vg = np.gradient(w[n]) / np.gradient(self.kxi)
        return vg

    def GetLifetime(self, n=0):
        """Gives lifetimes for defined k.
        Lifetime is computed as tau = (alpha*w*dw/dw0)^-1.
        The output is in s.

        Parameters
        ----------
        n : {-1, 0, 1, 2, ..., N-1}, optional
            Quantization number.  If -1, data for all (positive)
            calculated modes are returned.  Default is 0.

        Returns
        -------
        lifetime : ndarray
            (s ) lifetime.
        """
        w0_ori = self.w0
        step = 1e-5
        self.w0 = w0_ori * (1 - step)
        dw_lo, _ = self.GetDispersion()
        self.w0 = w0_ori * (1 + step)
        dw_hi, _ = self.GetDispersion()
        self.w0 = w0_ori
        w_mid, _ = self.GetDispersion()
        lifetime = (
            (self.alpha * w_mid + self.gamma * self.mu0dH0)
            * (dw_hi - dw_lo)
            / (w0_ori * 2 * step)
        ) ** -1
        if n != -1:
            return lifetime[n]
        return lifetime

    def GetDecLen(self, n=0):
        """Give decay lengths for defined k.
        Decay length is computed as lambda = v_g*tau.
        Output is given in m.

        .. warning::
            Works only when ``kxi.shape[0] >= 2``.

        Parameters
        ----------
        n : {-1, 0, 1, 2, ..., N-1}, optional
            Quantization number.  If -1, data for all (positive)
            calculated modes are returned.  Default is 0.

        Returns
        -------
        declen : ndarray
            (m ) decay length.
        """
        return self.GetLifetime(n=n) * self.GetGroupVelocity(n=n)

    def GetDensityOfStates(self, n=0):
        """Give density of states for given mode.
        Density of states is computed as DoS = 1/v_g.
        Output is density of states in 1D for given dispersion
        characteristics.

        .. warning::
            Works only when ``kxi.shape[0] >= 2``.

        Parameters
        ----------
        n : {-1, 0, 1, 2, ..., N-1}, optional
            Quantization number.  If -1, data for all (positive)
            calculated modes are returned.  Default is 0.

        Returns
        -------
        dos : ndarray
            (s/m) value proportional to density of states.
        """
        return 1 / self.GetGroupVelocity(n=n)

    def GetBlochFunction(self, n=0, Nf=200, temp=None, mu=None):
        """Gives Bloch function for given mode.

        Bloch function is calculated with margin of 10% of
        the lowest and the highest frequency (including
        Gilbert broadening).

        Parameters
        ----------
        n : {0, 1, 2, ..., N-1}, optional
            Quantization number.  The -1 value is not supported here.
            Default is 0.
        Nf : int, optional
            Number of frequency levels for the Bloch function.
        temp : float or None, optional
            (K ) temperature for the Bose-Einstein distribution.
            If `temp` or `mu` is None, no distribution is applied.
        mu : float or None, optional
            (J ) chemical potential for the Bose-Einstein distribution.
            If `temp` or `mu` is None, no distribution is applied.
            Otherwise, the Bloch function is weighted by
            ``sqrt(2*n_BE)``, where ``n_BE`` is the Bose-Einstein
            distribution (see :func:`~SpinWaveToolkit.distBE`).

        Returns
        -------
        w : ndarray
            (rad*Hz) frequency axis for the 2D Bloch function.
        blochFunc : ndarray
            () 2D Bloch function for given `w` and `kxi` with shape
            ``(Nf, kxi.shape[0])``.
        """
        w, _ = self.GetDispersion()
        lifeTime = self.GetLifetime(n=n)
        w00 = w[n]

        w = np.linspace(
            (np.min(w00) - 2 * np.pi * 1 / np.max(lifeTime)) * 0.9,
            (np.max(w00) + 2 * np.pi * 1 / np.max(lifeTime)) * 1.1,
            Nf,
        )
        wMat = np.tile(w, (len(lifeTime), 1)).T
        blochFunc = 1 / ((wMat - w00) ** 2 + (2 / lifeTime) ** 2)

        if temp is not None and mu is not None:
            # amplitude ~ sqrt(2*n_BE), i.e. BLS intensity ~ n_BE
            blochFunc *= np.sqrt(2 * distBE(w, temp=temp, mu=mu))[:, np.newaxis]

        return w, blochFunc

    def GetExchangeLen(self):
        """Calculate exchange length in meters from the parameter `A`."""
        return np.sqrt(self.A)

    def set_DE(self):
        """Changes angles theta and phi to match the Damon Eshbach
        geometry, i.e. M || y, Bext || y.
        """
        self.theta, self.phi = np.pi / 2, np.pi / 2

    def set_BV(self):
        """Changes angles theta and phi to match the backward volume
        geometry, i.e. M || x, Bext || x.
        """
        self.theta, self.phi = np.pi / 2, 0
