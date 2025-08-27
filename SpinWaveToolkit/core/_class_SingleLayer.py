"""
Single-layer dipole-exchange spin waves with arbitrary field & anisotropy.
Implements Kalinikos (1990) Eq. (15) and anisotropy block Eq. (17b).
"""

import numpy as np
from SpinWaveToolkit.helpers import MU0, roots


class SingleLayer:
    """
    Parameters
    ----------
    Bext : float
        External field magnitude (Tesla), i.e. μ0|H0|.
    material : object
        Must provide: Ms (A/m), Aex (J/m), alpha (), gamma (rad s^-1 T^-1),
        mu0dH0 (T) inhomogeneous broadening.
    d : float
        Film thickness (m).
    kxi : float or 1D array
        Tangential wavenumber(s) (rad/m).
    theta_M : float
        Polar tilt of equilibrium M from film normal +z (rad). Default π/2 (in-plane).
    phi_k : float
        In-plane angle of k relative to M (rad). Default π/2 (DE geometry).
    weff : float
        Effective waveguide width for nT>0 quantization (m).
    boundary_cond : {1,2,3,4}
        1: unpinned, 2: pinned, 3: long-wave unpinned, 4: partially pinned.
    dp : float
        Pinning parameter for BC=4, dp = Ks/Aex (rad/m).
    theta_H, phi_H : float
        External field orientation (rad). If None, field is taken collinear with M.
    Nd : (3,3) array
        Shape demag tensor in LAB (x,y in plane, z = film normal). Default diag(0,0,1).
    Na : (3,3) array
        Effective anisotropy “demag” tensor in LAB. Default zeros.
    Ku_perp : float or None
        Optional perpendicular uniaxial anisotropy density (J/m^3). If given,
        it adds Na_zz = -4*Ku/(μ0 Ms^2) (i.e. N^a_zz = -2 H_U / Ms with H_U=2Ku/(μ0 Ms)).

    Notes
    -----
    Dispersion uses Eq. (15): ω^2 = Q_n [ Q_n + ω_M (F_nn + F_nn^(a)) ],
    with the anisotropy block F_nn^(a) from Eq. (17b), evaluated in the M-frame (z' || M).
    """

    # -------------------------- init & constants --------------------------

    def __init__(self,
                 Bext,
                 material,
                 d,
                 kxi=np.linspace(1e-12, 25e6, 200),
                 theta_M=np.pi/2,
                 phi_k=np.pi/2,
                 weff=3e-6,
                 boundary_cond=1,
                 dp=0.0,
                 theta_H=None,
                 phi_H=0.0,
                 Nd=None,
                 Na=None,
                 Ku_perp=None):

        self.kxi = np.atleast_1d(kxi).astype(float)
        self.d = float(d)
        self.weff = float(weff)
        self.boundary_cond = int(boundary_cond)
        self.dp = float(dp)

        # Material
        self.Ms = float(material.Ms)
        self.Aex = float(material.Aex)
        self.alpha = float(material.alpha)
        self.gamma = float(material.gamma)
        self.mu0dH0 = float(material.mu0dH0)

        # Angles & geometry
        self.theta_M = float(theta_M)
        self.phi_k = float(phi_k)     # k–M in-plane angle used in Fnn (DE: pi/2)
        self.theta_H = self.theta_M if (theta_H is None) else float(theta_H)
        self.phi_H = float(phi_H)

        # Demag tensors in LAB (z = film normal)
        self.Nd = np.diag([0.0, 0.0, 1.0]) if Nd is None else np.array(Nd, dtype=float)
        self.Na = np.zeros((3, 3)) if Na is None else np.array(Na, dtype=float)

        # Optional perpendicular uniaxial (adds to Na in LAB)
        if Ku_perp is not None:
            Nazz = -4.0 * float(Ku_perp) / (MU0 * self.Ms**2)   # = -2 H_U / Ms
            self.Na = self.Na.copy()
            self.Na[2, 2] += Nazz

        # Slavin–Kalinikos constants
        self.wM = self.gamma * MU0 * self.Ms                 # rad/s
        self.A = 2.0 * self.Aex / (MU0 * self.Ms**2)         # m^2

        # External field magnitude (Tesla)
        self.Bext = float(Bext)

        # Compute internal static ω0 from projection along M (Kalinikos: Hi ∥ M)
        self.w0 = self._compute_w0()

    # -------------------------- small helpers -----------------------------

    @staticmethod
    def _sph_to_cart(theta, phi):
        st, ct = np.sin(theta), np.cos(theta)
        cp, sp = np.cos(phi), np.sin(phi)
        return np.array([st*cp, st*sp, ct], dtype=float)  # unit vector

    def _rotation_lab_to_Mframe(self):
        """Return R such that columns are (x',y',z') in LAB, with z' || M."""
        m = self._sph_to_cart(self.theta_M, 0.0)  # M azimuth irrelevant for k–M angle definition
        zlab = np.array([0.0, 0.0, 1.0])
        x_try = zlab - (zlab @ m) * m
        if np.linalg.norm(x_try) < 1e-14:  # M ∥ z
            x_try = np.cross(m, np.array([1.0, 0.0, 0.0]))
            if np.linalg.norm(x_try) < 1e-14:
                x_try = np.cross(m, np.array([0.0, 1.0, 0.0]))
        x = x_try / np.linalg.norm(x_try)
        y = np.cross(m, x)
        return np.column_stack([x, y, m])  # LAB basis columns

    def _tensor_in_Mframe(self, Tlab):
        R = self._rotation_lab_to_Mframe()
        return R.T @ Tlab @ R

    def _compute_w0(self):
        """ω0 = γ * (B_ext⋅m  - μ0 Ms mᵀNd m  - μ0 Ms mᵀNa m)."""
        mhat = self._sph_to_cart(self.theta_M, 0.0)
        hhat = self._sph_to_cart(self.theta_H, self.phi_H)
        B_par = self.Bext * float(mhat @ hhat)
        B_par += -MU0 * self.Ms * float(mhat @ self.Nd @ mhat)
        B_par += -MU0 * self.Ms * float(mhat @ self.Na @ mhat)
        return self.gamma * B_par

    # ------------------------- k-quantization -----------------------------

    def _kxi_total(self, nT):
        """Tangential k including transverse quantization nT."""
        if nT == 0:
            return self.kxi
        return np.sqrt(self.kxi**2 + (nT*np.pi/self.weff)**2)

    def _kappa_n(self, n):
        if self.boundary_cond != 4:
            return n * np.pi / self.d
        # Partially pinned: solve transverse equation
        def trans_eq(kappa, d, dp):  # Eq. for BC=4
            return (kappa**2 - dp**2) * np.tan(kappa*d) - 2*kappa*dp
        lo, hi = n*np.pi/self.d, (n+1)*np.pi/self.d
        kappa0 = roots(trans_eq, lo, hi, np.pi/self.d*4e-4, np.pi/self.d*1e-9,
                       args=(self.d, self.dp))
        # drop singularities and zeros, take the first valid root
        for i in range(n+1):
            bad = np.isclose(kappa0, np.pi/self.d*(i+0.5))
            kappa0[bad] = np.nan
        kappa0[kappa0 == 0.0] = np.nan
        kappa0 = kappa0[~np.isnan(kappa0)]
        return kappa0[0] if kappa0.size else 0.0

    # --------------------- dipolar kernels Pnn, Qnn -----------------------

    def _Pnn(self, n, nc, nT, kxi):
        if nc == -1:
            nc = n
        kappa, kappac = n*np.pi/self.d, nc*np.pi/self.d
        if self.boundary_cond == 4:
            kappa, kappac = self._kappa_n(n), self._kappa_n(nc)
        k = np.sqrt(kxi**2 + kappa**2)
        kc = np.sqrt(kxi**2 + kappac**2)

        if self.boundary_cond == 1:  # unpinned
            Fn = 2.0/(kxi*self.d) * (1.0 - (-1)**n * np.exp(-kxi*self.d))
            if n == 0 and nc == 0:
                P = (kxi**2)/(kc**2) - (kxi**4)/(k**2*kc**2) * 0.5 * ((1+(-1)**(n+nc))/2) * Fn
            elif (n == 0 and nc != 0) or (nc == 0 and n != 0):
                P = -(kxi**4)/(k**2*kc**2) * (1/np.sqrt(2)) * ((1+(-1)**(n+nc))/2) * Fn
            elif n == nc:
                P = (kxi**2)/(kc**2) - (kxi**4)/(k**2*kc**2) * ((1+(-1)**(n+nc))/2) * Fn
            else:
                P = -(kxi**4)/(k**2*kc**2) * ((1+(-1)**(n+nc))/2) * Fn

        elif self.boundary_cond == 2:  # pinned
            if n == nc:
                P = (kxi**2)/(kc**2) + (kxi**2)/(k**2) * (kappa*kappac)/(kc**2) \
                    * (1 + (-1)**(n+nc)/2) * 2/(kxi*self.d) * (1 - (-1)**n*np.exp(-kxi*self.d))
            else:
                P = (kxi**2)/(k**2) * (kappa*kappac)/(kc**2) * (1 + (-1)**(n+nc)/2) \
                    * 2/(kxi*self.d) * (1 - (-1)**n*np.exp(-kxi*self.d))

        elif self.boundary_cond == 3:  # long-wave unpinned
            P = (kxi*self.d/2.0) if (n == 0) else (kxi*self.d)**2/(n**2*np.pi**2)

        elif self.boundary_cond == 4:  # partially pinned (compact, robust form)
            dp = self.dp
            if kappa == 0:  kappa = 1e-9
            if kappac == 0: kappac = 1e-9
            k  = np.sqrt(kxi**2 + kappa**2)
            kc = np.sqrt(kxi**2 + kappac**2)
            An = np.sqrt(2 / ((kappa**2+dp**2)/kappa**2
                              + np.sin(kappa*self.d)/(kappa*self.d)
                              * ((kappa**2-dp**2)/kappa**2*np.cos(kappa*self.d)
                                 + 2*dp/kappa*np.sin(kappa*self.d))))
            Anc = np.sqrt(2 / ((kappac**2+dp**2)/kappac**2
                               + np.sin(kappac*self.d)/(kappac*self.d)
                               * ((kappac**2-dp**2)/kappac**2*np.cos(kappac*self.d)
                                  + 2*dp/kappac*np.sin(kappac*self.d))))
            term = (
                (kxi*(An*Anc))/(2*self.d*k**2*kc**2) * (
                    (kxi**2 - dp**2)*np.exp(-kxi*self.d)*(np.cos(kappa*self.d)+np.cos(kappac*self.d))
                    + (kxi - dp)*np.exp(-kxi*self.d)*(
                        (dp*kxi - kappa**2)*np.sin(kappa*self.d)/kappa
                        + (dp*kxi - kappac**2)*np.sin(kappac*self.d)/kappac
                    )
                    - (kxi**2 - dp**2)*(1 + np.cos(kappa*self.d)*np.cos(kappac*self.d))
                    + (kappa**2*kappac**2 - dp**2*kxi**2)
                      * np.sin(kappa*self.d)/kappa * np.sin(kappac*self.d)/kappac
                    - dp * ( k**2*np.cos(kappac*self.d)*np.sin(kappa*self.d)/kappa
                           + kc**2*np.cos(kappa*self.d)*np.sin(kappac*self.d)/kappac )
                )
            )
            P = kxi**2/kc**2 + term if (n == nc) else term
        else:
            raise ValueError("Unknown boundary condition.")
        return P

    def _Qnn(self, n, nc, nT, kxi):
        if nc == -1:
            nc = n
        kappa, kappac = n*np.pi/self.d, nc*np.pi/self.d
        if self.boundary_cond == 4:
            kappa, kappac = self._kappa_n(n), self._kappa_n(nc)
        if kappa == 0:  kappa = 1e-12
        if kappac == 0: kappac = 1e-12
        k = np.sqrt(kxi**2 + kappa**2)
        kc = np.sqrt(kxi**2 + kappac**2)

        if self.boundary_cond == 1:
            Fn = 2.0/(kxi*self.d) * (1 - (-1)**n * np.exp(-kxi*self.d))
            Q = (kxi**2/kc**2) * ( (kappac**2)/(kappac**2 - kappa**2) * 2/(kxi*self.d)
                                   - (kxi**2)/(2*k**2) * Fn ) * ((1-(-1)**(n+nc))/2)

        elif self.boundary_cond == 4:
            dp = self.dp
            An = np.sqrt(2 / ((kappa**2+dp**2)/kappa**2
                              + np.sin(kappa*self.d)/(kappa*self.d)
                              * ((kappa**2-dp**2)/kappa**2*np.cos(kappa*self.d)
                                 + 2*dp/kappa*np.sin(kappa*self.d))))
            Anc = np.sqrt(2 / ((kappac**2+dp**2)/kappac**2
                               + np.sin(kappac*self.d)/(kappac*self.d)
                               * ((kappac**2-dp**2)/kappac**2*np.cos(kappac*self.d)
                                  + 2*dp/kappac*np.sin(kappac*self.d))))
            Q = (kxi*An*Anc)/(2*self.d*k**2*kc**2) * (
                (kxi**2 - dp**2)*np.exp(-kxi*self.d)*(np.cos(kappa*self.d)-np.cos(kappac*self.d))
                + (kxi - dp)*np.exp(-kxi*self.d) * (
                    (dp*kxi - kappa**2)*np.sin(kappa*self.d)/kappa
                    - (dp*kxi - kappac**2)*np.sin(kappac*self.d)/kappac
                )
                + (kxi - dp) * (
                    (dp*kxi - kappac**2)*np.cos(kappa*self.d)*np.sin(kappac*self.d)/kappac
                    - (dp*kxi - kappa**2)*np.cos(kappac*self.d)*np.sin(kappa*self.d)/kappa
                )
                + (1
                   - np.cos(kappac*self.d)*np.cos(kappa*self.d)*2*(kxi**2*dp**2 + kappa**2*kappac**2
                      + (kappac**2+kappa**2)*(kxi**2+dp**2))/(kappac**2 - kappa**2)
                   - np.sin(kappa*self.d)*np.sin(kappac**2*self.d)/(kappa*kappac*(kappac**2-kappa**2))
                     * ( dp*kxi*(kappa**4+kappac**4)
                         + (dp**2*kxi**2 - kappa**2*kappac**2)*(kappa**2+kappac**2)
                         - 2*kappa**2*kappac**2*(dp**2 + kxi**2 - dp*kxi) )
                )
            )
        else:
            raise ValueError("Qnn is defined here for BC 1 and 4 only.")
        return Q

    # ---------------------------- F_nn (base + anisotropy) ----------------

    def _Fnn_total(self, n, nT, kxi):
        """
        Base F_nn (your implementation) + anisotropy correction F_nn^(a) (Eq. 17b).
        """
        # geometry and auxiliaries
        kappa = self._kappa_n(n) if self.boundary_cond == 4 else n*np.pi/self.d
        k = np.sqrt(kxi**2 + kappa**2)
        # in-plane angle used throughout (angle of k w.r.t. M in-plane)
        phi = np.arctan2((nT*np.pi/self.weff), self.kxi) - self.phi_k if nT != 0 else (0.0 - self.phi_k)
        # ensure array shape
        phi = np.broadcast_to(phi, self.kxi.shape)

        # Pnn base
        Pnn = self._Pnn(n=n, nc=n, nT=nT, kxi=kxi)

        # your original, dimensionless Fnn skeleton (kept for continuity)
        F_base = Pnn + np.sin(self.theta_M)**2 * (
            1.0
            - Pnn * (1.0 + np.cos(phi)**2)
            + (self.wM * (Pnn * (1.0 - Pnn) * np.sin(phi)**2)) / (self.w0 + self.A*self.wM*k**2)
        )

        # ---- anisotropy correction in M-frame (Eq. 17b) ----
        NaM = self._tensor_in_Mframe(self.Na)
        Nxx, Nyy, Nzz = NaM[0, 0], NaM[1, 1], NaM[2, 2]
        Nxz = NaM[0, 2]  # = Nzx

        Qn = self.w0 + self.A*self.wM*(k**2)
        sT, cT = np.sin(self.theta_M), np.cos(self.theta_M)

        Fa = (Nxx + Nyy
              + (self.wM/Qn) * (Nxx*Nyy + Nzz*(sT**2) - Nxz**2)
              + (self.wM/Qn) * Pnn * (
                    Nxx * (np.cos(phi)**2 - sT**2 * (1.0 + np.cos(phi)**2))
                  + Nyy * (np.sin(phi)**2)
                  - Nxz * (cT * np.sin(2.0*phi))
                )
             )

        return F_base + Fa

    # ----------------------------- dispersion etc. ------------------------

    def GetDispersion(self, n=0, nT=0):
        """ω(k) per Eq. (15) with F_nn + F_nn^(a). Returns rad/s."""
        kxi = self._kxi_total(nT)
        kappa = self._kappa_n(n) if self.boundary_cond == 4 else n*np.pi/self.d
        k = np.sqrt(kxi**2 + kappa**2)
        Qn = self.w0 + self.A*self.wM*(k**2)
        Ftot = self._Fnn_total(n=n, nT=nT, kxi=kxi)
        return np.sqrt(np.maximum(0.0, Qn * (Qn + self.wM*Ftot)))

    def GetGroupVelocity(self, n=0, nT=0):
        """v_g = dω/dk_tangential (m/s)."""
        w = self.GetDispersion(n=n, nT=nT)
        # finite difference over kxi grid
        return np.gradient(w, self.kxi)

    def GetLifetime(self, n=0, nT=0):
        """τ = [α ω (∂ω/∂ω0) + γ μ0 ΔH0]^-1 (s)."""
        w0 = self.w0
        step = 1e-5
        self.w0 = w0*(1 - step)
        w_lo = self.GetDispersion(n=n, nT=nT)
        self.w0 = w0*(1 + step)
        w_hi = self.GetDispersion(n=n, nT=nT)
        self.w0 = w0
        dw_dw0 = (w_hi - w_lo) / (2*step*w0)
        # avoid division by zero
        denom = np.abs(self.alpha * self.GetDispersion(n=n, nT=nT) * dw_dw0 + self.gamma * self.mu0dH0)
        denom = np.where(denom == 0, np.inf, denom)
        return 1.0 / denom

    def GetDecLen(self, n=0, nT=0):
        """Decay length λ = v_g * τ (m)."""
        return self.GetGroupVelocity(n=n, nT=nT) * self.GetLifetime(n=n, nT=nT)

    def GetDensityOfStates(self, n=0, nT=0):
        """1D density of states ∝ 1/|v_g| (s/m)."""
        vg = self.GetGroupVelocity(n=n, nT=nT)
        out = np.zeros_like(vg)
        nz = np.abs(vg) > 0
        out[nz] = 1.0 / vg[nz]
        out[~nz] = np.inf
        return out

    def GetExchangeLen(self):
        """Exchange length sqrt(A) in meters (from A = 2 Aex / (μ0 Ms^2))."""
        return np.sqrt(self.A)

    # Optional extras matching your previous API ---------------------------

    def GetBlochFunction(self, n=0, nT=0, Nf=200):
        """Simple Lorentzian Bloch map around the band with Gilbert broadening."""
        w0s = self.GetDispersion(n=n, nT=nT)
        tau = self.GetLifetime(n=n, nT=nT)
        w = np.linspace((np.min(w0s) - 2*np.pi*1/np.max(tau))*0.9,
                        (np.max(w0s) + 2*np.pi*1/np.max(tau))*1.1, Nf)
        W = np.tile(w, (len(tau), 1)).T
        return w, 1.0 / ((W - w0s)**2 + (2.0/tau)**2)

    # -------------------------- convenience setters -----------------------

    def set_external_field(self, Bext=None, theta_H=None, phi_H=None):
        """Update external field magnitude/orientation and recompute ω0."""
        if Bext is not None:
            self.Bext = float(Bext)
        if theta_H is not None:
            self.theta_H = float(theta_H)
        if phi_H is not None:
            self.phi_H = float(phi_H)
        self.w0 = self._compute_w0()

    def set_magnetization(self, theta_M=None, phi_k=None):
        """Update equilibrium M tilt and/or k–M in-plane angle; keep ω0 consistent."""
        if theta_M is not None:
            self.theta_M = float(theta_M)
        if phi_k is not None:
            self.phi_k = float(phi_k)
        self.w0 = self._compute_w0()

    def set_tensors(self, Nd=None, Na=None, Ku_perp=None):
        """Update demag/aniso tensors (LAB)."""
        if Nd is not None:
            self.Nd = np.array(Nd, dtype=float)
        if Na is not None:
            self.Na = np.array(Na, dtype=float)
        if Ku_perp is not None:
            Nazz = -4.0 * float(Ku_perp) / (MU0 * self.Ms**2)
            self.Na = self.Na.copy()
            self.Na[2, 2] += Nazz
        self.w0 = self._compute_w0()
