"""Tests of `SpinWaveToolkit.DoubleLayerNumeric` (equilibrium of the two
magnetizations, vectorized dispersion, 2D dispersion of a synthetic
antiferromagnet, tilted magnetizations)."""

import numpy as np
import pytest
from scipy.optimize import brentq
import SpinWaveToolkit as SWT
from SpinWaveToolkit.core._class_DoubleLayerNumeric import _local_frame, _zeta
from SpinWaveToolkit.helpers import MU0

# SAF of Adam et al., npj Spintronics 3, 33 (2025): CoFeB(17)/Ru(0.7)/CoFeB(17)
CFB = SWT.Material(Ms=1.35e6, Aex=16e-12, alpha=0.011, gamma=28 * 2e9 * np.pi)
D, S, JBL, B = 17e-9, 0.7e-9, -1.0e-3, 0.05
ALPHA = np.arccos(D * CFB.Ms * B / (2 * abs(JBL)))  # scissors half-angle
# single Py layer for the tilted-magnetization checks
PY = SWT.Material(Ms=800e3, Aex=13e-12, alpha=0.01, gamma=28 * 2e9 * np.pi)


def _saf(**kwargs):
    kw = dict(kxi=np.array([1e6]), theta=np.pi / 2, phi=np.pi / 2, Jbl=JBL, s=S)
    kw.update(kwargs)
    return SWT.DoubleLayerNumeric(B, CFB, D, **kw)


def _uncoupled(Bext, theta_H, phi, kxi=np.array([1e-3]), **kwargs):
    """Two identical uncoupled Py layers (no exchange coupling, spacer of
    1 m), i.e. two copies of a single layer."""
    kw = dict(
        kxi=kxi,
        theta=theta_H,
        phi=phi,
        Jbl=0,
        Jbq=0,
        s=1.0,
        d2=10e-9,
        thetaInit1=theta_H,
        thetaInit2=theta_H,
        phiInit1=0,
        phiInit2=0,
    )
    kw.update(kwargs)
    return SWT.DoubleLayerNumeric(Bext, PY, 10e-9, **kw)


@pytest.fixture(scope="module")
def grid():
    """2D wavevector grid and the corresponding `phi` for Bext along x."""
    kx = np.linspace(-40e6, 40e6, 21)
    kx[10] = 1e-3
    KX, KY = np.meshgrid(kx, kx, indexing="xy")
    return kx, KX, KY, np.hypot(KX, KY), -np.arctan2(KY, KX)


class TestEquilibrium:
    def test_scissors_state(self):
        """Field below the saturation field: symmetric scissors state
        with cos(alpha) = d*Ms*B/(2|J|), reached from the default start,
        in the plane of the film."""
        saf = _saf()
        assert np.allclose(saf.GetPhis(), [ALPHA, -ALPHA], atol=1e-8)
        assert np.allclose(saf.GetPhis(global_minimum=True), [ALPHA, -ALPHA], atol=1e-8)
        eq = saf.GetEquilibrium()
        assert np.allclose(eq, [np.pi / 2, ALPHA, np.pi / 2, -ALPHA], atol=1e-8)
        assert np.allclose(saf.GetEquilibrium(global_minimum=True), eq, atol=1e-8)
        # the other scissors state from the mirrored start
        saf.phiInit1, saf.phiInit2 = -np.pi / 2, np.pi / 2
        assert np.allclose(saf.GetPhis(), [-ALPHA, ALPHA], atol=1e-8)
        assert np.allclose(saf.GetPhis(global_minimum=True), [-ALPHA, ALPHA], atol=1e-8)
        assert np.allclose(
            saf.GetEquilibrium(), [np.pi / 2, -ALPHA, np.pi / 2, ALPHA], atol=1e-8
        )
        assert np.allclose(
            saf.GetEquilibrium(global_minimum=True),
            [np.pi / 2, -ALPHA, np.pi / 2, ALPHA],
            atol=1e-8,
        )

    def test_saturation(self):
        saf = _saf()
        saf.Bext = 2 * abs(JBL) / (D * CFB.Ms) * 1.5
        assert np.allclose(saf.GetPhis(), [0, 0], atol=1e-8)
        assert np.allclose(
            saf.GetEquilibrium(), [np.pi / 2, 0, np.pi / 2, 0], atol=1e-8
        )

    def test_labile_start(self):
        """Antiparallel state along the field is a saddle point: the
        minimization leaves it deterministically (same state from both
        methods)."""
        saf = _saf(phiInit1=0.0, phiInit2=np.pi)
        assert np.allclose(saf.GetPhis(), [ALPHA, -ALPHA], atol=1e-8)
        assert np.allclose(
            saf.GetEquilibrium(), [np.pi / 2, ALPHA, np.pi / 2, -ALPHA], atol=1e-8
        )

    def test_find_equilibria_in_plane(self):
        """Two degenerate minima, three saddle points and one maximum
        (minima - saddles + maxima = 0 on the torus)."""
        saf = _saf()
        th1, phi1, th2, phi2, eden, eq_type = saf.find_equilibria(in_plane=True)
        assert (
            len(phi1) == 6
            and np.allclose(th1, np.pi / 2)
            and np.allclose(th2, np.pi / 2)
        )
        assert np.all(np.diff(eden) >= 0)
        assert np.sum(eq_type == 0) == 2 and np.sum(eq_type == 2) == 3
        assert np.sum(eq_type == 1) == 1
        minima = sorted(zip(phi1[eq_type == 0], phi2[eq_type == 0]))
        assert np.allclose(minima, [[-ALPHA, ALPHA], [ALPHA, -ALPHA]], atol=1e-8)
        assert np.isclose(eden[0], saf.GetFreeEnergy(ALPHA, -ALPHA))
        # zero torque at all equilibria
        grad = saf._DoubleLayerNumeric__ip_grad(np.array([phi1, phi2]))
        assert np.abs(grad).max() < 1e-12 * abs(JBL)

    def test_find_equilibria_general(self):
        """All stationary points on the two spheres: the in-plane ones
        are included (the two scissors states are the only minima), plus
        out-of-plane saddle points and maxima."""
        saf = _saf()
        th1, phi1, th2, phi2, eden, eq_type = saf.find_equilibria()
        assert np.all(np.diff(eden) >= 0)
        assert np.sum(eq_type == 0) == 2 and np.sum(eq_type == 1) == 4
        assert np.sum(eq_type == 2) >= 12
        in_plane = np.isclose(th1, np.pi / 2) & np.isclose(th2, np.pi / 2)
        assert in_plane.sum() == 6
        _, p1, _, p2, e_ip, _ = saf.find_equilibria(in_plane=True)
        for a, b, e in zip(p1, p2, e_ip):
            i = np.argmin(np.abs(phi1 - a) + np.abs(phi2 - b) + ~in_plane)
            assert abs(phi1[i] - a) < 1e-6 and abs(phi2[i] - b) < 1e-6
            assert np.isclose(eden[i], e)
        assert np.allclose(sorted(phi1[eq_type == 0]), [-ALPHA, ALPHA], atol=1e-8)
        # zero torque at all equilibria
        m1, m2 = SWT.sphr2cart(th1, phi1), SWT.sphr2cart(th2, phi2)
        G1, G2 = saf._DoubleLayerNumeric__vec_grad(m1, m2)
        torque = np.linalg.norm(np.cross(m1, G1, axis=0), axis=0)
        torque2 = np.linalg.norm(np.cross(m2, G2, axis=0), axis=0)
        assert max(torque.max(), torque2.max()) < 1e-10 * abs(JBL)

    def test_anisotropy_equilibrium(self):
        """Without field, the layers lie antiparallel along their easy
        axes (here at 30 deg from the field direction)."""
        saf = _saf(
            Ku=5e3,
            Ku2=5e3,
            phiAnis1=0.5,
            phiAnis2=0.5,
            phiInit1=0.6,
            phiInit2=0.6 - np.pi,
        )
        saf.Bext = 0.0
        assert np.allclose(saf.GetPhis(), [0.5, 0.5 - np.pi], atol=1e-8)
        assert np.allclose(
            saf.GetEquilibrium(), [np.pi / 2, 0.5, np.pi / 2, 0.5 - np.pi], atol=1e-8
        )

    def test_oblique_field(self):
        """Field at 45 deg from the normal, in-plane component above the
        saturation field: both layers parallel and tilted out of the
        plane like a single film, B*sin(theta_H - theta_M) =
        MU0*Ms*sin(theta_M)*cos(theta_M)."""
        Bext, thH = 0.3, np.deg2rad(45)
        saf = _saf(theta=thH)
        saf.Bext = Bext
        eq = saf.GetEquilibrium()
        thM = brentq(
            lambda t: Bext * np.sin(t - thH) - MU0 * CFB.Ms * np.sin(t) * np.cos(t),
            thH,
            np.pi / 2,
        )
        assert np.allclose(eq, [thM, 0, thM, 0], atol=1e-8)
        assert np.allclose(saf.GetEquilibrium(global_minimum=True), eq, atol=1e-8)
        # the in-plane state is not the equilibrium
        assert saf.GetFreeEnergy(0, 0, thM, thM) < saf.GetFreeEnergy(0, 0)
        assert np.allclose(
            saf.GetPhis(), [0, 0], atol=1e-8
        )  # (restricted to the plane)

    def test_perpendicular_anisotropy(self):
        """Ks above the demagnetizing energy: out-of-plane state at zero
        field (both layers antiparallel along the normal for J < 0)."""
        saf = _saf(Ks1=1.3 * 0.5 * MU0 * CFB.Ms**2, Ks2=1.3 * 0.5 * MU0 * CFB.Ms**2)
        saf.Bext = 0.0
        eq = saf.GetEquilibrium()
        assert np.allclose(np.sort(eq[[0, 2]]), [0, np.pi], atol=1e-8)
        assert np.isclose(
            saf.GetFreeEnergy(0, 0, 0, np.pi),
            saf.GetFreeEnergy(*eq[[1, 3]], *eq[[0, 2]]),
        )


def test_energy_derivatives():
    """Analytical gradient and Hessian agree with finite differences,
    for the in-plane energy and for the energy in the local frames of
    the two magnetizations (all interactions active)."""
    saf = _saf(
        Ku=5e3, Ku2=2e3, Jbq=-0.1e-3, phiAnis1=0.4, phiAnis2=1.1, Ks1=1e4, Ks2=3e4
    )
    saf.theta = 0.9
    eps = 1e-6
    # in-plane
    E = lambda x: saf.GetFreeEnergy(x[0], x[1])
    grad = saf._DoubleLayerNumeric__ip_grad
    hess = saf._DoubleLayerNumeric__ip_hess
    x = np.array([0.7, 2.9])
    num_g = [(E(x + eps * e) - E(x - eps * e)) / (2 * eps) for e in np.eye(2)]
    num_h = [(grad(x + eps * e) - grad(x - eps * e)) / (2 * eps) for e in np.eye(2)]
    assert np.allclose(grad(x), num_g, rtol=1e-6)
    assert np.allclose(hess(x), num_h, rtol=1e-6)
    assert np.isclose(
        saf.GetFreeEnergy(0.7, 2.9), saf._DoubleLayerNumeric__ip_energy(x)
    )
    # general (local frames)
    R1, R2 = _local_frame(SWT.sphr2cart(1.0, 0.4)), _local_frame(
        SWT.sphr2cart(2.0, -1.0)
    )
    fun, jac, hess = saf._DoubleLayerNumeric__local_functions(R1, R2, 1.0)
    x = np.array([1.3, 0.2, 1.7, -0.3])
    num_g = [(fun(x + eps * e) - fun(x - eps * e)) / (2 * eps) for e in np.eye(4)]
    num_h = [(jac(x + eps * e) - jac(x - eps * e)) / (2 * eps) for e in np.eye(4)]
    assert np.allclose(jac(x), num_g, rtol=1e-6, atol=1e-12)
    assert np.allclose(hess(x), num_h, rtol=1e-6, atol=1e-12)


def test_free_energy():
    saf = _saf(Ku=5e3, Ku2=2e3, phiAnis1=0.4, phiAnis2=1.1)
    E0 = saf.GetFreeEnergy(0.7, 2.9)
    saf.Ku2 = 7e3  # (was ignored by the energy before)
    assert saf.GetFreeEnergy(0.7, 2.9) != E0
    # perpendicular anisotropy lowers the cost of tilting out of the plane
    dE = saf.GetFreeEnergy(0.7, 2.9, 1.2, 1.4) - saf.GetFreeEnergy(0.7, 2.9)
    saf.Ks1 = 2e4
    dE_Ks = saf.GetFreeEnergy(0.7, 2.9, 1.2, 1.4) - saf.GetFreeEnergy(0.7, 2.9)
    assert np.isclose(dE - dE_Ks, D * 2e4 * np.cos(1.2) ** 2)
    assert np.isclose(saf.Hs1, 2 * 2e4 / (MU0 * CFB.Ms))
    # deprecated wrappers
    with pytest.warns(DeprecationWarning):
        assert saf.GetFreeEnergyIP([0.7, 2.9]) == saf.GetFreeEnergy(0.7, 2.9)
    with pytest.warns(DeprecationWarning):
        assert saf.GetFreeEnergyOOP([1.2, 1.4], phis=[0.7, 2.9]) == saf.GetFreeEnergy(
            0.7, 2.9, 1.2, 1.4
        )
    phis = saf.GetPhis()
    with pytest.warns(DeprecationWarning):
        assert saf.GetFreeEnergyOOP([1.2, 1.4]) == saf.GetFreeEnergy(*phis, 1.2, 1.4)
    # vectorization and the demagnetizing energy
    p = np.linspace(0, 1, 5)
    assert saf.GetFreeEnergy(p, 2 * p).shape == (5,)
    assert saf.GetFreeEnergy(0.7, 2.9, 0.3, 0.3) > saf.GetFreeEnergy(0.7, 2.9)


def test_zeta():
    k = np.array([-2e7, -1e-20, 0.0, 1e3, 3e7])
    assert np.allclose(
        _zeta(k, 17e-9)[[0, 4]],
        (1 - np.exp(-abs(k[[0, 4]]) * 17e-9)) / (abs(k[[0, 4]]) * 17e-9),
    )
    assert np.allclose(_zeta(k, 17e-9)[1:4], 1)


class TestDispersion:
    def test_adam_2d_map(self, grid):
        """2D dispersion of the Adam et al. SAF at 50 mT along x: uniform
        mode at 6.34 GHz, minimum at negative kx (bottom layer 1 has the
        positive-y component), reciprocal in ky, nonreciprocal in kx."""
        kx, KX, KY, K, PHI = grid
        saf = _saf(kxi=K.ravel(), phi=PHI.ravel())
        w, v = saf.GetDispersion()
        assert w.shape == (2, K.size) and v.shape == (4, 2, K.size)
        f = w.reshape(2, *K.shape) / (2e9 * np.pi)
        assert np.all(f[1] > f[0])
        assert abs(f[0, 10, 10] - 6.34) < 0.05  # (Adam et al.: 6.34 GHz)
        assert np.abs(f[0] - f[0][::-1]).max() < 1e-6  # f(kx, ky) = f(kx, -ky)
        i20 = np.argmin(abs(kx - 20e6))
        assert f[0, 10, 20 - i20] < 4.5 and f[0, 10, i20] > 12  # nonreciprocal in kx
        assert KX.ravel()[f[0].argmin()] < -10e6
        assert f[0, 10, 20 - i20] < f[0, 10, 10]  # f(-20, 0) < f(0, 0) < f(0, 20)
        assert f[0, 10, 10] < f[0, i20, 10]
        # the reference values of the Gallardo matrix for in-plane magnetizations
        assert np.allclose(
            [f[0, 10, 20 - i20], f[0, 10, i20], f[0, -1, 10]],
            [4.119, 13.493, 12.192],
            atol=2e-3,
        )

    def test_vectorized_vs_per_point(self, grid):
        _, _, _, K, PHI = grid
        saf = _saf(kxi=K.ravel(), phi=PHI.ravel())
        w, v = saf.GetDispersion()
        for i in (0, 57, 220, 440):
            one = _saf(kxi=np.array([K.ravel()[i]]), phi=PHI.ravel()[i])
            w1, v1 = one.GetDispersion()
            assert np.allclose(w1[:, 0], w[:, i], rtol=1e-12)
            assert np.allclose(np.abs(v1[:, :, 0]), np.abs(v[:, :, i]), atol=1e-10)
        # scalar kxi with an array of phi
        phis = np.linspace(0, 2 * np.pi, 7)
        w2 = _saf(kxi=5e6, phi=phis).GetDispersion()[0]
        assert w2.shape == (2, 7)
        assert np.allclose(w2[:, 0], w2[:, -1])

    def test_wavevector_sign(self):
        """The same wavevector is described by (kxi, phi) and (-kxi, phi + pi)."""
        k = np.linspace(-30e6, 30e6, 13)
        w1 = _saf(kxi=k, phi=0.7).GetDispersion()[0]
        w2 = _saf(kxi=-k, phi=0.7 + np.pi).GetDispersion()[0]
        assert np.allclose(w1, w2, rtol=1e-12)

    def test_layer_swap(self):
        """Rotating the stack by 180 deg about the wavevector swaps the
        layers and reverses the in-plane angles: same frequencies."""
        k = np.linspace(-30e6, 30e6, 13)
        co = SWT.Material(Ms=1.15e6, Aex=20e-12, alpha=0.01, gamma=1.7587e11)
        kw = dict(theta=np.pi / 2, Jbl=-0.5e-3, Jbq=-0.1e-3, s=1e-9)
        a = SWT.DoubleLayerNumeric(
            0.03,
            co,
            5e-9,
            kxi=k,
            phi=0.7,
            Ku=3e4,
            Ku2=1e3,
            d2=8e-9,
            material2=CFB,
            phiAnis1=0.3,
            phiAnis2=-0.9,
            phiInit1=0.2,
            phiInit2=2.5,
            Ks1=2e4,
            Ks2=5e4,
            **kw,
        )
        pa = a.GetPhis()
        b = SWT.DoubleLayerNumeric(
            0.03,
            CFB,
            8e-9,
            kxi=k,
            phi=-0.7,
            Ku=1e3,
            Ku2=3e4,
            d2=5e-9,
            material2=co,
            phiAnis1=0.9,
            phiAnis2=-0.3,
            phiInit1=-pa[1],
            phiInit2=-pa[0],
            Ks1=5e4,
            Ks2=2e4,
            **kw,
        )
        assert np.allclose(b.GetPhis(), [-pa[1], -pa[0]], atol=1e-8)
        wa, va = a.GetDispersion()
        wb, vb = b.GetDispersion()
        assert np.allclose(wa, wb, rtol=1e-10)
        assert np.allclose(np.abs(va[[2, 3, 0, 1]]), np.abs(vb), atol=1e-8)
        # nonreciprocal: the test is not trivial
        assert np.abs(wa[0] - wa[0][::-1]).max() > 0.1 * wa[0].max()

    def test_given_angles(self):
        """`angles` overrides the equilibrium (e.g. the other scissors
        state), mirroring the nonreciprocity."""
        k = np.linspace(-30e6, 30e6, 13)
        saf = _saf(kxi=k, phi=np.pi / 2)
        w1 = saf.GetDispersion()[0]
        w2 = saf.GetDispersion(angles=[np.pi / 2, -ALPHA, np.pi / 2, ALPHA])[0]
        assert np.allclose(w1, w2[:, ::-1], rtol=1e-10)

    @pytest.mark.parametrize(
        "theta_H_deg, Bext, phi",
        [(60, 0.5, 0.7), (30, 0.8, np.pi / 2), (80, 0.1, 0.0), (10, 1.2, 1.0)],
    )
    def test_tilted_single_layer(self, theta_H_deg, Bext, phi):
        """Uncoupled identical layers in an oblique field: the frequencies
        equal those of `SingleLayer` (n = 0, same thickness-uniform
        model) for the tilted magnetization, and the Smit-Beljers
        formula of a thin film at k = 0."""
        thH = np.deg2rad(theta_H_deg)
        k = np.linspace(-20e6, 20e6, 9)
        k[4] = 1e-3
        dl = _uncoupled(Bext, thH, phi, kxi=k)
        thM, phM = dl.GetEquilibrium()[:2]
        assert thH < thM < np.pi / 2 and abs(phM) < 1e-10
        f = dl.GetDispersion()[0]
        assert np.allclose(f[0], f[1], rtol=1e-10)  # two identical uncoupled layers
        sl = SWT.SingleLayer(
            Bext, PY, 10e-9, kxi=k, theta=thM, phi=phi + phM, theta_H=thH, phi_H=phi
        )
        assert np.allclose(f[0], sl.GetDispersion(n=0), rtol=1e-6)
        H = Bext / MU0
        a = H * np.cos(thH - thM)
        w_sb = (
            PY.gamma
            * MU0
            * np.sqrt((a - PY.Ms * np.cos(2 * thM)) * (a - PY.Ms * np.cos(thM) ** 2))
        )
        assert np.isclose(f[0, 4], w_sb, rtol=1e-8)

    def test_per_layer_gamma(self):
        """Each layer precesses with its own gyromagnetic ratio: uncoupled
        layers of different materials give the two single-layer modes."""
        k = np.linspace(-20e6, 20e6, 9)
        k[4] = 1e-3
        co = SWT.Material(Ms=1.15e6, Aex=20e-12, alpha=0.01, gamma=1.7587e11)
        dl = _uncoupled(
            0.2,
            np.pi / 2,
            0.7,
            kxi=k,
            material2=co,
            Ku=2e3,
            Ku2=5e3,
            phiAnis1=0.0,
            phiAnis2=0.0,
        )
        assert dl.gamma == PY.gamma and dl.gamma2 == co.gamma
        f = dl.GetDispersion()[0]
        # (anisotropy axes along the field, at 0.7 rad from k in the frame of SingleLayer)
        uu = np.outer(*[[np.cos(0.7), np.sin(0.7), 0]] * 2)
        f_py = SWT.SingleLayer(
            0.2, PY, 10e-9, kxi=k, phi=0.7, Na=-2 * 2e3 / (MU0 * PY.Ms**2) * uu
        ).GetDispersion()
        f_co = SWT.SingleLayer(
            0.2, co, 10e-9, kxi=k, phi=0.7, Na=-2 * 5e3 / (MU0 * co.Ms**2) * uu
        ).GetDispersion()
        expected = np.sort(np.stack([f_py, f_co]), axis=0)
        assert np.allclose(f, expected, rtol=1e-6)
        # the frequencies scale with the gyromagnetic ratios
        dl.gamma2 *= 1.1
        f2 = dl.GetDispersion()[0]
        assert np.allclose(np.sort(np.stack([f_py, 1.1 * f_co]), axis=0), f2, rtol=1e-6)

    def test_perpendicular_anisotropy_kittel(self):
        """Ks reduces the effective magnetization in the Kittel formula
        of the in-plane uniform mode."""
        Ks = 0.2 * 0.5 * MU0 * PY.Ms**2
        dl = _uncoupled(0.1, np.pi / 2, np.pi / 2, Ks1=Ks, Ks2=Ks)
        f0 = dl.GetDispersion()[0][0, 0]
        Meff = PY.Ms - 2 * Ks / (MU0 * PY.Ms)
        assert np.isclose(
            f0, PY.gamma * MU0 * np.sqrt((0.1 / MU0) * (0.1 / MU0 + Meff)), rtol=1e-8
        )

    def test_continuity_in_plane_limit(self):
        """A field 1e-6 rad out of the plane changes nothing."""
        saf = _saf(kxi=np.array([5e6]), theta=np.pi / 2 - 1e-6, phi=0.7)
        f1 = saf.GetDispersion()[0][:, 0]
        saf.theta = np.pi / 2
        f2 = saf.GetDispersion()[0][:, 0]
        assert np.allclose(f1, f2, rtol=1e-9)

    @pytest.mark.filterwarnings("ignore:divide by zero")
    def test_derived_quantities_zero_field(self):
        """The docstring example (zero field) runs; the lifetime formula
        is not meaningful for the antiparallel state (see GetLifetime)."""
        kxi = np.linspace(1e-6, 150e6, 30)
        dln = SWT.DoubleLayerNumeric(
            Bext=0,
            material=SWT.NiFe,
            d=30e-9,
            kxi=kxi,
            phi=np.pi / 2,
            Ku=3e3,
            Jbl=-1e-3,
            s=5e-10,
            phiInit1=0,
            phiInit2=np.pi,
        )
        assert np.allclose(dln.GetPhis(), [0, -np.pi], atol=1e-8)
        f = dln.GetDispersion()[0]
        assert np.all(f > 0) and np.all(np.isfinite(f))
        for q in (dln.GetGroupVelocity(), dln.GetDensityOfStates()):
            assert q.shape == (30,) and np.all(np.isfinite(q))
        for q in (dln.GetLifetime(), dln.GetDecLen(), dln.GetLifetime(n=-1)[1]):
            assert q.shape == (30,)
        w, bloch = dln.GetBlochFunction(Nf=50, lifeTime=1e-9)
        assert bloch.shape == (50, 30) and np.all(np.isfinite(bloch))
