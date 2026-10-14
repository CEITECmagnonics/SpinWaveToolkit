"""Tests of `SpinWaveToolkit.DoubleLayerNumeric` (equilibrium, vectorized
dispersion, 2D dispersion of a synthetic antiferromagnet)."""

import numpy as np
import pytest
import SpinWaveToolkit as SWT
from SpinWaveToolkit.core._class_DoubleLayerNumeric import _zeta

# SAF of Adam et al., npj Spintronics 3, 33 (2025): CoFeB(17)/Ru(0.7)/CoFeB(17)
CFB = SWT.Material(Ms=1.35e6, Aex=16e-12, alpha=0.011, gamma=28 * 2e9 * np.pi)
D, S, JBL, B = 17e-9, 0.7e-9, -1.0e-3, 0.05
ALPHA = np.arccos(D * CFB.Ms * B / (2 * abs(JBL)))  # scissors half-angle


def _saf(**kwargs):
    kw = dict(kxi=np.array([1e6]), theta=np.pi / 2, phi=np.pi / 2, Jbl=JBL, s=S)
    kw.update(kwargs)
    return SWT.DoubleLayerNumeric(B, CFB, D, **kw)


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
        with cos(alpha) = d*Ms*B/(2|J|), reached from the default start."""
        saf = _saf()
        phis = saf.GetPhis()
        assert np.allclose(phis, [ALPHA, -ALPHA], atol=1e-8)
        assert np.allclose(saf.GetPhis(global_minimum=True), [ALPHA, -ALPHA], atol=1e-8)
        # the other scissors state from the mirrored start
        saf.phiInit1, saf.phiInit2 = -np.pi / 2, np.pi / 2
        assert np.allclose(saf.GetPhis(), [-ALPHA, ALPHA], atol=1e-8)
        assert np.allclose(saf.GetPhis(global_minimum=True), [-ALPHA, ALPHA], atol=1e-8)

    def test_saturation(self):
        saf = _saf()
        saf.Bext = 2 * abs(JBL) / (D * CFB.Ms) * 1.5
        assert np.allclose(saf.GetPhis(), [0, 0], atol=1e-8)

    def test_labile_start(self):
        """Antiparallel state along the field is a saddle point: the
        minimization leaves it deterministically."""
        saf = _saf(phiInit1=0.0, phiInit2=np.pi)
        assert np.allclose(saf.GetPhis(), [ALPHA, -ALPHA], atol=1e-8)

    def test_find_equilibria(self):
        """Two degenerate minima, three saddle points and one maximum
        (minima - saddles + maxima = 0 on the torus)."""
        saf = _saf()
        phi1, phi2, eden, eq_type = saf.find_equilibria()
        assert len(phi1) == 6
        assert np.all(np.diff(eden) >= 0)
        assert np.sum(eq_type == 0) == 2 and np.sum(eq_type == 2) == 3
        assert np.sum(eq_type == 1) == 1
        minima = sorted(zip(phi1[eq_type == 0], phi2[eq_type == 0]))
        assert np.allclose(minima, [[-ALPHA, ALPHA], [ALPHA, -ALPHA]], atol=1e-8)
        assert np.isclose(eden[0], saf.GetFreeEnergyIP([ALPHA, -ALPHA]))
        # zero torque at all equilibria
        grad = saf._DoubleLayerNumeric__ip_grad(np.array([phi1, phi2]))
        assert np.abs(grad).max() < 1e-12 * abs(JBL)

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


def test_energy_derivatives():
    """Analytical gradient and Hessian of the in-plane energy agree with
    finite differences (with both anisotropies and both couplings)."""
    saf = _saf(Ku=5e3, Ku2=2e3, Jbq=-0.1e-3, phiAnis1=0.4, phiAnis2=1.1)
    E = saf.GetFreeEnergyIP
    grad = saf._DoubleLayerNumeric__ip_grad
    hess = saf._DoubleLayerNumeric__ip_hess
    x = np.array([0.7, 2.9])
    eps = 1e-6
    num_g = [(E(x + eps * e) - E(x - eps * e)) / (2 * eps) for e in np.eye(2)]
    num_h = [(grad(x + eps * e) - grad(x - eps * e)) / (2 * eps) for e in np.eye(2)]
    assert np.allclose(grad(x), num_g, rtol=1e-6)
    assert np.allclose(hess(x), num_h, rtol=1e-6)
    # the general energy on unit vectors is the same in-plane
    assert np.isclose(
        saf.GetFreeEnergy(0.7, 2.9), saf._DoubleLayerNumeric__ip_energy(x)
    )


def test_free_energy():
    saf = _saf(Ku=5e3, Ku2=2e3, phiAnis1=0.4, phiAnis2=1.1)
    E0 = saf.GetFreeEnergy(0.7, 2.9)
    saf.Ku2 = 7e3  # (was ignored by the energy before)
    assert saf.GetFreeEnergy(0.7, 2.9) != E0
    # wrappers and vectorization
    assert saf.GetFreeEnergyIP([0.7, 2.9]) == saf.GetFreeEnergy(0.7, 2.9)
    assert saf.GetFreeEnergyOOP([1.2, 1.4], phis=[0.7, 2.9]) == saf.GetFreeEnergy(
        0.7, 2.9, 1.2, 1.4
    )
    phis = saf.GetPhis()
    assert saf.GetFreeEnergyOOP([1.2, 1.4]) == saf.GetFreeEnergy(*phis, 1.2, 1.4)
    p = np.linspace(0, 1, 5)
    assert saf.GetFreeEnergy(p, 2 * p).shape == (5,)
    # out of plane costs demagnetizing energy
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
        # (the model uses the gyromagnetic ratio of layer 1 for both layers)
        co = SWT.Material(Ms=1.15e6, Aex=20e-12, alpha=0.01, gamma=CFB.gamma)
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
            **kw,
        )
        assert np.allclose(b.GetPhis(), [-pa[1], -pa[0]], atol=1e-8)
        wa, va = a.GetDispersion()
        wb, vb = b.GetDispersion()
        assert np.allclose(wa, wb, rtol=1e-10)
        assert np.allclose(np.abs(va[[2, 3, 0, 1]]), np.abs(vb), atol=1e-8)
        # nonreciprocal: the test is not trivial
        assert np.abs(wa[0] - wa[0][::-1]).max() > 0.1 * wa[0].max()

    def test_given_phis(self):
        """`phis` overrides the equilibrium (e.g. the other scissors
        state), mirroring the nonreciprocity."""
        k = np.linspace(-30e6, 30e6, 13)
        saf = _saf(kxi=k, phi=np.pi / 2)
        w1 = saf.GetDispersion()[0]
        w2 = saf.GetDispersion(phis=[-ALPHA, ALPHA])[0]
        assert np.allclose(w1, w2[:, ::-1], rtol=1e-10)

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
