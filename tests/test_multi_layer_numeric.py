"""Tests of `SpinWaveToolkit.MultiLayerNumeric` (equivalence with
`DoubleLayerNumeric` and `SingleLayer`, equilibria of several
magnetizations, dispersion of multilayers, a thick layer modelled by
exchange-coupled sublayers)."""

import numpy as np
import pytest
import SpinWaveToolkit as SWT
from SpinWaveToolkit.core._class_MultiLayerNumeric import _angle_functions
from SpinWaveToolkit.core._sphere import local_frame
from SpinWaveToolkit.helpers import MU0

CFB = SWT.Material(Ms=1.35e6, Aex=16e-12, alpha=0.011, gamma=28 * 2e9 * np.pi)
PY = SWT.Material(Ms=800e3, Aex=13e-12, alpha=0.01, gamma=29 * 2e9 * np.pi)
CO = SWT.Material(Ms=1.15e6, Aex=20e-12, alpha=0.01, gamma=1.7587e11)


def _pair(theta, **kwargs):
    """The same CoFeB/Py bilayer as `DoubleLayerNumeric` and as
    `MultiLayerNumeric` (all interactions active)."""
    k = np.linspace(-40e6, 40e6, 21)
    kw = dict(kxi=k, theta=theta, phi=0.7, Jbl=-0.6e-3, Jbq=-0.1e-3, s=0.8e-9)
    kw.update(kwargs)
    dln = SWT.DoubleLayerNumeric(
        0.05,
        CFB,
        12e-9,
        d2=8e-9,
        material2=PY,
        Ku=2e3,
        Ku2=1e3,
        phiAnis1=0.3,
        phiAnis2=-0.2,
        Ks1=1e5,
        Ks2=3e5,
        **kw,
    )
    mln = SWT.MultiLayerNumeric(
        0.05,
        [CFB, PY],
        [12e-9, 8e-9],
        Ku=[2e3, 1e3],
        phiAnis=[0.3, -0.2],
        Ks=[1e5, 3e5],
        **kw,
    )
    return dln, mln


def _trilayer(**kwargs):
    """CoFeB/Py/Co trilayer with different couplings of the interfaces."""
    kw = dict(
        kxi=np.linspace(-30e6, 30e6, 13),
        theta=1.2,
        phi=0.7,
        s=[0.8e-9, 1.1e-9],
        Jbl=[-0.4e-3, -0.2e-3],
        Jbq=[-0.05e-3, 0.0],
        Ku=[2e3, 0.0, 1e4],
        phiAnis=[0.3, 0.0, -0.9],
        Ks=[1e5, 0.0, 3e5],
    )
    kw.update(kwargs)
    return SWT.MultiLayerNumeric(0.04, [CFB, PY, CO], [6e-9, 4e-9, 5e-9], **kw)


class TestParameters:
    def test_number_of_layers(self):
        ml = SWT.MultiLayerNumeric(0.1, PY, [5e-9] * 4, s=1e-9, Jbl=[1e-3, 0, -1e-3])
        assert ml.nlayers == 4
        assert ml.s.shape == (3,) and np.allclose(ml.s, 1e-9)
        assert np.allclose(ml.Jbl, [1e-3, 0, -1e-3]) and np.allclose(ml.JblDyn, ml.Jbl)
        assert np.allclose(ml.Ms, PY.Ms) and ml.Ms.shape == (4,)
        assert np.allclose(ml.phiInit, [np.pi / 2, -np.pi / 2] * 2)
        # number of layers from the interface parameters only
        assert SWT.MultiLayerNumeric(0.1, PY, 5e-9, Jbl=[0.0, 0.0]).nlayers == 3
        # derived parameters follow the changes
        ml.Ks = [1e4, 0, 0, 0]
        assert np.isclose(ml.Hs[0], 2 * 1e4 / (MU0 * PY.Ms)) and ml.Hs[1] == 0
        ml.Aex = 2 * PY.Aex
        assert np.allclose(ml.A, 4 * PY.Aex / (MU0 * PY.Ms**2))
        assert np.allclose(ml.GetExchangeLen(), np.sqrt(ml.A))

    def test_invalid(self):
        with pytest.raises(ValueError):  # no sequence: number of layers unknown
            SWT.MultiLayerNumeric(0.1, PY, 5e-9)
        with pytest.raises(ValueError):  # inconsistent lengths
            SWT.MultiLayerNumeric(0.1, PY, [5e-9] * 3, Jbl=[0.0])
        ml = SWT.MultiLayerNumeric(0.1, PY, [5e-9] * 3)
        with pytest.raises(ValueError):
            ml.Ku = [1.0, 2.0]
        with pytest.raises(ValueError):
            ml.GetFreeEnergy([0.0, 0.0])


class TestDoubleLayerEquivalence:
    @pytest.mark.parametrize("theta", [np.pi / 2, 1.0])
    def test_same_as_double_layer(self, theta):
        """For two layers, the results are the same as those of
        `DoubleLayerNumeric` (in-plane and oblique field)."""
        dln, mln = _pair(theta)
        a = dln.GetEquilibrium()
        b = mln.GetEquilibrium()
        assert b.shape == (2, 2)
        assert np.allclose(a, b.T.ravel(), atol=1e-10)
        assert np.allclose(dln.GetPhis(), mln.GetPhis(), atol=1e-10)
        w1, v1 = dln.GetDispersion()
        w2, v2 = mln.GetDispersion()
        assert w2.shape == (2, 21) and v2.shape == (4, 2, 21)
        assert np.allclose(w1, w2, rtol=1e-12)
        assert np.allclose(np.abs(v1), np.abs(v2), atol=1e-10)
        assert np.isclose(
            dln.GetFreeEnergy(0.3, 0.5, 1.0, 2.0),
            mln.GetFreeEnergy([0.3, 0.5], [1.0, 2.0]),
        )
        assert np.allclose(dln.GetGroupVelocity(n=-1), mln.GetGroupVelocity(n=-1))
        assert np.allclose(dln.GetLifetime(n=-1), mln.GetLifetime(n=-1))

    def test_equilibria_as_double_layer(self):
        dln, mln = _pair(1.0)
        th1, ph1, th2, ph2, eden, eq_type = dln.find_equilibria()
        theta, phi, eden2, eq_type2 = mln.find_equilibria()
        assert theta.shape == (2, len(th1))
        assert np.allclose(eden, eden2, rtol=1e-10) and np.all(eq_type == eq_type2)
        assert np.allclose(theta[:, 0], [th1[0], th2[0]], atol=1e-8)
        assert np.allclose(phi[:, 0], [ph1[0], ph2[0]], atol=1e-8)
        assert np.allclose(
            dln.GetEquilibrium(global_minimum=True),
            mln.GetEquilibrium(global_minimum=True).T.ravel(),
            atol=1e-8,
        )
        eden_ip = mln.find_equilibria(in_plane=True)[2]
        assert np.allclose(dln.find_equilibria(in_plane=True)[4], eden_ip, rtol=1e-10)
        assert np.allclose(
            dln.GetPhis(global_minimum=True), mln.GetPhis(global_minimum=True)
        )


class TestEquilibrium:
    def test_equilibria_trilayer(self):
        """All found equilibria have zero torque, are sorted by energy,
        the global minimum is a minimum, and the local minimization
        from the global minimum stays there."""
        ml = _trilayer()
        theta, phi, eden, eq_type = ml.find_equilibria()
        assert np.all(np.diff(eden) >= 0) and eq_type[0] == 0
        assert np.allclose(eden, ml.GetFreeEnergy(phi, theta))
        energy = ml._MultiLayerNumeric__energy()
        ms = list(SWT.sphr2cart(theta, phi).swapaxes(0, 1))
        torque = [np.cross(m, G, axis=0) for m, G in zip(ms, energy.grad(ms))]
        assert np.abs(torque).max() < 1e-10 * energy.scale
        glob = ml.GetEquilibrium(global_minimum=True)
        assert np.allclose(glob, [theta[:, 0], phi[:, 0]], atol=1e-8)
        ml.thetaInit, ml.phiInit = glob
        assert np.allclose(ml.GetEquilibrium(), glob, atol=1e-8)
        # the local minimum from the default start is one of the minima
        ml.thetaInit, ml.phiInit = np.pi / 2, [np.pi / 2, -np.pi / 2, np.pi / 2]
        loc = ml.GetEquilibrium()
        d = np.abs(theta - loc[0][:, None]).sum(0) + np.abs(phi - loc[1][:, None]).sum(
            0
        )
        assert eq_type[np.argmin(d)] == 0 and d.min() < 1e-6

    def test_saturation(self):
        """Above the saturation field, all layers along the field."""
        ml = SWT.MultiLayerNumeric(1.0, PY, [5e-9] * 5, s=1e-9, Jbl=-0.2e-3)
        assert np.allclose(ml.GetPhis(), 0, atol=1e-8)
        assert np.allclose(ml.GetEquilibrium(), [[np.pi / 2] * 5, [0] * 5], atol=1e-8)

    def test_antiferromagnetic_chain(self):
        """Zero field, easy axes along x and antiferromagnetic coupling:
        the layers alternate along the easy axis, from any in-plane
        start in the same quadrants (labile start from the hard axis)."""
        ml = SWT.MultiLayerNumeric(
            0.0, PY, [5e-9] * 4, s=1e-9, Jbl=-0.5e-3, Ku=1e4, phiInit=[0, np.pi] * 2
        )
        eq = np.abs(ml.GetEquilibrium())
        assert np.allclose(eq, [[np.pi / 2] * 4, [0, np.pi] * 2], atol=1e-8)
        ml.phiInit = [0.2, 2.9, -0.1, 3.0]
        assert np.allclose(np.abs(ml.GetPhis()), [0, np.pi] * 2, atol=1e-8)
        ml.phiInit = np.pi / 2  # (all along the hard axis, saddle point)
        phis = ml.GetPhis()
        assert np.allclose(np.cos(phis), [1, -1] * 2, atol=1e-8) or np.allclose(
            np.cos(phis), [-1, 1] * 2, atol=1e-8
        )

    def test_many_layers_random_starts(self):
        """For many layers, the starting points are sampled randomly and
        the minima are still found."""
        ml = SWT.MultiLayerNumeric(0.05, PY, [5e-9] * 6, s=0.7e-9, Jbl=-0.3e-3)
        theta, _, eden, eq_type = ml.find_equilibria(in_plane=True, max_starts=3000)
        assert np.sum(eq_type == 0) >= 1 and np.allclose(theta, np.pi / 2)
        eq = ml.GetPhis()
        assert ml.GetFreeEnergy(eq) >= eden[0] - 1e-12 * abs(eden[0])


def test_energy_derivatives():
    """Analytical gradient and Hessian agree with finite differences,
    for the energy in the local frames of the magnetizations and
    restricted to the film plane (all interactions active)."""
    ml = _trilayer()
    eps = 1e-6
    energy = ml._MultiLayerNumeric__energy()
    Rs = [
        local_frame(SWT.sphr2cart(t, p)) for t, p in ((1.0, 0.4), (2.0, -1.0), (0.3, 2))
    ]
    for in_plane, x in (
        (False, np.array([1.3, 0.2, 1.7, -0.3, 1.4, 0.5])),
        (True, np.array([0.7, 2.9, -1.2])),
    ):
        fun, jac, hess = _angle_functions(energy, Rs, in_plane)
        basis = np.eye(len(x))
        num_g = [(fun(x + eps * e) - fun(x - eps * e)) / (2 * eps) for e in basis]
        num_h = [(jac(x + eps * e) - jac(x - eps * e)) / (2 * eps) for e in basis]
        assert np.allclose(jac(x), num_g, rtol=1e-6, atol=1e-12)
        assert np.allclose(hess(x), num_h, rtol=1e-6, atol=1e-12)
    # the in-plane energy is the free energy (up to its constant)
    fun = _angle_functions(energy, None, True)[0]
    x1, x2 = np.array([0.7, 2.9, -1.2]), np.array([0.1, 1.0, 0.4])
    assert np.isclose(
        (fun(x1) - fun(x2)) * energy.scale,
        ml.GetFreeEnergy(x1) - ml.GetFreeEnergy(x2),
    )
    # vectorization of the free energy
    assert ml.GetFreeEnergy(np.zeros((3, 4, 5)), 1.0).shape == (4, 5)


class TestDispersion:
    @pytest.mark.parametrize("phi", [np.pi / 2, 0.0, 0.6])
    def test_single_layer(self, phi):
        """One layer: the uniform mode of `SingleLayer` (n = 0, unpinned)."""
        k = np.linspace(1e5, 40e6, 21)
        sl = SWT.SingleLayer(0.03, PY, 20e-9, kxi=k, phi=phi, boundary_cond=1)
        ml = SWT.MultiLayerNumeric(0.03, PY, [20e-9], kxi=k, phi=phi)
        w, v = ml.GetDispersion()
        assert w.shape == (1, 21) and v.shape == (2, 1, 21)
        assert np.allclose(w[0], sl.GetDispersion(n=0), rtol=1e-6)

    def test_uncoupled_layers(self):
        """Identical layers far apart: degenerate single-layer modes;
        different materials: the individual single-layer modes."""
        k = np.linspace(-20e6, 20e6, 9)
        k[4] = 1e-3
        ml = SWT.MultiLayerNumeric(
            0.1, PY, [10e-9] * 3, kxi=k, phi=0.7, s=1.0, phiInit=0.0
        )
        f = ml.GetDispersion()[0]
        sl = SWT.SingleLayer(0.1, PY, 10e-9, kxi=k, phi=0.7).GetDispersion()
        assert np.allclose(f, sl[None], rtol=1e-6)
        ml = SWT.MultiLayerNumeric(
            0.1, [PY, CO, CFB], 10e-9, kxi=k, phi=0.7, s=1.0, phiInit=0.0
        )
        expected = np.sort(
            [
                SWT.SingleLayer(0.1, m, 10e-9, kxi=k, phi=0.7).GetDispersion()
                for m in (PY, CO, CFB)
            ],
            axis=0,
        )
        assert np.allclose(ml.GetDispersion()[0], expected, rtol=1e-6)

    def test_stack_reversal(self):
        """Rotating the stack by 180 deg about the wavevector reverses the
        order of the layers and the in-plane angles: same frequencies."""
        a = _trilayer(theta=np.pi / 2)
        pa = a.GetPhis()
        b = SWT.MultiLayerNumeric(
            0.04,
            [CO, PY, CFB],
            [5e-9, 4e-9, 6e-9],
            kxi=a.kxi,
            phi=-0.7,
            s=a.s[::-1],
            Jbl=a.Jbl[::-1],
            Jbq=a.Jbq[::-1],
            Ku=a.Ku[::-1],
            phiAnis=-a.phiAnis[::-1],
            Ks=a.Ks[::-1],
            phiInit=-pa[::-1],
        )
        assert np.allclose(b.GetPhis(), -pa[::-1], atol=1e-8)
        wa, va = a.GetDispersion()
        wb, vb = b.GetDispersion()
        assert np.allclose(wa, wb, rtol=1e-10)
        assert np.allclose(np.abs(va[[4, 5, 2, 3, 0, 1]]), np.abs(vb), atol=1e-8)
        # nonreciprocal: the test is not trivial
        assert np.abs(wa[0] - wa[0][::-1]).max() > 0.02 * wa[0].max()

    def test_wavevector_sign_and_2d(self):
        """(kxi, phi) and (-kxi, phi + pi) are the same wavevector; arrays
        of phi give 2D maps."""
        ml = _trilayer()
        w1 = ml.GetDispersion()[0]
        ml.kxi, ml.phi = -ml.kxi, ml.phi + np.pi
        assert np.allclose(w1, ml.GetDispersion()[0], rtol=1e-10)
        ml.kxi, ml.phi = 5e6, np.linspace(0, 2 * np.pi, 7)
        w2 = ml.GetDispersion()[0]
        assert w2.shape == (3, 7) and np.allclose(w2[:, 0], w2[:, -1])

    def test_given_angles(self):
        ml = _trilayer()
        eq = ml.GetEquilibrium()
        assert np.allclose(ml.GetDispersion()[0], ml.GetDispersion(angles=eq)[0])
        assert np.allclose(
            ml.GetDispersion()[0], ml.GetDispersion(angles=eq.ravel())[0]
        )

    def test_thick_layer_from_sublayers(self):
        """A thick layer modelled by sublayers coupled by Jbl = 2*Aex/d_sub
        (no spacers) gives the thickness modes of `SingleLayerNumeric`
        (unpinned)."""
        D, n = 40e-9, 20
        k = np.linspace(1e-3, 30e6, 7)
        ml = SWT.MultiLayerNumeric(
            0.05, PY, [D / n] * n, kxi=k, Jbl=2 * PY.Aex / (D / n), phiInit=0
        )
        f = ml.GetDispersion()[0][:3]
        sln = SWT.SingleLayerNumeric(0.05, PY, D, kxi=k, boundary_cond=1, N=3)
        assert np.allclose(f, sln.GetDispersion()[0], rtol=1e-2)

    @pytest.mark.filterwarnings("ignore:divide by zero")
    def test_derived_quantities(self):
        k = np.linspace(1e-6, 50e6, 30)
        ml = SWT.MultiLayerNumeric(
            0.05, SWT.CoFeB, [10e-9] * 3, kxi=k, s=0.7e-9, Jbl=-0.5e-3
        )
        for q in (ml.GetGroupVelocity(), ml.GetDensityOfStates(), ml.GetLifetime()):
            assert q.shape == (30,) and np.all(np.isfinite(q))
        assert ml.GetGroupVelocity(n=-1).shape == (3, 30)
        assert ml.GetDecLen(n=-1).shape == (3, 30)
        assert np.allclose(ml.GetLifetime(n=-1)[2], ml.GetLifetime(n=2))
        _, bloch = ml.GetBlochFunction(n=1, Nf=50, lifetime=1e-9)
        assert bloch.shape == (50, 30) and np.all(np.isfinite(bloch))
        assert ml.Bext == 0.05  # (restored by GetLifetime)
