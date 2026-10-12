"""Tests of the Fresnel coefficients and Green functions of the layer
stack in `SpinWaveToolkit.bls`."""

import numpy as np
import pytest
import SpinWaveToolkit as SWT

WAVELENGTH = 532e-9
K0 = 2 * np.pi / WAVELENGTH
Q = np.array([0, 0.3 * K0, 0.7 * K0])


def _coefficients(DF, d, source, output, q=Q):
    htp, hts = SWT.bls.fresnel_coefficients(
        WAVELENGTH, DF, np.ones(len(DF)), d, source, output
    )
    return htp(q), hts(q)


class TestFresnel:
    @pytest.mark.parametrize("n", [1.0, 1.5])
    def test_homogeneous_stack(self, n):
        """Without interfaces, the light only propagates through the
        source layer (phase exp(i kz d)), nothing is reflected."""
        d = 30e-9
        tp, ts = _coefficients([n**2] * 3, [d], 1, 0)
        assert tp.shape == ts.shape == (2, len(Q))
        kz = np.sqrt((n * K0) ** 2 - Q**2)
        for t in (tp, ts):
            assert np.allclose(t[0], np.exp(1j * kz * d))
            assert np.allclose(t[1], 0)

    @pytest.mark.parametrize(
        "source, output, nonzero",
        [
            (1, 0, [1, 0]),
            (1, 3, [0, 1]),
            (1, 1, [[0, 0], [0, 0]]),
            (1, 2, [[0, 0], [0, 1]]),
            (2, 1, [[1, 0], [0, 0]]),
            (2, 0, [1, 0]),
        ],
    )
    def test_homogeneous_output_layers(self, source, output, nonzero):
        """All branches of the output layer: in a homogeneous stack, the
        coefficients are pure propagation phases or zero."""
        tp, ts = _coefficients([2.25] * 4, [30e-9, 50e-9], source, output)
        nonzero = np.array(nonzero, dtype=bool)
        assert tp.shape == (*nonzero.shape, len(Q))
        for t in (tp, ts):
            assert np.allclose(np.abs(t[nonzero]), 1)
            assert np.allclose(t[~nonzero], 0)

    def test_single_interface_normal_incidence(self):
        """Source layer with n = 1.5 on a substrate with the same index:
        transmission to air is 2n/(n+1) times the propagation phase."""
        n, d = 1.5, 30e-9
        tp, ts = _coefficients([1, n**2, n**2], [d], 1, 0, q=0.0)
        t_expected = 2 * n / (n + 1) * np.exp(1j * n * K0 * d)
        assert np.isclose(tp[0], t_expected) and np.isclose(ts[0], t_expected)
        assert np.isclose(tp[1], 0) and np.isclose(ts[1], 0)

    @pytest.mark.parametrize("source, output", [(1, 0), (1, 3), (1, 1), (1, 2), (2, 1)])
    def test_normal_incidence_degeneracy(self, source, output):
        """At normal incidence, p and s coefficients have the same
        magnitude (they differ in sign conventions of the reflection)."""
        tp, ts = _coefficients(
            [1, 2.25, 4 + 0.5j, 2.25], [30e-9, 50e-9], source, output, q=0.0
        )
        assert np.allclose(np.abs(tp), np.abs(ts))
        assert np.all(np.abs(tp) > 0)

    def test_infinite_layer_scalar_q(self):
        tp, ts = _coefficients([1, 2.25, 2.25], [np.inf], 1, 0, q=0.0)
        assert np.allclose(tp, [1.2, 0]) and np.allclose(ts, [1.2, 0])

    @pytest.mark.xfail(
        reason="fresnel_coefficients sets rpij[i] = 0 (scalar) for d = inf, which fails "
        + "for array q (as used by the GF signal functions)",
        raises=ValueError,
        strict=True,
    )
    def test_infinite_layer_array_q(self):
        tp, ts = _coefficients([1, 2.25, 2.25], [np.inf], 1, 0)
        assert tp.shape == (2, len(Q))


class TestGreenFunction:
    def _green(self, tp=None, ts=None):
        k = np.linspace(-0.9 * K0, 0.9 * K0, 7)
        KX, KY = np.meshgrid(k, k, indexing="ij")
        DF = -8.1653 + 15.348j
        ones = np.ones_like(KX)
        tp = [ones, 0.5 * ones] if tp is None else tp
        ts = [ones, 0.5 * ones] if ts is None else ts
        pGF, sGF = SWT.bls.sph_green_function(KX, KY, DF, WAVELENGTH, tp, ts)
        return KX, KY, pGF, sGF

    def test_shapes(self):
        KX, _, pGF, sGF = self._green()
        assert len(pGF) == len(sGF) == 3
        for G in (pGF, sGF):
            for row in G:
                assert len(row) == 2
                for g in row:
                    assert np.shape(g) == KX.shape

    def test_polarization_directions(self):
        """The s-polarized field is perpendicular to the in-plane
        wavevector and has no z component; the in-plane p-polarized
        field is parallel to it."""
        KX, KY, pGF, sGF = self._green()
        K = np.hypot(KX, KY)
        nz = K > 0
        c, s = KX[nz] / K[nz], KY[nz] / K[nz]
        for j in range(2):
            scale = max(np.abs(g[j]).max() for g in pGF + sGF)
            assert np.abs(sGF[0][j][nz] * c + sGF[1][j][nz] * s).max() < 1e-12 * scale
            assert np.all(sGF[2][j] == 0)
            assert np.abs(pGF[0][j][nz] * s - pGF[1][j][nz] * c).max() < 1e-12 * scale

    def test_normal_incidence(self):
        """At k = 0, p and s polarizations are degenerate (x and y)."""
        _, _, pGF, sGF = self._green()
        c = 3  # center of the 7x7 grid
        for j in range(2):
            assert np.isclose(abs(pGF[0][j][c, c]), abs(sGF[1][j][c, c]), rtol=1e-12)
            assert pGF[2][j][c, c] == 0 and sGF[0][j][c, c] == 0
