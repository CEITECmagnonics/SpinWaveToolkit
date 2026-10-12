"""Tests of the magneto-optic susceptibilities in
`SpinWaveToolkit.bls.susceptibilities`."""

import numpy as np
import pytest
from SpinWaveToolkit.bls import susceptibilities as sus

AXES = {"x": 0, "y": 1, "z": 2}


@pytest.fixture
def m():
    """Random complex dynamic magnetization with shape (3, Nf, Nkx, Nky)."""
    rng = np.random.default_rng(1)
    return rng.normal(size=(3, 2, 4, 5)) + 1j * rng.normal(size=(3, 2, 4, 5))


def _perturbed(axis, dm):
    """Static magnetization along `axis` plus a small transverse `dm`."""
    i = AXES[axis]
    mtot = dm.copy()
    mtot[i] = 1  # dynamic component along the static magnetization is 0 (1st order)
    return mtot


class TestMOLinear:
    def test_shape_and_antisymmetry(self, m):
        chi = sus.mo_linear(m, Q=0.3 + 0.1j)
        assert chi.shape == (3, 3, *m.shape[1:])
        assert np.allclose(chi, -chi.transpose(1, 0, 2, 3, 4))

    def test_components(self, m):
        Q = 2.0
        chi = sus.mo_linear(m, Q=Q)
        assert np.allclose(chi[0, 1], 1j * Q * m[2])
        assert np.allclose(chi[1, 2], 1j * Q * m[0])
        assert np.allclose(chi[2, 0], 1j * Q * m[1])

    def test_linear_in_m_and_Q(self, m):
        assert np.allclose(sus.mo_linear(2 * m, Q=3), 6 * sus.mo_linear(m))


class TestMOQuadratic:
    def test_full_tensor(self, m):
        mx, my, mz = m
        Bii, Bij = [1, 2, 3], [4, 5, 6]
        chi = sus.mo_quadratic(m, Bii=Bii, Bij=Bij)
        assert chi.shape == (3, 3, *m.shape[1:])
        assert np.allclose(chi, chi.transpose(1, 0, 2, 3, 4))  # symmetric
        assert np.allclose(chi[0, 0], 1 * mx**2)
        assert np.allclose(chi[1, 1], 2 * my**2)
        assert np.allclose(chi[2, 2], 3 * mz**2)
        assert np.allclose(chi[1, 2], 4 * 2 * my * mz)
        assert np.allclose(chi[0, 2], 5 * 2 * mx * mz)
        assert np.allclose(chi[0, 1], 6 * 2 * mx * my)

    def test_scalar_constants(self, m):
        assert np.allclose(
            sus.mo_quadratic(m, Bii=2, Bij=3),
            sus.mo_quadratic(m, Bii=[2] * 3, Bij=[3] * 3),
        )

    @pytest.mark.parametrize("axis", ["x", "y", "z"])
    def test_linearization(self, m, axis):
        """The linearized tensor is the first-order change of the full
        tensor for a small transverse deviation from the static
        magnetization."""
        eps = 1e-6
        dm = eps * m
        dm[AXES[axis]] = 0
        Bii, Bij = [1, 2, 3], [4, 5, 6]
        full = sus.mo_quadratic(_perturbed(axis, dm), Bii, Bij)
        static = sus.mo_quadratic(_perturbed(axis, 0 * dm), Bii, Bij)
        lin = sus.mo_quadratic(dm, Bii, Bij, linearize_along=axis)
        assert np.allclose(full - static, lin, rtol=0, atol=1e-10)
        assert np.allclose(np.diagonal(lin), 0)

    def test_invalid_inputs(self, m):
        with pytest.raises(ValueError):
            sus.mo_quadratic(m, linearize_along="w")
        with pytest.raises(ValueError):
            sus.mo_quadratic(m, Bii=[1, 2])
        with pytest.raises(ValueError):
            sus.mo_quadratic(m, Bij=[1, 2])


class TestMOQuadraticYIG:
    def test_symmetric(self, m):
        chi = sus.mo_quadratic_yig111(m, 1.0, 0.3, 0.2)
        assert chi.shape == (3, 3, *m.shape[1:])
        assert np.allclose(chi, chi.transpose(1, 0, 2, 3, 4))

    def test_isotropic_limit(self, m):
        """For g11 - g12 - 2*g44 = 0 and g12 = 0, the tensor equals
        `mo_quadratic` with Bii = g11 and Bij = g44."""
        g11, g44 = 1.4, 0.7
        assert np.allclose(
            sus.mo_quadratic_yig111(m, g11, 0, g44),
            sus.mo_quadratic(m, Bii=g11, Bij=g44),
        )
        for axis in AXES:
            assert np.allclose(
                sus.mo_quadratic_yig111(m, g11, 0, g44, linearize_along=axis),
                sus.mo_quadratic(m, Bii=g11, Bij=g44, linearize_along=axis),
            )

    @pytest.mark.parametrize("axis", ["x", "y", "z"])
    def test_linearization(self, m, axis):
        eps = 1e-6
        dm = eps * m
        dm[AXES[axis]] = 0
        g = (1.0, 0.3, 0.2)
        full = sus.mo_quadratic_yig111(_perturbed(axis, dm), *g)
        static = sus.mo_quadratic_yig111(_perturbed(axis, 0 * dm), *g)
        lin = sus.mo_quadratic_yig111(dm, *g, linearize_along=axis)
        assert np.allclose(full - static, lin, rtol=0, atol=1e-10)

    def test_invalid_axis(self, m):
        with pytest.raises(ValueError):
            sus.mo_quadratic_yig111(m, 1, 0, 0, linearize_along="w")
