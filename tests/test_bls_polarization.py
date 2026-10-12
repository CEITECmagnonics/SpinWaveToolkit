"""Tests of the Jones calculus in `SpinWaveToolkit.bls.polarization`."""

import numpy as np
import pytest
from SpinWaveToolkit.bls import polarization as pol


def _same_state(a, b):
    """True if Jones vectors/fields `a` and `b` (normalized) are equal up
    to a global phase (at each point)."""
    overlap = np.abs(np.sum(np.conj(a) * b, axis=0))
    return np.allclose(overlap, 1)


@pytest.fixture
def grid():
    x = np.linspace(-1, 1, 7)
    return np.meshgrid(x, x, indexing="ij")


class TestJonesVector:
    @pytest.mark.parametrize("pol_type", ["linear", "rcp", "lcp", "elliptical"])
    def test_uniform_normalized(self, pol_type):
        e = pol.jones_vector(pol_type, angle=30, axis_ratio=0.4)
        assert e.shape == (2,)
        assert np.isclose(np.linalg.norm(e), 1)

    def test_linear(self):
        assert np.allclose(pol.jones_vector("linear", 0), [1, 0])
        assert np.allclose(pol.jones_vector("linear", 90), [0, 1])
        assert np.allclose(pol.jones_vector("linear", 45), [1, 1] / np.sqrt(2))

    def test_circular(self):
        rcp, lcp = pol.jones_vector("rcp"), pol.jones_vector("lcp")
        assert np.allclose(rcp, [1, -1j] / np.sqrt(2))
        assert np.allclose(lcp, [1, 1j] / np.sqrt(2))
        assert np.isclose(np.vdot(rcp, lcp), 0)  # orthogonal

    def test_elliptical_limits(self):
        """Axis ratio 0 is linear, +1 is lcp and -1 is rcp."""
        for angle in (0, 25, 90):
            assert _same_state(
                pol.jones_vector("elliptical", angle, axis_ratio=0),
                pol.jones_vector("linear", angle),
            )
            assert _same_state(
                pol.jones_vector("elliptical", angle, axis_ratio=1),
                pol.jones_vector("lcp"),
            )
            assert _same_state(
                pol.jones_vector("elliptical", angle, axis_ratio=-1),
                pol.jones_vector("rcp"),
            )

    def test_elliptical_axes(self):
        """Major axis along `angle` with the given axis ratio."""
        e = pol.jones_vector("elliptical", 30, axis_ratio=0.5)
        t = np.linspace(0, 2 * np.pi, 3601)
        trace = np.real(np.outer(e, np.exp(-1j * t)))  # field over one period
        r = np.hypot(*trace)
        major = trace[:, np.argmax(r)]
        assert np.isclose(r.min() / r.max(), 0.5, atol=1e-4)
        assert np.isclose(
            abs(np.dot(major / r.max(), [np.cos(np.pi / 6), np.sin(np.pi / 6)])), 1
        )

    def test_radial_azimuthal(self, grid):
        X, Y = grid
        rad = pol.jones_vector("radial", X=X, Y=Y)
        azm = pol.jones_vector("azimuthal", X=X, Y=Y)
        assert rad.shape == azm.shape == (2, *X.shape)
        assert np.allclose(np.sum(np.abs(rad) ** 2, axis=0), 1)
        assert np.allclose(np.sum(np.conj(rad) * azm, axis=0), 0)  # orthogonal
        # radial is parallel to the position vector (except at the origin)
        assert np.allclose(rad[0] * Y - rad[1] * X, 0)
        assert np.allclose(rad[:, 3, 3], [1, 0])  # phi = 0 at the origin

    @pytest.mark.parametrize("pol_type", ["radial", "azimuthal"])
    def test_spatial_requires_grid(self, pol_type):
        with pytest.raises(ValueError):
            pol.jones_vector(pol_type)

    def test_invalid_type(self):
        with pytest.raises(ValueError):
            pol.jones_vector("diagonal")


class TestJonesMatrices:
    def test_apply_broadcasting(self, grid):
        """Uniform and spatially varying matrices and vectors can be mixed."""
        X, Y = grid
        e = pol.jones_vector("linear", 20)
        efield = pol.jones_vector("radial", X=X, Y=Y)
        M = pol.half_wave_plate(10)
        Mfield = pol.q_plate(X, Y)
        assert pol.apply_jones_matrix(M, e).shape == (2,)
        assert pol.apply_jones_matrix(M, efield).shape == (2, *X.shape)
        assert pol.apply_jones_matrix(Mfield, e).shape == (2, *X.shape)
        assert pol.apply_jones_matrix(Mfield, efield).shape == (2, *X.shape)
        assert np.allclose(
            pol.apply_jones_matrix(M, efield)[:, 2, 5], M @ efield[:, 2, 5]
        )
        assert np.allclose(
            pol.apply_jones_matrix(Mfield, e)[:, 2, 5], Mfield[:, :, 2, 5] @ e
        )

    @pytest.mark.parametrize("angle", [0, 30, 90, 135])
    def test_linear_polarizer(self, angle):
        P = pol.linear_polarizer(angle)
        assert np.allclose(P @ P, P)  # projector
        assert np.allclose(P, P.conj().T)
        e = pol.jones_vector("linear", angle)
        assert np.allclose(P @ e, e)
        assert np.allclose(P @ pol.jones_vector("linear", angle + 90), 0)
        # Malus's law
        assert np.isclose(
            np.linalg.norm(P @ pol.jones_vector("linear", angle + 60)) ** 2, 0.25
        )

    @pytest.mark.parametrize("retardance", [0.3, np.pi / 2, np.pi])
    def test_retarder_unitary(self, retardance):
        M = pol.retarder(retardance, 17)
        assert np.allclose(M.conj().T @ M, np.eye(2))
        # the field along the fast axis keeps its phase, along the slow axis is delayed
        fast, slow = pol.jones_vector("linear", 17), pol.jones_vector("linear", 107)
        assert np.allclose(M @ fast, fast)
        assert np.allclose(M @ slow, np.exp(1j * retardance) * slow)

    def test_half_wave_plate_rotates(self):
        """HWP at angle a mirrors linear polarization about a."""
        e = pol.apply_jones_matrix(
            pol.half_wave_plate(22.5), pol.jones_vector("linear", 0)
        )
        assert _same_state(e, pol.jones_vector("linear", 45))

    def test_quarter_wave_plate(self):
        """Documented convention: x -> rcp for 45 deg, x -> lcp for -45 deg."""
        x = pol.jones_vector("linear", 0)
        assert _same_state(pol.quarter_wave_plate(45) @ x, pol.jones_vector("rcp"))
        assert _same_state(pol.quarter_wave_plate(-45) @ x, pol.jones_vector("lcp"))

    def test_spiral_phase_plate(self, grid):
        X, Y = grid
        M = pol.spiral_phase_plate(X, Y, charge=2, angle=10)
        assert M.shape == (2, 2, *X.shape)
        phase = np.exp(2j * (np.arctan2(Y, X) - np.deg2rad(10)))
        assert np.allclose(M[0, 0], phase) and np.allclose(M[1, 1], phase)
        assert np.allclose(M[0, 1], 0) and np.allclose(M[1, 0], 0)

    def test_q_plate(self, grid):
        """Default q-plate: x -> radial, y -> azimuthal (up to a global phase)."""
        X, Y = grid
        M = pol.q_plate(X, Y)
        assert M.shape == (2, 2, *X.shape)
        rad = pol.apply_jones_matrix(M, pol.jones_vector("linear", 0))
        azm = pol.apply_jones_matrix(M, pol.jones_vector("linear", 90))
        assert _same_state(rad, pol.jones_vector("radial", X=X, Y=Y))
        assert _same_state(azm, pol.jones_vector("azimuthal", X=X, Y=Y))
        # unitary at each point
        MH_M = np.einsum("ji...,jk...->ik...", M.conj(), M)
        assert np.allclose(MH_M, np.eye(2)[:, :, None, None])
