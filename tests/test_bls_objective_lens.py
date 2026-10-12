"""Tests of the focal and pupil fields of `SpinWaveToolkit.bls.ObjectiveLens`."""

import numpy as np
import pytest
import SpinWaveToolkit as SWT
from SpinWaveToolkit.bls import polarization as pol

WAVELENGTH, NA, F0, FOCAL = 532e-9, 0.75, 1.0, 1e-3
K0 = 2 * np.pi / WAVELENGTH


@pytest.fixture(scope="module")
def lens():
    return SWT.bls.ObjectiveLens(WAVELENGTH, NA, F0, FOCAL)


@pytest.fixture(scope="module")
def pupil_grid():
    kx = np.linspace(-1.2 * K0 * NA, 1.2 * K0 * NA, 61)
    return np.meshgrid(kx, kx, indexing="ij")


def _synthesize(Ek, kx, x):
    """Focal field E(r) = int E_k(k) exp(i k.r) d^2k on the grid (x, x),
    evaluated as a sum over a fine k-grid."""
    F = np.exp(1j * np.outer(x, kx))
    return [F @ c @ F.T * (kx[1] - kx[0]) ** 2 for c in Ek]


class TestFocalField:
    def test_shapes_and_grid(self, lens):
        x, y, Ex, Ey, Ez = lens.getFocalField(0, 1e-6, 21)
        assert np.allclose(x, np.linspace(-1e-6, 1e-6, 21)) and np.allclose(y, x)
        assert Ex.shape == Ey.shape == Ez.shape == (21, 21)

    def test_symmetry(self, lens):
        """x-polarized input: Ex, Ey even/odd in x and y, Ez odd in x."""
        _, _, Ex, Ey, Ez = lens.getFocalField(0.2e-6, 1e-6, 21)
        assert np.allclose(Ex, Ex[::-1, :]) and np.allclose(Ex, Ex[:, ::-1])
        assert np.allclose(Ey, -Ey[::-1, :]) and np.allclose(Ey, -Ey[:, ::-1])
        assert np.allclose(Ez, -Ez[::-1, :]) and np.allclose(Ez, Ez[:, ::-1])
        assert abs(Ey[10, 10]) < 1e-12 * abs(Ex[10, 10])
        assert abs(Ez[10, 10]) < 1e-12 * abs(Ex[10, 10])

    def test_matches_angular_spectrum(self, lens):
        """The focal field equals the angular spectrum representation
        of the pupil field (Novotny & Hecht, eq. (3.47)), also out of
        focus (the error is dominated by the discretization of the
        pupil)."""
        x = np.linspace(-1.5e-6, 1.5e-6, 15)
        kx = np.linspace(-1.01 * K0 * NA, 1.01 * K0 * NA, 401)
        KX, KY = np.meshgrid(kx, kx, indexing="ij")
        for z in (0, 0.5e-6):
            _, _, *Ef = lens.getFocalField(z, x[-1], len(x), rtol=1e-6)
            Es = _synthesize(lens.getPupilField(z, KX, KY), kx, x)
            scale = np.abs(Ef[0]).max()
            for c in range(3):
                assert np.abs(Ef[c] - Es[c]).max() < 3e-4 * scale

    def test_rtol(self, lens):
        """The error of the field is below `rtol` (relative to its maximum)."""
        _, _, *Eref = lens.getFocalField(0.3e-6, 3e-6, 101, rtol=1e-9)
        scale = max(np.abs(c).max() for c in Eref)
        for rtol in (1e-2, 1e-4):
            _, _, *E = lens.getFocalField(0.3e-6, 3e-6, 101, rtol=rtol)
            for c in range(3):
                assert np.abs(E[c] - Eref[c]).max() < rtol * scale

    @pytest.mark.parametrize("rtol", [0, 1e-11, 1, 2])
    def test_invalid_rtol(self, lens, rtol):
        with pytest.raises(ValueError):
            lens.getFocalField(0, 1e-6, 11, rtol=rtol)

    def test_single_point(self, lens):
        """For N = 1, the grid is the single point (-rho_max, -rho_max)."""
        x, _, Ex, Ey, Ez = lens.getFocalField(0, 1e-6, 1)
        assert np.allclose(x, [-1e-6]) and Ex.shape == (1, 1)
        _, _, *E3 = lens.getFocalField(0, 1e-6, 3)
        scale = np.abs(E3[0]).max()
        for c1, c3 in zip((Ex, Ey, Ez), E3):
            assert abs(c1[0, 0] - c3[0, 0]) < 1e-4 * scale

    def test_radial(self, lens):
        x, y, Ex, Ey, Ez = lens.getFocalFieldRad(0, 1e-6, 21)
        X, Y = np.meshgrid(x, y, indexing="ij")
        scale = np.abs(Ez).max()
        assert np.allclose(
            Ex * Y - Ey * X, 0, atol=1e-12 * scale
        )  # radial in-plane field
        assert np.argmax(np.abs(Ez)) == np.ravel_multi_index((10, 10), Ez.shape)
        assert abs(Ex[10, 10]) < 1e-12 * scale

    def test_azimuthal(self, lens):
        x, y, Ex, Ey, Ez = lens.getFocalFieldAzm(0, 1e-6, 21)
        X, Y = np.meshgrid(x, y, indexing="ij")
        scale = np.abs(Ex).max()
        assert np.all(Ez == 0)
        assert np.allclose(Ex * X + Ey * Y, 0, atol=1e-12 * scale)  # azimuthal field
        assert abs(Ey[10, 10]) < 1e-12 * scale

    def test_doughnut_beams_match_pupil_field(self, lens):
        """Radial and azimuthal doughnut beams equal the pupil fields
        weighted by 2*rho/w0 as described in `getPupilField`."""
        x = np.linspace(-1e-6, 1e-6, 11)
        kx = np.linspace(-1.01 * K0 * NA, 1.01 * K0 * NA, 401)
        KX, KY = np.meshgrid(kx, kx, indexing="ij")
        sin_theta = np.hypot(KX, KY) / K0
        for pol_type, method in (
            ("radial", lens.getFocalFieldRad),
            ("azimuthal", lens.getFocalFieldAzm),
        ):
            e_in = 2 * sin_theta / (F0 * NA) * pol.jones_vector(pol_type, X=KX, Y=KY)
            Es = _synthesize(lens.getPupilField(0, KX, KY, pol_type=e_in), kx, x)
            _, _, *Ef = method(0, x[-1], len(x), rtol=1e-6)
            scale = max(np.abs(c).max() for c in Ef)
            for c in range(3):
                assert np.abs(Ef[c] - Es[c]).max() < 1e-3 * scale


class TestPupilField:
    def test_aperture(self, lens, pupil_grid):
        KX, KY = pupil_grid
        Ek = lens.getPupilField(0, KX, KY)
        # (points exactly on the edge of the pupil are skipped, since their
        # assignment depends on round-off, which differs between platforms)
        rho = np.hypot(KX, KY) / (K0 * NA)
        outside, inside = rho > 1 + 1e-9, rho < 1 - 1e-9
        assert outside.any() and np.any(np.abs(rho - 1) < 1e-9)
        for c in Ek:
            assert c.shape == KX.shape
            assert np.all(c[outside] == 0)
        assert np.all(Ek[0][inside] != 0)

    @pytest.mark.parametrize(
        "pol_type", ["linear", "rcp", "radial", "azimuthal", "elliptical"]
    )
    def test_transversality(self, lens, pupil_grid, pol_type):
        """Each plane wave of the angular spectrum is transverse, k.E = 0."""
        KX, KY = pupil_grid
        n = 1.5
        Ek = lens.getPupilField(
            0.1e-6, KX, KY, pol_type=pol_type, pol_angle=30, axis_ratio=0.3, n=n
        )
        KZ = np.sqrt((n * K0) ** 2 - KX**2 - KY**2 + 0j)
        kE = KX * Ek[0] + KY * Ek[1] + KZ * Ek[2]
        assert np.abs(kE).max() < 1e-12 * K0 * np.abs(Ek[0]).max()

    def test_defocus_phase(self, lens, pupil_grid):
        KX, KY = pupil_grid
        z = 0.4e-6
        E0, Ez = lens.getPupilField(0, KX, KY)[0], lens.getPupilField(z, KX, KY)[0]
        KZ = np.sqrt(np.maximum(K0**2 - KX**2 - KY**2, 0))
        assert np.allclose(Ez, E0 * np.exp(1j * KZ * z))

    def test_jones_input(self, lens, pupil_grid):
        """Strings, Jones vectors and Jones fields give the same field."""
        KX, KY = pupil_grid
        ref = lens.getPupilField(0, KX, KY, pol_type="linear", pol_angle=30)
        vec = lens.getPupilField(0, KX, KY, pol_type=pol.jones_vector("linear", 30))
        field = np.broadcast_to(
            pol.jones_vector("linear", 30)[:, None, None], (2, *KX.shape)
        )
        fld = lens.getPupilField(0, KX, KY, pol_type=field)
        assert np.allclose(ref, vec) and np.allclose(ref, fld)
        # q-plate after x-polarized light gives radial polarization
        e_q = pol.apply_jones_matrix(pol.q_plate(KX, KY), pol.jones_vector("linear", 0))
        assert np.allclose(
            lens.getPupilField(0, KX, KY, pol_type=e_q),
            lens.getPupilField(0, KX, KY, pol_type="radial"),
        )

    def test_linearity_in_polarization(self, lens, pupil_grid):
        KX, KY = pupil_grid
        Ex = np.array(lens.getPupilField(0, KX, KY, pol_type=[1, 0]))
        Ey = np.array(lens.getPupilField(0, KX, KY, pol_type=[0, 1]))
        Ercp = np.array(lens.getPupilField(0, KX, KY, pol_type="rcp"))
        assert np.allclose(Ercp, (Ex - 1j * Ey) / np.sqrt(2))

    def test_invalid_inputs(self, lens, pupil_grid):
        KX, KY = pupil_grid
        with pytest.raises(ValueError):
            lens.getPupilField(0, KX, KY, n=0.5)  # NA > n
        with pytest.raises(ValueError):
            lens.getPupilField(0, KX, KY, pol_type=np.ones(3))
        with pytest.raises(ValueError):
            lens.getPupilField(0, KX, KY, pol_type=np.ones((2, 3, 3)))
        with pytest.raises(TypeError):
            lens.getPupilField(0, KX, KY, polarisation="linear")

    def test_deprecated_names(self, lens, pupil_grid):
        KX, KY = pupil_grid
        ref = lens.getPupilField(0, KX, KY, pol_type="linear", pol_angle=60)
        with pytest.warns(DeprecationWarning):
            old = lens.getPupilField(
                0, KX, KY, polarization_type="linear", polarization_angle_deg=60
            )
        assert np.allclose(ref, old)


def test_rotate_field(lens):
    """Rotating the x-polarized focal field by 90 deg gives the
    y-polarized one (exact on a square grid)."""
    KX, KY = np.meshgrid(
        *[np.linspace(-1.1 * K0 * NA, 1.1 * K0 * NA, 31)] * 2, indexing="ij"
    )
    x = np.linspace(-1e-6, 1e-6, 21)
    Ex_pol = _synthesize(lens.getPupilField(0, KX, KY, pol_angle=0), KX[:, 0], x)
    Ey_pol = _synthesize(lens.getPupilField(0, KX, KY, pol_angle=90), KX[:, 0], x)
    rot = SWT.rotate_field(Ex_pol, x, x, 90)
    rot0 = SWT.rotate_field(Ex_pol, x, x, 0)
    for c in range(3):
        assert np.allclose(rot[c], Ey_pol[c], atol=1e-12 * np.abs(Ex_pol[0]).max())
        assert np.allclose(rot0[c], Ex_pol[c])
