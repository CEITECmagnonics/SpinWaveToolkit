"""Tests of the BLS signal functions (reciprocity theorem and Green
functions) in `SpinWaveToolkit.bls`.

The focal fields of the `bls_setup` fixture are synthesized exactly from
the pupil fields on the conjugate real-space grid, so the focal and
pupil versions of the functions must agree to round-off.
"""

import warnings
import numpy as np
import pytest
import SpinWaveToolkit as SWT
from SpinWaveToolkit.bls import polarization as pol

pytestmark = pytest.mark.filterwarnings("ignore:`get_signal_.*experimental:UserWarning")


def _rel(a, b):
    return np.abs(np.asarray(a) - np.asarray(b)).max() / np.abs(b).max()


# ----------------------------------------------------------------------
# reciprocity theorem
# ----------------------------------------------------------------------
class TestRT:
    @pytest.mark.parametrize("coherent", [False, True])
    def test_pupil_equals_focal(self, bls_setup, coherent):
        s = bls_setup
        sp, qp = SWT.bls.get_signal_RT_pupil(
            [s.kx, s.kx], s.Ei_k, s.Ej_k, s.chi, coherent_exc=coherent
        )
        sf, qf = SWT.bls.get_signal_RT_focal(
            [s.x, s.x], s.Ei_r, s.Ej_r, [s.kx, s.kx], s.chi, coherent_exc=coherent
        )
        assert sp.shape == (3,) and qp.shape == (3, 3, len(s.kx), len(s.kx))
        assert np.all(sp > 0)
        assert _rel(qp, qf) < 1e-12
        assert _rel(sp, sf) < 1e-12

    @pytest.mark.parametrize("coherent", [False, True])
    def test_focal_3d_single_depth(self, bls_setup, coherent):
        s = bls_setup
        sf, qf = SWT.bls.get_signal_RT_focal(
            [s.x, s.x], s.Ei_r, s.Ej_r, [s.kx, s.kx], s.chi, coherent_exc=coherent
        )
        with pytest.warns(UserWarning, match="experimental"):
            s3, q3 = SWT.bls.get_signal_RT_focal_3d(
                [s.x, s.x], s.Ei_r, s.Ej_r, [s.kx, s.kx], s.chi, coherent_exc=coherent
            )
        assert q3.shape == (*qf.shape, 1)
        assert _rel(q3[..., 0], qf) < 1e-12
        assert _rel(s3, sf) < 1e-12

    def test_focal_3d_depths(self, bls_setup):
        """Two identical depths: the transfer function is the same for
        both; the amplitudes add up, i.e. the signal is 4 times larger."""
        s = bls_setup
        Ei = [np.stack([c, c], axis=-1) for c in s.Ei_r]
        Ej = [np.stack([c, c], axis=-1) for c in s.Ej_r]
        chi2 = np.stack([s.chi, s.chi])
        s1, _ = SWT.bls.get_signal_RT_focal_3d(
            [s.x, s.x], s.Ei_r, s.Ej_r, [s.kx, s.kx], s.chi
        )
        s2, q2 = SWT.bls.get_signal_RT_focal_3d([s.x, s.x], Ei, Ej, [s.kx, s.kx], chi2)
        assert np.allclose(q2[..., 0], q2[..., 1])
        assert _rel(s2, 4 * s1) < 1e-12
        with pytest.raises(ValueError):
            SWT.bls.get_signal_RT_focal_3d(
                [s.x, s.x], Ei, Ej, [s.kx, s.kx], s.chi[None]
            )

    def test_conv_methods(self, bls_setup):
        s = bls_setup
        kxy = [s.kx, s.kx]
        q_fft = SWT.bls.get_transfer_function_RT_pupil(kxy, s.Ei_k, s.Ej_k)
        q_dir = SWT.bls.get_transfer_function_RT_pupil(
            kxy, s.Ei_k, s.Ej_k, conv_method="direct"
        )
        assert _rel(q_fft, q_dir) < 1e-12
        with pytest.raises(ValueError):
            SWT.bls.get_transfer_function_RT_pupil(
                kxy, s.Ei_k, s.Ej_k, conv_method="fast"
            )

    def test_transfer_function_reuse_and_mask(self, bls_setup):
        """The signal is assembled from the transfer function, which is
        calculated only for nonzero components of chi."""
        s = bls_setup
        kxy = [s.kx, s.kx]
        q = SWT.bls.get_transfer_function_RT_pupil(kxy, s.Ei_k, s.Ej_k)
        sig, q_sig = SWT.bls.get_signal_RT_pupil(kxy, s.Ei_k, s.Ej_k, s.chi)
        mask = np.any(s.chi, axis=(2, 3, 4))
        assert not mask.all()  # (the test needs some zero components)
        assert np.allclose(q_sig[mask], q[mask]) and np.all(q_sig[~mask] == 0)
        dK = (s.kx[1] - s.kx[0]) ** 2
        tmp = np.einsum("uvxy,uvfxy->fxy", q, s.chi)
        assert _rel(sig, np.sum(np.abs(tmp) ** 2, axis=(1, 2)) * dK) < 1e-12
        q_m = SWT.bls.get_transfer_function_RT_pupil(kxy, s.Ei_k, s.Ej_k, mask=mask)
        assert np.all(q_m[~mask] == 0) and np.allclose(q_m[mask], q[mask])

    def test_transfer_function_at_zero(self, bls_setup):
        """qm(q=0) is the overlap integral of the fields, int Ej Ei d^2r."""
        s = bls_setup
        q = SWT.bls.get_transfer_function_RT_pupil([s.kx, s.kx], s.Ei_k, s.Ei_k)
        dS = (s.x[1] - s.x[0]) ** 2
        c = len(s.kx) // 2
        for u in range(3):
            for v in range(3):
                overlap = np.sum(s.Ei_r[u] * s.Ei_r[v]) * dS
                assert abs(q[u, v, c, c] - overlap) < 1e-12 * np.abs(q).max()

    def test_symmetries(self, bls_setup):
        """Swapping Ei and Ej with transposed chi gives the same signal;
        the linear MO signal vanishes for Ei = Ej (symmetric transfer
        function, antisymmetric chi)."""
        s = bls_setup
        kxy = [s.kx, s.kx]
        sig, _ = SWT.bls.get_signal_RT_pupil(kxy, s.Ei_k, s.Ej_k, s.chi)
        swp, _ = SWT.bls.get_signal_RT_pupil(
            kxy, s.Ej_k, s.Ei_k, s.chi.transpose(1, 0, 2, 3, 4)
        )
        assert _rel(swp, sig) < 1e-12
        chi_lin = SWT.bls.susceptibilities.mo_linear(s.m)
        lin, _ = SWT.bls.get_signal_RT_pupil(kxy, s.Ej_k, s.Ei_k, chi_lin)
        par, _ = SWT.bls.get_signal_RT_pupil(kxy, s.Ei_k, s.Ei_k, chi_lin)
        assert par.max() < 1e-20 * lin.max()

    def test_scaling(self, bls_setup):
        """The signal is quadratic in chi and zero for zero chi."""
        s = bls_setup
        kxy = [s.kx, s.kx]
        for coherent in (False, True):
            s1, _ = SWT.bls.get_signal_RT_pupil(kxy, s.Ei_k, s.Ej_k, s.chi, coherent)
            s2, _ = SWT.bls.get_signal_RT_pupil(
                kxy, s.Ei_k, s.Ej_k, 2j * s.chi, coherent
            )
            s0, _ = SWT.bls.get_signal_RT_pupil(
                kxy, s.Ei_k, s.Ej_k, 0 * s.chi, coherent
            )
            assert _rel(s2, 4 * s1) < 1e-12
            assert np.all(s0 == 0)
            # bloch is proportional to [1, 0.5, 0.2] for the 3 frequencies
            assert np.allclose(s1 / s1[0], [1, 0.25, 0.04])

    def test_grid_checks(self, bls_setup):
        s = bls_setup
        kx = s.kx
        with pytest.raises(ValueError, match="symmetric"):
            SWT.bls.get_transfer_function_RT_pupil([kx + 1e5, kx], s.Ei_k, s.Ej_k)
        k_uneq = kx.copy()
        k_uneq[5] += 0.3 * (kx[1] - kx[0])
        with pytest.raises(ValueError, match="equidistant"):
            SWT.bls.get_transfer_function_RT_pupil([kx, k_uneq], s.Ei_k, s.Ej_k)
        k_even = np.linspace(kx[0], kx[-1], 40)
        KX, KY = np.meshgrid(k_even, k_even, indexing="ij")
        Ek = s.lens.getPupilField(0, KX, KY)
        with pytest.warns(UserWarning, match="even number"):
            SWT.bls.get_transfer_function_RT_pupil([k_even, k_even], Ek, Ek)


# ----------------------------------------------------------------------
# Green functions
# ----------------------------------------------------------------------
def _gf_pupil(s, **kwargs):
    return SWT.bls.get_signal_GF_pupil(
        [s.kx, s.kx], s.Ei_k, s.chi, s.DF, s.PM, s.d, s.NA, **kwargs
    )


def _gf_focal(s, E=None, **kwargs):
    E = np.array(s.Ei_r) if E is None else E
    return SWT.bls.get_signal_GF_focal(
        [s.x, s.x], E, [s.kx, s.kx], s.chi, s.DF, s.PM, s.d, s.NA, **kwargs
    )


class TestGF:
    @pytest.mark.parametrize("coherent", [False, True])
    def test_pupil_equals_focal(self, bls_setup, coherent):
        s = bls_setup
        sp = _gf_pupil(s, coherent_exc=coherent)
        sf = _gf_focal(s, coherent_exc=coherent)
        assert sp.shape == (3,) and np.all(sp > 0)
        assert _rel(sf, sp) < 1e-12

    def test_focal_field_from_lens(self, bls_setup):
        """With the focal field from `getFocalField` (finite window of a
        field with a sharp pupil edge), the signal differs slightly."""
        s = bls_setup
        _, _, *E = s.lens.getFocalField(0, s.x[-1], len(s.x), rtol=1e-6)
        for coherent in (False, True):
            sp = _gf_pupil(s, coherent_exc=coherent)
            sf = _gf_focal(s, np.array(E), coherent_exc=coherent)
            assert _rel(sf, sp) < 1e-2

    def test_experimental_warning(self, bls_setup):
        s = bls_setup
        with pytest.warns(UserWarning, match="experimental"):
            _gf_pupil(s)
        with pytest.warns(UserWarning, match="experimental"):
            _gf_focal(s)

    def test_scaling(self, bls_setup):
        s = bls_setup
        s1 = _gf_pupil(s)
        assert np.allclose(s1 / s1[0], [1, 0.25, 0.04])
        sig2 = SWT.bls.get_signal_GF_pupil(
            [s.kx, s.kx], [2 * c for c in s.Ei_k], 1j * s.chi, s.DF, s.PM, s.d, s.NA
        )
        assert _rel(sig2, 4 * s1) < 1e-12
        sig0 = SWT.bls.get_signal_GF_pupil(
            [s.kx, s.kx], s.Ei_k, 0 * s.chi, s.DF, s.PM, s.d, s.NA
        )
        assert np.all(sig0 == 0)

    def test_full_output(self, bls_setup):
        s = bls_setup
        N = len(s.kx)
        out = _gf_pupil(s, full_output=True)
        assert len(out) == 10
        sigma, Px, Py, Pz, Qx, Qy, Ex, Ey, Xi, Yi = out
        assert np.allclose(sigma, _gf_pupil(s))
        for a in (Px, Py, Pz, Ex, Ey):
            assert a.shape == (3, N, N)
        for a in (Qx, Qy, Xi, Yi):
            assert a.shape == (N, N)
        assert np.allclose(Qx[:, 0], s.kx) and np.allclose(Qy[0], s.kx)
        dS = (Xi[1, 0] - Xi[0, 0]) * (Yi[0, 1] - Yi[0, 0])
        assert (
            _rel(sigma, dS * np.sum(np.abs(Ex) ** 2 + np.abs(Ey) ** 2, axis=(1, 2)))
            < 1e-12
        )
        coh = _gf_pupil(s, coherent_exc=True)
        assert (
            _rel(
                coh,
                np.abs(dS * Ex.sum(axis=(1, 2))) ** 2
                + np.abs(dS * Ey.sum(axis=(1, 2))) ** 2,
            )
            < 1e-12
        )

    def test_polarization_is_convolution(self, bls_setup):
        """P_u(k) = sum_v int chi_uv(k - k') E_v(k') d^2k'/(2 pi)^2 (times
        the volume factor), checked at one point by a direct sum."""
        s = bls_setup
        _, Px, Py, Pz, *_ = _gf_pupil(s, full_output=True)
        N, c = len(s.kx), len(s.kx) // 2
        dk = s.kx[1] - s.kx[0]
        E = np.array(s.Ei_k) * (2 * np.pi) ** 2  # Fourier transform of E(r)
        i, j = c + 4, c - 2  # output point
        P = np.zeros(3, dtype=complex)
        for a in range(N):
            for b in range(N):
                ia, jb = i - a + c, j - b + c  # chi at k - k'
                if 0 <= ia < N and 0 <= jb < N:
                    P += s.chi[:, :, 0, ia, jb] @ E[:, a, b]
        P *= dk**2 / (2 * np.pi) ** 2
        res = np.array([Px[0, i, j], Py[0, i, j], Pz[0, i, j]])
        # the volume factor is the common real scale of P
        ratio = res / P
        assert np.allclose(ratio, ratio[0].real, rtol=1e-10)
        assert 0 < ratio[0].real < s.d[0]

    @pytest.mark.parametrize("angle", [0, 30, 90])
    def test_analyzer_pairs(self, bls_setup, angle):
        """Orthogonal analyzers split the signal without analyzer (the
        spatially varying radial and azimuthal analyzers only the thermal
        one, since the coherent signal is summed over the detector after
        the projection)."""
        s = bls_setup
        for coherent in (False, True):
            s0 = _gf_pupil(s, coherent_exc=coherent)
            pairs = [
                [
                    dict(output_analyzer="linear", output_analyzer_angle_deg=a)
                    for a in (angle, angle + 90)
                ],
                [dict(output_analyzer="rcp"), dict(output_analyzer="lcp")],
                [dict(output_analyzer="radial"), dict(output_analyzer="azimuthal")],
                [
                    dict(
                        output_analyzer="elliptical",
                        output_analyzer_angle_deg=angle,
                        output_analyzer_axis_ratio=r,
                    )
                    for r in (0.4, -0.4)
                ],
            ]
            for kw1, kw2 in pairs[: 2 if coherent else 3]:
                s1 = _gf_pupil(s, coherent_exc=coherent, **kw1)
                s2 = _gf_pupil(s, coherent_exc=coherent, **kw2)
                assert _rel(s1 + s2, s0) < 1e-12
            # elliptical with opposite handedness and orthogonal major axes
            kw1, kw2 = pairs[3]
            kw2["output_analyzer_angle_deg"] += 90
            s1 = _gf_pupil(s, coherent_exc=coherent, **kw1)
            s2 = _gf_pupil(s, coherent_exc=coherent, **kw2)
            assert _rel(s1 + s2, s0) < 1e-12

    def test_analyzer_inputs(self, bls_setup):
        """Strings, Jones vectors, Jones fields and callables are equivalent."""
        s = bls_setup
        ref = _gf_pupil(s, output_analyzer="linear", output_analyzer_angle_deg=30)
        e = pol.jones_vector("linear", 30)
        vec = _gf_pupil(s, output_analyzer=e)
        _, *_, Xi, Yi = _gf_pupil(s, full_output=True)
        fld = _gf_pupil(
            s, output_analyzer=np.broadcast_to(e[:, None, None], (2, *Xi.shape))
        )
        call = _gf_pupil(s, output_analyzer=lambda X, Y: (np.conj(e[0]), np.conj(e[1])))
        assert (
            _rel(vec, ref) < 1e-12
            and _rel(fld, ref) < 1e-12
            and _rel(call, ref) < 1e-12
        )
        # optics M followed by a polarizer e_p equals an analyzer e = M^H e_p
        M = pol.quarter_wave_plate(20)
        ep = pol.jones_vector("linear", 70)
        _, _, _, _, _, _, Ex, Ey, *_ = _gf_pupil(s, full_output=True)
        dS = (Xi[1, 0] - Xi[0, 0]) * (Yi[0, 1] - Yi[0, 0])
        Eout = np.einsum("ij,jf...->if...", M, np.array([Ex, Ey]))
        manual = dS * np.sum(
            np.abs(np.conj(ep[0]) * Eout[0] + np.conj(ep[1]) * Eout[1]) ** 2,
            axis=(1, 2),
        )
        assert _rel(_gf_pupil(s, output_analyzer=M.conj().T @ ep), manual) < 1e-12

    def test_analyzer_invalid(self, bls_setup):
        s = bls_setup
        with pytest.raises(ValueError):
            _gf_pupil(s, output_analyzer="diagonal")
        with pytest.raises(ValueError):
            _gf_pupil(s, output_analyzer=np.ones(3))

    def test_input_checks(self, bls_setup):
        s = bls_setup
        with pytest.raises(ValueError):  # wrong shape of the focal field
            _gf_focal(s, np.array(s.Ei_r)[:2])
        with pytest.raises(ValueError):  # wrong shape of chi
            SWT.bls.get_signal_GF_pupil(
                [s.kx, s.kx], s.Ei_k, s.chi[:, :, :, 1:], s.DF, s.PM, s.d, s.NA
            )
        with pytest.raises(ValueError):  # wrong shape of the pupil field
            SWT.bls.get_signal_GF_pupil(
                [s.kx, s.kx[1:-1]], s.Ei_k, s.chi, s.DF, s.PM, s.d, s.NA
            )
        with pytest.raises(ValueError, match="symmetric"):
            SWT.bls.get_signal_GF_pupil(
                [s.kx + 1e5, s.kx], s.Ei_k, s.chi, s.DF, s.PM, s.d, s.NA
            )

    def test_grid_notes(self, bls_setup):
        """Notes on the k-grid limit compared to 2*k0*NA and on truncated
        or aliased incident fields."""
        s = bls_setup
        kmax = 2 * s.k0 * s.NA
        for factor, match in ((0.8, "smaller than 2"), (11, "more than 10 times")):
            k = np.linspace(-factor * kmax, factor * kmax, 21)
            KX, KY = np.meshgrid(k, k, indexing="ij")
            chi = np.zeros((3, 3, 1, 21, 21))
            chi[0, 1] = 1
            Ek = s.lens.getPupilField(0, KX, KY)
            with pytest.warns(UserWarning, match=match):
                SWT.bls.get_signal_GF_pupil([k, k], Ek, chi, s.DF, s.PM, s.d, s.NA)
        # pupil field truncated by the k-grid
        k = np.linspace(-0.5 * s.k0, 0.5 * s.k0, 21)
        KX, KY = np.meshgrid(k, k, indexing="ij")
        Ek = s.lens.getPupilField(0, KX, KY)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Note: the k-grid")
            with pytest.warns(UserWarning, match="truncated"):
                SWT.bls.get_signal_GF_pupil([k, k], Ek, chi, s.DF, s.PM, s.d, s.NA)
        # aliased focal field (too coarse sampling)
        x = np.linspace(-3e-6, 3e-6, 11)
        _, _, *E = s.lens.getFocalField(0, x[-1], len(x))
        with pytest.warns(UserWarning, match="aliased"):
            SWT.bls.get_signal_GF_focal(
                [x, x], np.array(E), [s.kx, s.kx], s.chi, s.DF, s.PM, s.d, s.NA
            )


def test_getBLSsignal_deprecated(bls_setup):
    """The deprecated function still runs and warns."""
    s = bls_setup
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.warns(DeprecationWarning):
            out = SWT.bls.getBLSsignal(
                np.arange(3),
                [s.kx, s.kx],
                np.array(s.m),
                [s.x, s.x],
                np.array(s.Ei_r),
                s.DF,
                s.PM,
                s.d,
                s.NA,
                Nq=10,
            )
    assert len(out) == 7
    assert out[0].shape == (3,)
