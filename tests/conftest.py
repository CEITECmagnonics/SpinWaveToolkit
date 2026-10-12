"""Shared fixtures for the tests of the `bls` submodule."""

from types import SimpleNamespace
import numpy as np
import pytest
import SpinWaveToolkit as SWT


@pytest.fixture(scope="session")
def bls_setup():
    """Small BLS setup shared by the tests of the signal functions.

    Contains an objective lens, a symmetric odd k-grid reaching beyond
    ``2*k0*NA``, pupil fields polarized along x (incident) and y
    (detected), the focal fields synthesized from them on the conjugate
    real-space grid (exact discrete inverse Fourier transform, so that
    the focal and pupil functions get the same input), a susceptibility
    of a Gaussian magnon peak (3 frequencies, linear + quadratic MO) and
    a NiFe-like layer stack.
    """
    wavelength, NA = 532e-9, 0.75
    k0 = 2 * np.pi / wavelength
    lens = SWT.bls.ObjectiveLens(wavelength, NA, 1.0, 1e-3)

    N = 41
    kmax = 2.2 * k0 * NA
    kx = np.linspace(-kmax, kmax, N)
    dk = kx[1] - kx[0]
    KX, KY = np.meshgrid(kx, kx, indexing="ij")
    Ei_k = lens.getPupilField(0, KX, KY, pol_type="linear", pol_angle=0)
    Ej_k = lens.getPupilField(0, KX, KY, pol_type="linear", pol_angle=90)

    # conjugate real-space grid and E(r) = int E_k(k) exp(i k.r) d^2k
    dx = 2 * np.pi / (N * dk)
    x = (np.arange(N) - (N - 1) / 2) * dx
    F = np.exp(1j * np.outer(x, kx))
    Ei_r = [F @ c @ F.T * dk**2 for c in Ei_k]
    Ej_r = [F @ c @ F.T * dk**2 for c in Ej_k]

    # susceptibility of a magnon peak at kx = 5 rad/um (M || y)
    b = np.exp(-((KX - 5e6) ** 2 + KY**2) / 3e6**2)
    bloch = np.stack([b, 0.5 * b, 0.2 * b])  # (Nf, Nkx, Nky)
    m = (bloch, np.zeros_like(bloch), -1j * bloch)
    chi = SWT.bls.susceptibilities.mo_linear(m)
    chi += SWT.bls.susceptibilities.mo_quadratic(m, linearize_along="y")

    return SimpleNamespace(
        wavelength=wavelength,
        NA=NA,
        k0=k0,
        lens=lens,
        kx=kx,
        KX=KX,
        KY=KY,
        Ei_k=Ei_k,
        Ej_k=Ej_k,
        x=x,
        Ei_r=Ei_r,
        Ej_r=Ej_r,
        bloch=bloch,
        m=m,
        chi=chi,
        DF=np.array([1, -8.1653 + 15.348j, 17.237 + 0.43004j]),
        PM=np.ones(3),
        d=np.array([30e-9]),
    )
