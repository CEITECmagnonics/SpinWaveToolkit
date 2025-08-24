"""
Submodule for calculations regarding the BLS signal model.
"""

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.signal import convolve2d
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import trapezoid
from SpinWaveToolkit.greenAndFresnel import *


def getBLSsignal(
    SweepBloch,
    KxKyBloch,
    Bloch,
    Exy,
    E,
    DF,
    PM,
    d,
    NA,
    Nq=30,
    source_layer_index=1,
    output_layer_index=0,
    wavelength=532e-9,
    collectionSpot=1e-6,
    focalLength=1e-3,
):
    """
    Calculate the BLS signal from the Bloch functions.

    Parameters
    ----------
    SweepBloch : ndarray
        Sweep vector of the Bloch functions. Usually frequency of spin
        waves.
    KxKyBloch : tuple[ndarray]
        (rad/m) tuple containing the 1D grids ``(kx_grid, ky_grid)`` on
        which the Bloch functions are defined.
    Bloch : ndarray
        Array with shape ``(3, Nf, Nkx, Nky)`` containing the Bloch function
        components ``(Mx, My, Mz)`` for each frequency and KxKy grid point.
    Exy : ndarray
        (m) XY grid for the electric field.
        2D array with shape ``(Ny, Nx)`` containing the X and Y
        coordinates of the electric field.
    E : ndarray
        (V/m) 3D array with shape ``(3, Ny, Nx)`` containing the X, Y, Z
        components of the electric field.
    DF : ndarray
        () vector of the complex dielectric functions for each material
        in the stack.
    PM : ndarray
        () vector of the complex permeability functions for each
        material in the stack.
    d : ndarray
        (m) thickness of all layers in the stack excluding the
        superstrate and substrate.  Usually just the thickness of the
        magnetic layer.
    NA : float
        Numerical aperture of the optical system.
    Nq : int, optional
        Number of points in the q-space grid.  Default is 30.
    source_layer_index : int, optional
        Index of the source layer in the stack.  Default is 1.
    output_layer_index : int, optional
        Index of the output layer in the stack.  Default is 0.
    wavelength : float, optional
        (m) wavelength of the light.  Default is 532e-9.
    collectionSpot : float, optional
        (m) collection spot size - used here as the beam waist.  Default
        is 1e-6.
    focalLength : float, optional
        (m) focal length of the lens.  Default is 1e-3.

    Returns
    -------
    ExS : ndarray
        (V/m) scattered electric field in the X axis.
        1D array with shape ``(Nf,)`` containing the scattered electric
        field in the X direction for each frequency in SweepBloch.
    EyS : ndarray
        (V/m) scattered electric field in the Y axis.
        1D array with shape ``(Nf,)`` containing the scattered electric
        field in the Y direction for each frequency in SweepBloch.
    """
    k0 = 2 * np.pi / wavelength

    # --- Set up q-space grid (qx and qy) ---
    qxHalf = np.linspace(0, 1.1, Nq) * k0
    qx = np.concatenate((-qxHalf[1:][::-1], qxHalf))
    # dqx_raw = np.diff(qx)  # ### unused?
    # dqx_padded = np.concatenate(([0], dqx_raw, [0]))  # ### unused?
    # dqx = (dqx_padded[:-1] + dqx_padded[1:]) / 2  # ### unused?

    # qy is taken identical to qx
    qy = qx.copy()
    # dqy = dqx.copy()  # ### unused?

    # Create the 2D grid using ndgrid convention (like Matlab)
    Qx, Qy = np.meshgrid(qx, qy, indexing="ij")
    Q = np.sqrt(Qx**2 + Qy**2)
    # Use complex square root to avoid NaNs for negative arguments
    Kzs = np.sqrt(d[source_layer_index - 1] * k0**2 - Q**2 + 0j)
    Kz = np.sqrt(k0**2 - Q**2 + 0j)
    # -------------------------------------------------------------

    # --- Compute the Fourier transform of the electric field components ---
    # We assume E has shape (3, Ny, Nx) where E[0] is the X component, etc.
    fftEI = np.empty_like(E, dtype=complex)
    for comp in range(3):
        # Apply ifftshift in both axes, then fft2, then fftshift back.
        temp = ifftshift(E[comp])
        temp = fft2(temp)
        temp = fftshift(temp)
        fftEI[comp, :, :] = temp
    # -------------------------------------------------------------

    # --- Suppose Exy is given as a tuple of 2D arrays (X, Y) for the spatial coordinates ---
    EX, EY = Exy  # e.g. X, Y = np.meshgrid(x, y, indexing='ij')
    # Determine grid spacings (assuming uniform spacing)
    dx = EX[1] - EX[0]
    dy = EY[1] - EY[0]
    Nx = EX.shape[0]
    Ny = EY.shape[0]

    # --- Compute the Fourier domain grid corresponding to the spatial grid ---
    # The FFT frequency bins (in radians per meter) are given by:
    kx_fft = fftshift(2 * np.pi * np.fft.fftfreq(Nx, d=dx))
    ky_fft = fftshift(2 * np.pi * np.fft.fftfreq(Ny, d=dy))

    # --- Interpolate the computed FFT of the E-field onto the Qx, Qy grid ---
    # Here fftEI has shape (3, Ny, Nx) and is defined on (KX_fft, KY_fft)
    interp_fftEI = np.empty((3, Qx.shape[0], Qx.shape[1]), dtype=complex)
    for comp in range(3):
        # Create an interpolator for each component
        interp_func = RegularGridInterpolator(
            (kx_fft, ky_fft), fftEI[comp, :, :], bounds_error=False, fill_value=0
        )
        # Prepare the target points as an (M,2) array where M = number of Qx points
        points = np.stack([Qx.ravel(), Qy.ravel()], axis=-1)
        interp_fftEI[comp, :, :] = interp_func(points).reshape(Qx.shape)

    # --- Prepare for frequency loop ---
    # Get the kx, ky grid for Bloch functions (assumed to be 1D arrays)
    kx_grid, ky_grid = KxKyBloch

    # Compute a volume factor by integrating an exponential decay over the layer thickness
    zs = np.linspace(0, d[source_layer_index - 1], 100)
    ExtinCoefMagLayer = np.sqrt(
        (abs(DF[source_layer_index]) - np.real(DF[source_layer_index])) / 2
    )
    Volume = np.exp(-ExtinCoefMagLayer * k0 * zs)
    VolumeFac = trapezoid(Volume, zs)

    # Multiply the electric field by the volume factor
    interp_fftEI *= VolumeFac

    # Prepare arrays to store the results for each frequency
    Nf = len(SweepBloch)
    ExS = np.zeros(Nf, dtype=complex)
    EyS = np.zeros(Nf, dtype=complex)

    # --- Evaluate Fresnel coefficients and spherical Green functions ---
    # The Fresnelq function is expected to return two objects (htp and hts) that can be evaluated on Q.
    htp, hts = fresnel_coefficients(
        lambda_=wavelength,
        DF=DF,
        PM=PM,
        d=d,
        source_layer_index=source_layer_index,
        output_layer_index=output_layer_index,
    )
    # Evaluate on Q; here we assume htp and hts are callable (or arrays) so that:
    tp = np.zeros((*Q.shape, 2), dtype=complex)
    ts = np.zeros((*Q.shape, 2), dtype=complex)
    # Loop over all Q values and evaluate the Fresnel coefficients
    for ix in range(Q.shape[0]):
        for iy in range(Q.shape[1]):
            tp[ix, iy, 0] = htp(Q[ix, iy])[0]  # First component of htp
            tp[ix, iy, 1] = htp(Q[ix, iy])[1]  # Second component of htp
            ts[ix, iy, 0] = hts(Q[ix, iy])[0]  # First component of hts
            ts[ix, iy, 1] = hts(Q[ix, iy])[1]  # Second component of hts
    # Replace NaNs with zeros in the two (assumed) components.
    tp0 = np.nan_to_num(tp[..., 0], nan=0)
    tp1 = np.nan_to_num(tp[..., 1], nan=0)
    ts0 = np.nan_to_num(ts[..., 0], nan=0)
    ts1 = np.nan_to_num(ts[..., 1], nan=0)
    tp_fixed = [tp0, tp1]
    ts_fixed = [ts0, ts1]
    # Compute the spherical Green functions
    pGF, sGF = sph_green_function(
        Kx=Qx,
        Ky=Qy,
        DFMagLayer=DF[source_layer_index],
        wavelength=wavelength,
        tp=tp_fixed,
        ts=ts_fixed,
    )
    # -------------------------------------------------------------

    # Loop over frequencies returned by SpinWaveGreen.
    # Here we assume that the first dimension of Bloch (after the component index)
    # corresponds to the sweep index (and that len(SweepBloch)==Nf).
    for i, s in enumerate(SweepBloch):
        # --- Interpolate Bloch function components onto the Qx-Qy grid ---
        # We assume Bloch has shape (3, Nf, Nkx, Nky)
        interp_Mx = RegularGridInterpolator(
            (kx_grid, ky_grid), Bloch[0, i, :, :], bounds_error=False, fill_value=0
        )
        interp_My = RegularGridInterpolator(
            (kx_grid, ky_grid), Bloch[1, i, :, :], bounds_error=False, fill_value=0
        )
        interp_Mz = RegularGridInterpolator(
            (kx_grid, ky_grid), Bloch[2, i, :, :], bounds_error=False, fill_value=0
        )
        # Evaluate at the (Qx, Qy) points:
        points = np.stack([Qx.ravel(), Qy.ravel()], axis=-1)
        Bloch_interp_Mx = interp_Mx(points).reshape(Qx.shape)
        Bloch_interp_My = interp_My(points).reshape(Qx.shape)
        Bloch_interp_Mz = interp_Mz(points).reshape(Qx.shape)
        # -------------------------------------------------------------

        # --- Convolve the electric field with the (interpolated) Bloch components ---
        # We do not care about
        Px = convolve2d(
            interp_fftEI[2, :, :], 1j * Bloch_interp_My, mode="same"
        ) + convolve2d(interp_fftEI[1, :, :], -1j * Bloch_interp_Mz, mode="same")
        Py = convolve2d(
            interp_fftEI[0, :, :], 1j * Bloch_interp_Mz, mode="same"
        ) + convolve2d(interp_fftEI[2, :, :], -1j * Bloch_interp_Mx, mode="same")
        Pz = convolve2d(
            interp_fftEI[1, :, :], 1j * Bloch_interp_Mx, mode="same"
        ) + convolve2d(interp_fftEI[0, :, :], -1j * Bloch_interp_My, mode="same")
        # -------------------------------------------------------------

        # --- Calculate the p- and s-polarized electric field contributions ---
        # pGF and sGF are assumed to be 3×2 structures (lists of lists or similar).
        Ep = pGF[0][0] * Px * np.exp(-1j * Kzs * d[source_layer_index - 1]) + pGF[0][
            1
        ] * Px * np.exp(1j * Kzs * d[source_layer_index - 1])
        Ep += pGF[1][0] * Py * np.exp(-1j * Kzs * d[source_layer_index - 1]) + pGF[1][
            1
        ] * Py * np.exp(1j * Kzs * d[source_layer_index - 1])
        Ep += pGF[2][0] * Pz * np.exp(-1j * Kzs * d[source_layer_index - 1]) + pGF[2][
            1
        ] * Pz * np.exp(1j * Kzs * d[source_layer_index - 1])

        Es = sGF[0][0] * Px * np.exp(-1j * Kzs * d[source_layer_index - 1]) + sGF[0][
            1
        ] * Px * np.exp(1j * Kzs * d[source_layer_index - 1])
        Es += sGF[1][0] * Py * np.exp(-1j * Kzs * d[source_layer_index - 1]) + sGF[1][
            1
        ] * Py * np.exp(1j * Kzs * d[source_layer_index - 1])
        Es += sGF[2][0] * Pz * np.exp(-1j * Kzs * d[source_layer_index - 1]) + sGF[2][
            1
        ] * Pz * np.exp(1j * Kzs * d[source_layer_index - 1])
        # -------------------------------------------------------------

        # --- Convert to X and Y components in the laboratory frame ---
        # Avoid division by zero: when Q==0 set cosPhi=1 and sinPhi=0.
        cosPhi = np.divide(Qx, Q, out=np.ones_like(Qx), where=Q != 0)
        sinPhi = np.divide(Qy, Q, out=np.zeros_like(Qy), where=Q != 0)
        Ex_field = Ep * cosPhi - Es * sinPhi
        Ey_field = Ep * sinPhi + Es * cosPhi
        # -------------------------------------------------------------

        # --- Apply a polarization-dependent factor ---
        Factor = (
            (-2j * np.pi * np.sqrt(Kz * k0))
            * np.exp(1j * k0 * focalLength)
            / focalLength
        )
        Ex_field *= Factor
        Ey_field *= Factor
        # -------------------------------------------------------------

        # --- Compute real-space grids and apply the point-spread filter ---
        # This represent limited ability to propagate the electric field to the detector.
        dkx = qx[1] - qx[0]
        dky = qy[1] - qy[0]
        Nxi = len(qx)
        Nyi = len(qy)
        DXi = 2 * np.pi / dkx
        DYi = 2 * np.pi / dky
        dxi = DXi / Nxi
        dyi = DYi / Nyi
        xi = np.linspace(-(Nxi - 1) / 2, (Nxi - 1) / 2, Nxi) * dxi
        yi = np.linspace(-(Nyi - 1) / 2, (Nyi - 1) / 2, Nyi) * dyi
        Xi, Yi = np.meshgrid(xi, yi, indexing="ij")
        PSFFilter = np.exp(-(Xi**2 + Yi**2) / (2 * np.pi**2 * collectionSpot**2))
        # -------------------------------------------------------------

        # --- Transform back to real space with an applied numerical aperture mask ---
        # Create a mask for Q values within k0*NA.
        mask = (Q <= k0 * NA).astype(float)
        # Compute a common scaling factor (note: np.size returns the total number of elements).
        factor_fft = (
            (focalLength / k0) ** 2 * Ex_field.size / (4 * np.pi**2) * dkx * dky
        )
        Ex_real = factor_fft * fftshift(ifft2(ifftshift(Ex_field * mask)))
        Ey_real = factor_fft * fftshift(ifft2(ifftshift(Ey_field * mask)))
        # Apply the point-spread (PSF) filter in real space.
        Ex_real *= PSFFilter
        Ey_real *= PSFFilter
        # -------------------------------------------------------------

        # --- Compute signal integrals over the image (spatial integration on the detector) ---
        ExS[i] = dxi * dyi * np.sum(Ex_real)
        EyS[i] = dxi * dyi * np.sum(Ey_real)

    # Return the computed scattered electric field in x and y direction (1D array over sweep)
    return ExS, EyS
