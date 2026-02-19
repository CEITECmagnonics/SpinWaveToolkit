"""
Submodule for calculations regarding the BLS signal model.
"""

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.signal import convolve2d, fftconvolve
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import trapezoid
from SpinWaveToolkit.bls.greenAndFresnel import *

def getBLSsignal_RT3D(
    Exy, 
    Ei_fields, 
    Ej_fields,
    KxKyChi, 
    Chi,
    coherent_exc=False
):
    """
    Compute Brillouin light scattering (BLS) spectrum using the reciprocity theorem, 
    evaluating signal contribution from multiple depths.

    EXPERIMENTAL FUNCTION

    Source paper: https://doi.org/10.1126/sciadv.ady8833

    Parameters
    ----------
    Exy : tuple[ndarray]
        (m ) XY grid for the electric field.
        Tuple of two 1D arrays `(x, y)` with shapes ``(Nx,)`` and ``(Ny,)`` containing 
        the spatial coordinates of the electric field grid.
    Ei_fields : list of ndarray
        (V/m) List of the three spatial components `[Ex, Ey, Ez]` of the driving 
        electric field E_dr (incident laser). Each component must have shape 
        ``(Nx, Ny)`` for a single layer or ``(Nx, Ny, Nz)`` for multiple depths.
    Ej_fields : list of ndarray
        (V/m) List of the three spatial components `[Ex, Ey, Ez]` of the virtual 
        electric field E_v (detector side). Each component must have shape 
        ``(Nx, Ny)`` for a single layer or ``(Nx, Ny, Nz)`` for multiple depths.
    KxKyChi : tuple[ndarray]
        (rad/m) Tuple of two 1D arrays `(kx, ky)` with shapes ``(Nkx,)`` and ``(Nky,)`` 
        containing the reciprocal space coordinates of the Bloch function.
    Chi : ndarray
        () Dynamic magnetic susceptibility tensor. Must have shape ``(Nz, 3, 3, Nf, Nkx, Nky)``.
        If inputs are 2D (`Nz=1`), the first dimension can be omitted resulting in 
        shape ``(3, 3, Nf, Nkx, Nky)``, which will be reshaped automatically.
    coherent_exc : bool, optional
        () If True, calculates the coherent BLS signal (amplitudes sum first).
        If False (default), calculates the non-coherent / thermal BLS 
        signal (intensities sum first).

    Returns
    -------
    sigmaSW : ndarray
        () Calculated BLS spectrum. 
        1D array with shape ``(Nf,)``.
    qmEiEj : ndarray
        () Transfer function of the system. 
        Array with shape ``(3, 3, Nkx, Nky, Nz)``.
    """

    # --- Axis and Grid Weights ---
    x, y = Exy
    kx, ky = KxKyChi
    Nkx, Nky = len(kx), len(ky)

    # Calculate reciprocal space weights for proper integration
    dkx = np.mean(np.gradient(kx)) if Nkx > 1 else 1.0
    dky = np.mean(np.gradient(ky)) if Nky > 1 else 1.0
    dK = dkx * dky

    # --- Standardize dimensions to (Nx, Ny, Nz) ---
    def ensure_3d(arr):
        return arr[:, :, np.newaxis] if arr.ndim == 2 else arr

    Ei = np.stack([ensure_3d(f) for f in Ei_fields], axis=-1) 
    Ej = np.stack([ensure_3d(f) for f in Ej_fields], axis=-1)
    Nx, Ny, Nz, _ = Ei.shape

    # Ensure Chi has the depth dimension (Nz, 3, 3, Nf, Nkx, Nky)
    if Chi.ndim == 5:
        Chi = Chi[np.newaxis, ...]
    
    if Chi.shape[0] != Nz:
        raise ValueError(
            f"Chi depth dimension ({Chi.shape[0]}) does not match field depth dimension ({Nz})."
        )
    
    # --- Local grid spacings and Fourier factors ---
    dS_3d = np.outer(np.gradient(x), np.gradient(y))[:, :, np.newaxis]

    ExFac = np.exp(1j * np.outer(kx, x)) 
    EyFac = np.exp(1j * np.outer(ky, y)) 

    # --- Transfer Function Calculation ---
    qmEiEj = np.zeros((3, 3, Nkx, Nky, Nz), dtype=complex)

    for u in range(3):
        for v in range(3):
            # Check if this tensor component is zero across all depths to skip FT
            if not np.any(Chi[:, u, v]):
                continue
            
            F = (Ej[..., u] * Ei[..., v]) * dS_3d 
            
            # Efficient 2D Fourier Transform across all depths simultaneously
            # (Nx, Ny, Nz) @ (Nky, Ny).T -> (Nx, Nz, Nky)
            B = np.tensordot(F, EyFac, axes=([1], [1])) 
            # (Nkx, Nx) @ (Nx, Nz, Nky) -> (Nkx, Nz, Nky)
            M = np.tensordot(ExFac, B, axes=([1], [0]))
            
            qmEiEj[u, v] = np.transpose(M, (0, 2, 1)) # To (Nkx, Nky, Nz)

    # --- BLS Spectrum Assembly via Einstein Summation ---
    if coherent_exc:
        # Sum amplitudes over tensor (uv), k-space (xy), and depth (z), then square
        # Weight by dK inside the square for coherent integration
        tmp = np.einsum('uvxyz,zuvfxy->f', qmEiEj, Chi)
        sigmaSW = np.abs(tmp * dK)**2
    else:
        # Sum amplitudes over tensor (uv) and depth (z), square, then integrate intensities over k-space (xy)
        tmp = np.einsum('uvxyz,zuvfxy->fxy', qmEiEj, Chi)
        sigmaSW = np.sum(np.abs(tmp)**2, axis=(1, 2)) * dK

    return sigmaSW, qmEiEj

def getBLSsignal_RT_pupil(
    KxKy, 
    Ei_fields, 
    Ej_fields,
    Chi, 
    coherent_exc=False,
    method='fft'
):
    """
    Compute Brillouin light scattering (BLS) spectrum using the reciprocity theorem, 
    starting directly from fields in reciprocal (k) space.

    The transfer function `qmEiEj` requires the convolution of the k-space 
    fields: `qmEiEj = FT(Ej) * FT(Ei)`. 

    IMPORTANT: To maintain a valid physical representation of the convolution 
    integral, the input k-space grid (`KxKy`) MUST be strictly equidistant.

    Source paper: https://doi.org/10.1126/sciadv.ady8833

    Parameters
    ----------
    KxKy : tuple[ndarray]
        (rad/m) Tuple of two 1D arrays `(kx, ky)` with shapes ``(Nkx,)`` and ``(Nky,)`` 
        containing the reciprocal space coordinates. Assumed to be a 
        uniform/equidistant grid.
    Ei_fields : list of ndarray
        (V/m) List of the three reciprocal pupil field components `[Ekx, Eky, Ekz]` 
        corresponding to the driving field E_dr (incident laser). Each must 
        have shape ``(Nkx, Nky)``.
    Ej_fields : list of ndarray
        (V/m) List of the three reciprocal pupil field components `[Ekx, Eky, Ekz]` 
        corresponding to the virtual field E_v (detector side). Each must 
        have shape ``(Nkx, Nky)``.
    Chi : ndarray
        () Dynamic magnetic susceptibility tensor. Must have shape ``(3, 3, Nf, Nkx, Nky)`` 
        containing the tensor components `Chi_ij` for each frequency and grid point.
    coherent_exc : bool, optional
        () If True, calculates the coherent BLS signal (amplitudes sum first).
        If False (default), calculates the non-coherent / thermal BLS 
        signal (intensities sum first).
    method : {'fft', 'direct'}, optional
        () The computational method used to perform the 2D convolution. 
        - 'fft' (default): Uses `scipy.signal.fftconvolve` (Convolution Theorem). 
          Scales as O(N log N). Highly recommended for standard or large grids.
        - 'direct': Uses `scipy.signal.convolve2d` (Sliding window sum). 
          Scales as O(N^2). Exceptionally slow for large arrays, but provided 
          as an alternative for testing or very small grids.

    Returns
    -------
    sigmaSW : ndarray
        () Calculated BLS spectrum. 
        1D array with shape ``(Nf,)``.
    qmEiEj : ndarray
        () Transfer function of the system. 
        Array with shape ``(3, 3, Nkx, Nky)``.
    """
    
    if method not in ['fft', 'direct']:
        raise ValueError("Invalid method. Expected 'fft' or 'direct'.")

    # --- K-space coordinates ---
    kx, ky = KxKy
    Nkx, Nky = len(kx), len(ky)

    # --- Validation: Ensure grid is equidistant ---
    # (Required by FFT, and physically required by direct discrete convolution)
    if Nkx > 1 and not np.allclose(np.diff(kx), kx[1] - kx[0]):
        raise ValueError("The kx grid must be strictly equidistant for valid discrete convolution.")
    if Nky > 1 and not np.allclose(np.diff(ky), ky[1] - ky[0]):
        raise ValueError("The ky grid must be strictly equidistant for valid discrete convolution.")

    # --- Stack fields ---
    Ei_k = np.stack(Ei_fields, axis=-1) # Shape (Nkx, Nky, 3)
    Ej_k = np.stack(Ej_fields, axis=-1) # Shape (Nkx, Nky, 3)

    # --- K-space grid spacings ---
    dkx = kx[1] - kx[0] if Nkx > 1 else 1.0
    dky = ky[1] - ky[0] if Nky > 1 else 1.0
    dK = dkx * dky
    
    # Normalization factor for continuous convolution approximation
    normalization = dK / (2 * np.pi)**2

    # --- Calculate Transfer Function qmEiEj ---
    qmEiEj = np.zeros((3, 3, Nkx, Nky), dtype=complex)

    for u in range(3):
        for v in range(3):
            # Skip components where susceptibility is zero
            if not np.any(Chi[u, v]):
                continue
            
            # Select and perform the convolution method
            if method == 'fft':
                conv = fftconvolve(
                    Ej_k[..., u], 
                    Ei_k[..., v], 
                    mode='same' 
                )
            else: # method == 'direct'
                conv = convolve2d(
                    Ej_k[..., u], 
                    Ei_k[..., v], 
                    mode='same'
                )
            
            qmEiEj[u, v] = normalization * conv

    # --- Assemble BLS spectrum ---
    if coherent_exc:
        # Coherent sum: | Sum_k( Sum_uv( qm[u,v,k] * Chi[u,v,f,k] ) ) |^2
        tmp = np.einsum('uvxy,uvfxy->f', qmEiEj, Chi)
        sigmaSW = np.abs(tmp * dK)**2
    
    else:
        # Thermal sum: Sum_k( | Sum_uv( qm[u,v,k] * Chi[u,v,f,k] ) |^2 )
        tmp = np.einsum('uvxy,uvfxy->fxy', qmEiEj, Chi)
        sigmaSW = np.sum(np.abs(tmp)**2, axis=(1, 2)) * dK

    return sigmaSW, qmEiEj

def getBLSsignal_RT(
    Exy, 
    Ei_fields, 
    Ej_fields,
    KxKyChi, 
    Chi,
    coherent_exc=False
):
    """
    Compute Brillouin light scattering (BLS) spectrum using the reciprocity theorem.

    Source paper: https://doi.org/10.1126/sciadv.ady8833

    Parameters
    ----------
    Exy : tuple[ndarray]
        (m ) XY grid for the electric field.
        Tuple of two 1D arrays `(x, y)` with shapes ``(Nx,)`` and ``(Ny,)`` containing 
        the spatial coordinates of the electric field grid.
    Ei_fields : list of ndarray
        (V/m) List of the three spatial components `[Ex, Ey, Ez]` of the focal driving 
        electric field E_dr (incident laser). Each component must have shape 
        ``(Nx, Ny)``.
    Ej_fields : list of ndarray
        (V/m) List of the three spatial components `[Ex, Ey, Ez]` of the focal virtual 
        electric field E_v (detector side). Each component must have shape 
        ``(Nx, Ny)``.
    KxKyChi : tuple[ndarray]
        (rad/m) Tuple of two 1D arrays `(kx, ky)` with shapes ``(Nkx,)`` and ``(Nky,)`` 
        containing the reciprocal space coordinates of the Bloch function.
    Chi : ndarray
        () Dynamic magnetic susceptibility tensor. Must have shape ``(3, 3, Nf, Nkx, Nky)`` 
        containing the tensor components `Chi_ij` for each frequency and grid point. 
    coherent_exc : bool, optional
        () If True, calculates the coherent BLS signal (amplitudes sum first).
        If False (default), calculates the non-coherent / thermal BLS 
        signal (intensities sum first).

    Returns
    -------
    sigmaSW : ndarray
        () Calculated BLS spectrum. 
        1D array with shape ``(Nf,)``.
    qmEiEj : ndarray
        () Transfer function of the system. 
        Array with shape ``(3, 3, Nkx, Nky)``. Uses the same coordinates 
        as the susceptibility tensor (`KxKyChi`).
    """

    # --- Axis and Grid ---
    x, y = Exy
    kx, ky = KxKyChi
    Nkx, Nky = len(kx), len(ky)

    # --- Stack fields into (Nx, Ny, 3) ---
    Ei = np.stack(Ei_fields, axis=-1) 
    Ej = np.stack(Ej_fields, axis=-1) 

    # --- Area elements (dS) ---
    dx = np.gradient(x)
    dy = np.gradient(y)
    dS = np.outer(dx, dy)

    # k-space weights
    dkx = np.mean(np.gradient(kx)) if len(kx) > 1 else 1.0
    dky = np.mean(np.gradient(ky)) if len(ky) > 1 else 1.0
    dK = dkx * dky

    # --- Fourier phase factors ---
    ExFac = np.exp(1j * np.outer(kx, x))  # (Nkx, Nx)
    EyFac = np.exp(1j * np.outer(ky, y))  # (Nky, Ny)

    # --- Calculate Transfer Function qmEiEj (3, 3, Nkx, Nky) ---
    qmEiEj = np.zeros((3, 3, Nkx, Nky), dtype=complex)
    for u in range(3):
        for v in range(3):
            # Skip components where susceptibility is zero
            if not np.any(Chi[u, v]):
                continue
            
            # Element-wise product weighted by area
            F = (Ej[..., u] * Ei[..., v]) * dS
            
            # Fast 2D Fourier Transform via matrix multiplication
            # Result: (Nkx, Nky)
            qmEiEj[u, v] = ExFac @ (F @ EyFac.T)

    # Assemble BLS spectrum by weighting overlaps with susceptibility
    if coherent_exc:
        # Coherent sum: | Sum_k( Sum_uv( qm[u,v,k] * Chi[u,v,f,k] ) ) |^2
        # 'uvxy' are qmEiEj dims (3, 3, Nkx, Nky)
        # 'uvfxy' are Chi dims (3, 3, Nf, Nkx, Nky)
        # einsum sums over u,v,x,y, leaving just 'f'
        tmp = np.einsum('uvxy,uvfxy->f', qmEiEj, Chi)
        sigmaSW = np.abs(tmp * dK)**2
    
    else:
        # Thermal sum: Sum_k( | Sum_uv( qm[u,v,k] * Chi[u,v,f,k] ) |^2 )
        # einsum sums over u,v, leaving 'f,x,y'
        tmp = np.einsum('uvxy,uvfxy->fxy', qmEiEj, Chi)
        
        # Sum over k-space (x and y axes)
        sigmaSW = np.sum(np.abs(tmp)**2, axis=(1, 2)) * dK

    return sigmaSW, qmEiEj

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
        Sweep vector of the Bloch functions with shape ``(Nf,)``.
        Usually frequency of spin waves.
    KxKyBloch : tuple[ndarray]
        (rad/m) Tuple of two vectors with shapes ``(Nkx,)``, ``(Nky,)`` 
        containing the kx and ky coordinates of the Bloch function.
    Bloch : ndarray
        Array with shape ``(3, Nf, Nkx, Nky)`` containing the Bloch 
        function components ``(Mx, My, Mz)`` for each frequency and KxKy 
        grid point.
    Exy : tuple[ndarray]
        (m ) XY grid for the electric field.
        Tuple of two vectors with shapes ``(Nx,)``, ``(Ny,)`` containing 
        the X and Y coordinates of the electric field.
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
        (m ) thickness of all layers in the stack excluding the
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
        (m ) wavelength of the light.  Default is 532e-9.
    collectionSpot : float, optional
        (m ) collection spot size - used here as the beam waist.  Default
        is 1e-6.
    focalLength : float, optional
        (m ) focal length of the lens.  Default is 1e-3.

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
    Px, Py, Pz : ndarray
        (V/m) induced polarization in the magnetic layer.  Corresponds
        to `P` in eq. (3) in Wojewoda et al. PRB 110, 224428 (2024).
        Each array has shape ``(Nf, 2*Nq-1, 2*Nq-1)``.
    Qx, Qy : ndarray
        (rad/m) k-space grids for polarizations `Px`, `Py`, `Pz`.
        Each array has shape ``(2*Nq-1, 2*Nq-1)``.
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
    # Evaluate the Fresnel coefficients at each Q:
    tp = htp(Q)  # htp at once, assumed shape (2, *Q.shape)
    ts = hts(Q)  # hts at once, assumed shape (2, *Q.shape)
    # Replace NaNs with zeros in the two (assumed) components.
    tp_fixed = np.nan_to_num(tp, nan=0)
    ts_fixed = np.nan_to_num(ts, nan=0)
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

    # Preallocate polarization.
    # Loop over frequencies returned by SpinWaveGreen.
    # Here we assume that the first dimension of Bloch (after the component index)
    # corresponds to the sweep index (and that len(SweepBloch)==Nf).
    Px = np.empty((Nf, Nq * 2 - 1, Nq * 2 - 1), dtype=complex)
    Py = np.empty((Nf, Nq * 2 - 1, Nq * 2 - 1), dtype=complex)
    Pz = np.empty((Nf, Nq * 2 - 1, Nq * 2 - 1), dtype=complex)
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
        Px[i] = convolve2d(
            interp_fftEI[2, :, :], 1j * Bloch_interp_My, mode="same"
        ) + convolve2d(interp_fftEI[1, :, :], -1j * Bloch_interp_Mz, mode="same")
        Py[i] = convolve2d(
            interp_fftEI[0, :, :], 1j * Bloch_interp_Mz, mode="same"
        ) + convolve2d(interp_fftEI[2, :, :], -1j * Bloch_interp_Mx, mode="same")
        Pz[i] = convolve2d(
            interp_fftEI[1, :, :], 1j * Bloch_interp_Mx, mode="same"
        ) + convolve2d(interp_fftEI[0, :, :], -1j * Bloch_interp_My, mode="same")
        # -------------------------------------------------------------

        # --- Calculate the p- and s-polarized electric field contributions ---
        # pGF and sGF are assumed to be 3×2 structures (lists of lists or similar).
        Ep = pGF[0][0] * Px[i] * np.exp(-1j * Kzs * d[source_layer_index - 1]) + pGF[0][
            1
        ] * Px[i] * np.exp(1j * Kzs * d[source_layer_index - 1])
        Ep += pGF[1][0] * Py[i] * np.exp(-1j * Kzs * d[source_layer_index - 1]) + pGF[
            1
        ][1] * Py[i] * np.exp(1j * Kzs * d[source_layer_index - 1])
        Ep += pGF[2][0] * Pz[i] * np.exp(-1j * Kzs * d[source_layer_index - 1]) + pGF[
            2
        ][1] * Pz[i] * np.exp(1j * Kzs * d[source_layer_index - 1])

        Es = sGF[0][0] * Px[i] * np.exp(-1j * Kzs * d[source_layer_index - 1]) + sGF[0][
            1
        ] * Px[i] * np.exp(1j * Kzs * d[source_layer_index - 1])
        Es += sGF[1][0] * Py[i] * np.exp(-1j * Kzs * d[source_layer_index - 1]) + sGF[
            1
        ][1] * Py[i] * np.exp(1j * Kzs * d[source_layer_index - 1])
        Es += sGF[2][0] * Pz[i] * np.exp(-1j * Kzs * d[source_layer_index - 1]) + sGF[
            2
        ][1] * Pz[i] * np.exp(1j * Kzs * d[source_layer_index - 1])
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
    # and the polarization currents in the magnetic layer [three 3D arrays of shape
    # (Nf, 2*Nq-1, 2*Nq-1)] with the respective wavevector grids [two 2D arrays of shape
    # (2*Nq-1, 2*Nq-1)]
    return ExS, EyS, Px, Py, Pz, Qx, Qy
