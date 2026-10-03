"""
Submodule for calculations regarding the BLS signal model.
"""

from warnings import warn
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import scipy.fft as spfft
from scipy.signal import convolve2d, fftconvolve
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import trapezoid
from SpinWaveToolkit.bls.greenAndFresnel import *
from SpinWaveToolkit.bls.polarization import jones_vector

__all__ = [
    "get_signal_RT_focal_3d",
    "get_signal_RT_pupil",
    "get_transfer_function_RT_pupil",
    "get_signal_RT_focal",
    "get_signal_GF_focal",
    "get_signal_GF_pupil",
    "getBLSsignal",
]


def get_signal_RT_focal_3d(Exy, Ei_fields, Ej_fields, KxKyChi, Chi, coherent_exc=False):
    """
    Compute Brillouin light scattering (BLS) spectrum using the
    reciprocity theorem, evaluating signal contribution from multiple
    depths.

    .. warning::

       This is an experimental function. Syntax and behavior may change
       in future releases. Please verify the results carefully.

    Source paper: https://doi.org/10.1126/sciadv.ady8833

    Parameters
    ----------
    Exy : list[ndarray]
        (m ) XY grid for the electric field.
        List of two 1D arrays `(x, y)` with shapes ``(Nx,)`` and
        ``(Ny,)`` containing the spatial coordinates of the electric
        field grid.
    Ei_fields : list[ndarray]
        (V/m) list of the three spatial components `[Ex, Ey, Ez]` of the
        driving electric field E_dr (incident laser).  Each component
        must have shape ``(Nx, Ny)`` for a single layer or
        ``(Nx, Ny, Nz)`` for multiple depths.
    Ej_fields : list[ndarray]
        (V/m) list of the three spatial components `[Ex, Ey, Ez]` of the
        virtual electric field E_v (detector side).  Each component must
        have shape ``(Nx, Ny)`` for a single layer or ``(Nx, Ny, Nz)``
        for multiple depths.
    KxKyChi : list[ndarray]
        (rad/m) list of two 1D arrays `(kx, ky)` with shapes ``(Nkx,)``
        and ``(Nky,)`` containing the reciprocal space coordinates of
        the Bloch function.
    Chi : ndarray
        () dynamic magneto-optic susceptibility tensor with shape
        ``(Nz, 3, 3, Nf, Nkx, Nky)``.  If inputs are 2D (``Nz=1``), the
        first dimension can be omitted resulting in shape
        ``(3, 3, Nf, Nkx, Nky)``, which will be reshaped automatically.
    coherent_exc : bool, optional
        If True, calculates the coherent BLS signal (amplitudes sum
        first).  If False (default), calculates the non-coherent/thermal
        BLS signal (intensities sum first).

    Returns
    -------
    sigma : ndarray
        () calculated BLS spectrum.  1D array with shape ``(Nf,)``.
    qmEiEj : ndarray
        () transfer function of the system. Array with shape
        ``(3, 3, Nkx, Nky, Nz)``.  Uses the same coordinates as the
        susceptibility tensor (`KxKyChi`).

    See also
    --------
    get_signal_RT_focal : A similar function without z-resolution.
    get_signal_RT_pupil, get_signal_GF_focal

    """
    warn(
        "`get_signal_RT_focal_3d` is an experimental function and may be subject to change."
        + " Please verify results carefully.",
        UserWarning,
        stacklevel=2,
    )

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
    _, _, Nz, _ = Ei.shape

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

            qmEiEj[u, v] = np.transpose(M, (0, 2, 1))  # To (Nkx, Nky, Nz)

    # --- BLS Spectrum Assembly via Einstein Summation ---
    if coherent_exc:
        # Sum amplitudes over tensor (uv), k-space (xy), and depth (z), then square
        # Weight by dK inside the square for coherent integration
        tmp = np.einsum("uvxyz,zuvfxy->f", qmEiEj, Chi)
        sigma = np.abs(tmp * dK) ** 2
    else:
        # Sum amplitudes over tensor (uv) and depth (z), square, then integrate intensities over k-space (xy)
        tmp = np.einsum("uvxyz,zuvfxy->fxy", qmEiEj, Chi)
        sigma = np.sum(np.abs(tmp) ** 2, axis=(1, 2)) * dK

    return sigma, qmEiEj


def get_signal_RT_pupil(
    KxKy, Ei_fields, Ej_fields, Chi, coherent_exc=False, conv_method="fft"
):
    """
    Compute Brillouin light scattering (BLS) spectrum using the
    reciprocity theorem, starting directly from the electric fields in
    reciprocal (k) space.

    The transfer function `qmEiEj` requires the convolution of the
    k-space fields: `qmEiEj = FT(Ej) * FT(Ei)`.

    .. important::

       To maintain a valid physical representation of the convolution
       integral, the input k-space grid (`KxKy`) MUST be strictly
       equidistant and symmetric with respect to ``k = 0`` (preferably
       with an odd number of points, so that ``k = 0`` is a grid point).

    Source paper: https://doi.org/10.1126/sciadv.ady8833

    Parameters
    ----------
    KxKy : list[ndarray]
        (rad/m) list of two 1D arrays `(kx, ky)` with shapes ``(Nkx,)``
        and ``(Nky,)`` containing the reciprocal space coordinates.
        Must be a uniform/equidistant grid symmetric with respect to
        ``k = 0``.
    Ei_fields : list[ndarray]
        (V/m) list of the three reciprocal pupil field components
        `[Ekx, Eky, Ekz]` corresponding to the driving field E_dr
        (incident laser), as given by
        :meth:`~SpinWaveToolkit.bls.ObjectiveLens.getPupilField`.  Each
        must have shape ``(Nkx, Nky)``.
    Ej_fields : list[ndarray]
        (V/m) list of the three reciprocal pupil field components
        `[Ekx, Eky, Ekz]` corresponding to the virtual field E_v
        (detector side). Each must have shape ``(Nkx, Nky)``.
    Chi : ndarray
        () dynamic magneto-optic susceptibility tensor with shape
        ``(3, 3, Nf, Nkx, Nky)``, containing the tensor components
        `Chi_ij` for each frequency and k-space grid point.
    coherent_exc : bool, optional
        If True, calculates the coherent BLS signal (amplitudes sum
        first).  If False (default), calculates the non-coherent/thermal
        BLS signal (intensities sum first).
    conv_method : {"fft", "direct"}, optional
        The computational method used to perform the 2D convolution.

        - "fft" (default): Uses :func:`scipy.signal.fftconvolve`
          (Convolution Theorem).  Scales as O(N log N). Highly
          recommended for standard or large grids.

        - "direct": Uses :func:`scipy.signal.convolve2d` (Sliding window
          sum).  Scales as O(N^2).  Exceptionally slow for large arrays,
          but provided as an alternative for testing or very small
          grids.

    Returns
    -------
    sigma : ndarray
        () calculated BLS spectrum.  1D array with shape ``(Nf,)``.
    qmEiEj : ndarray
        () transfer function of the system. Array with shape
        ``(3, 3, Nkx, Nky)``.  Uses the same coordinates as the
        electric fields and susceptibility tensor (`KxKy`).

    See also
    --------
    get_transfer_function_RT_pupil, get_signal_RT_focal, get_signal_GF_focal

    """
    # --- Calculate Transfer Function qmEiEj ---
    # Skip components where susceptibility is zero
    mask = np.any(Chi, axis=tuple(range(2, np.ndim(Chi))))
    qmEiEj = get_transfer_function_RT_pupil(
        KxKy, Ei_fields, Ej_fields, conv_method=conv_method, mask=mask
    )

    # --- K-space grid spacings ---
    kx, ky = KxKy
    dkx = kx[1] - kx[0] if len(kx) > 1 else 1.0
    dky = ky[1] - ky[0] if len(ky) > 1 else 1.0
    dK = dkx * dky

    # --- Assemble BLS spectrum ---
    if coherent_exc:
        # Coherent sum: | Sum_k( Sum_uv( qm[u,v,k] * Chi[u,v,f,k] ) ) |^2
        tmp = np.einsum("uvxy,uvfxy->f", qmEiEj, Chi)
        sigma = np.abs(tmp * dK) ** 2

    else:
        # Thermal sum: Sum_k( | Sum_uv( qm[u,v,k] * Chi[u,v,f,k] ) |^2 )
        tmp = np.einsum("uvxy,uvfxy->fxy", qmEiEj, Chi)
        sigma = np.sum(np.abs(tmp) ** 2, axis=(1, 2)) * dK

    return sigma, qmEiEj


def get_transfer_function_RT_pupil(
    KxKy, Ei_fields, Ej_fields, conv_method="fft", mask=None
):
    """
    Compute the transfer function of the BLS system using the
    reciprocity theorem, starting directly from the electric fields in
    reciprocal (k) space.

    The transfer function `qmEiEj` is given by the convolution of the
    k-space fields: `qmEiEj = FT(Ej) * FT(Ei)`.  It does not depend on
    the magnetization dynamics and can therefore be reused for
    calculation of BLS spectra for different susceptibility tensors,
    see :func:`get_signal_RT_pupil`.

    .. important::

       To maintain a valid physical representation of the convolution
       integral, the input k-space grid (`KxKy`) MUST be strictly
       equidistant and symmetric with respect to ``k = 0`` (preferably
       with an odd number of points, so that ``k = 0`` is a grid point).

    Source paper: https://doi.org/10.1126/sciadv.ady8833

    Parameters
    ----------
    KxKy : list[ndarray]
        (rad/m) list of two 1D arrays `(kx, ky)` with shapes ``(Nkx,)``
        and ``(Nky,)`` containing the reciprocal space coordinates.
        Must be a uniform/equidistant grid symmetric with respect to
        ``k = 0``.
    Ei_fields : list[ndarray]
        (V/m) list of the three reciprocal pupil field components
        `[Ekx, Eky, Ekz]` corresponding to the driving field E_dr
        (incident laser), as given by
        :meth:`~SpinWaveToolkit.bls.ObjectiveLens.getPupilField`.  Each
        must have shape ``(Nkx, Nky)``.
    Ej_fields : list[ndarray]
        (V/m) list of the three reciprocal pupil field components
        `[Ekx, Eky, Ekz]` corresponding to the virtual field E_v
        (detector side). Each must have shape ``(Nkx, Nky)``.
    conv_method : {"fft", "direct"}, optional
        The computational method used to perform the 2D convolution.

        - "fft" (default): Uses :func:`scipy.signal.fftconvolve`
          (Convolution Theorem).  Scales as O(N log N). Highly
          recommended for standard or large grids.

        - "direct": Uses :func:`scipy.signal.convolve2d` (Sliding window
          sum).  Scales as O(N^2).  Exceptionally slow for large arrays,
          but provided as an alternative for testing or very small
          grids.
    mask : array_like or None, optional
        Boolean array with shape ``(3, 3)``.  Components `qmEiEj[u, v]`
        where `mask[u, v]` is False are not calculated and left zero.
        If None (default), all components are calculated.

    Returns
    -------
    qmEiEj : ndarray
        () transfer function of the system.  Array with shape
        ``(3, 3, Nkx, Nky)``.  Uses the same coordinates as the
        electric fields (`KxKy`).

    See also
    --------
    get_signal_RT_pupil

    """
    if conv_method not in ["fft", "direct"]:
        raise ValueError("Invalid conv_method. Expected 'fft' or 'direct'.")
    if mask is None:
        mask = np.ones((3, 3), dtype=bool)

    # --- K-space coordinates ---
    kx, ky = KxKy
    Nkx, Nky = len(kx), len(ky)

    # --- Validation: Ensure grid is equidistant and symmetric ---
    dkx, dky = _check_pupil_grid(kx, ky)

    # --- Stack fields ---
    Ei_k = np.stack(Ei_fields, axis=-1)  # Shape (Nkx, Nky, 3)
    Ej_k = np.stack(Ej_fields, axis=-1)  # Shape (Nkx, Nky, 3)

    # --- K-space grid spacings ---
    dK = dkx * dky

    # Normalization factor for continuous convolution approximation: the
    # pupil fields (E(r) = int E_k(k) exp(i k.r) d^2k) are converted to
    # Fourier transforms by (2*pi)**2 each, and the convolution measure
    # is dK/(2*pi)**2
    normalization = (2 * np.pi) ** 2 * dK

    # --- Calculate Transfer Function qmEiEj ---
    qmEiEj = np.zeros((3, 3, Nkx, Nky), dtype=complex)

    for u in range(3):
        for v in range(3):
            if not mask[u, v]:
                continue

            # Select and perform the convolution method
            if conv_method == "fft":
                conv = fftconvolve(Ej_k[..., u], Ei_k[..., v], mode="same")
            else:  # conv_method == 'direct'
                conv = convolve2d(Ej_k[..., u], Ei_k[..., v], mode="same")

            qmEiEj[u, v] = normalization * conv

    return qmEiEj


def get_signal_RT_focal(Exy, Ei_fields, Ej_fields, KxKyChi, Chi, coherent_exc=False):
    """
    Compute Brillouin light scattering (BLS) spectrum using the
    reciprocity theorem.

    Source paper: https://doi.org/10.1126/sciadv.ady8833

    Parameters
    ----------
    Exy : list[ndarray]
        (m ) XY grid for the electric field.
        List of two 1D arrays `(x, y)` with shapes ``(Nx,)`` and
        ``(Ny,)`` containing the spatial coordinates of the electric
        field grid.
    Ei_fields : list[ndarray]
        (V/m) list of the three spatial components `[Ex, Ey, Ez]` of the
        focal driving electric field E_dr (incident laser). Each
        component must have shape ``(Nx, Ny)``.
    Ej_fields : list[ndarray]
        (V/m) list of the three spatial components `[Ex, Ey, Ez]` of the
        focal virtual electric field E_v (detector side). Each component
        must have shape ``(Nx, Ny)``.
    KxKyChi : list[ndarray]
        (rad/m) list of two 1D arrays `(kx, ky)` with shapes ``(Nkx,)``
        and ``(Nky,)`` containing the reciprocal space coordinates of
        the magneto-optic susceptibility/Bloch function.
    Chi : ndarray
        () dynamic magneto-optic susceptibility tensor. Must have shape
        ``(3, 3, Nf, Nkx, Nky)`` containing the tensor components
        `Chi_ij` for each frequency and k-space grid point.
    coherent_exc : bool, optional
        If True, calculates the coherent BLS signal (amplitudes sum
        first).  If False (default), calculates the non-coherent/thermal
        BLS signal (intensities sum first).

    Returns
    -------
    sigma : ndarray
        () calculated BLS spectrum.  1D array with shape ``(Nf,)``.
    qmEiEj : ndarray
        () transfer function of the system.  Array with shape
        ``(3, 3, Nkx, Nky)``.  Uses the same coordinates as the
        susceptibility tensor (`KxKyChi`).

    See also
    --------
    get_signal_RT_pupil, get_signal_GF_focal

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
        tmp = np.einsum("uvxy,uvfxy->f", qmEiEj, Chi)
        sigma = np.abs(tmp * dK) ** 2

    else:
        # Thermal sum: Sum_k( | Sum_uv( qm[u,v,k] * Chi[u,v,f,k] ) |^2 )
        # einsum sums over u,v, leaving 'f,x,y'
        tmp = np.einsum("uvxy,uvfxy->fxy", qmEiEj, Chi)

        # Sum over k-space (x and y axes)
        sigma = np.sum(np.abs(tmp) ** 2, axis=(1, 2)) * dK

    return sigma, qmEiEj


def get_signal_GF_focal(
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
    coherent_exc=False,
    output_analyzer="none",
    output_analyzer_angle_deg=0,
    output_analyzer_axis_ratio=1.0,
    full_output=False,
):
    """
    Compute Brillouin light scattering (BLS) spectrum using the
    Green function formalism.

    .. warning::

       This is an experimental function. Syntax and behavior may change
       in future releases. Please verify the results carefully.

    Source paper: https://doi.org/10.1103/PhysRevB.110.224428

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
        (m ) waist of the Gaussian collection spot in the sample plane,
        i.e. the filter is ``h = exp(-(x**2 + y**2)/collectionSpot**2)``
        in amplitude (``1/e**2`` radius in intensity).  Default is 1e-6.
    focalLength : float, optional
        (m ) focal length of the lens.  Default is 1e-3.
    coherent_exc : bool, optional
        If True, calculates the coherent BLS signal (amplitudes sum
        first).  If False (default), calculates the non-coherent/thermal
        BLS signal (intensities sum first).
    output_analyzer : {"none", "linear", "rcp", "lcp", "elliptical", \
            "radial", "azimuthal"}, array_like or callable, optional
        Output polarization analyzer applied in real space before the
        detector.  The polarization types are the same as in
        :func:`~SpinWaveToolkit.bls.polarization.jones_vector` and
        :meth:`~SpinWaveToolkit.bls.ObjectiveLens.getPupilField`, and
        the analyzer transmits the given polarization state.

        - ``"none"`` (default): no analyzer (keeps both Ex and Ey).
        - ``"linear"``: linear analyzer at `output_analyzer_angle_deg`.
        - ``"rcp"``: right-hand circular analyzer.
        - ``"lcp"``: left-hand circular analyzer.
        - ``"elliptical"``: elliptical analyzer with major axis at
          `output_analyzer_angle_deg` and axis ratio
          `output_analyzer_axis_ratio`.
        - ``"radial"``: spatially varying radial analyzer.
        - ``"azimuthal"``: spatially varying azimuthal analyzer.

        If an array is provided, it is the Jones vector with shape
        ``(2,)``, or the Jones field with shape ``(2, 2*Nq-1, 2*Nq-1)``
        defined on the real-space grid (see `x_scat`, `y_scat` in
        Returns), of the polarization transmitted by the analyzer.  The
        detected field is then ``conj(e[0])*Ex + conj(e[1])*Ey``.  Such
        arrays can be prepared using the
        :mod:`~SpinWaveToolkit.bls.polarization` module.  Note that
        optics with Jones matrix ``M`` followed by a polarizer
        transmitting ``e_p`` is equivalent to an analyzer transmitting
        ``e = M^H e_p`` (``M^H`` is the conjugate transpose of ``M``).
        The array is not normalized, i.e. it can also be used for
        amplitude masking.

        If a callable is provided, it must have signature
        ``f(x_scat, y_scat) -> (ax, ay)`` and return analyzer
        coefficients broadcastable to the shape of ``x_scat`` and
        ``y_scat`` (real space meshgrids - see Returns section).  The
        detected field is then ``ax*Ex + ay*Ey``.
    output_analyzer_angle_deg : float, optional
        (deg) angle of the "linear" output analyzer or of the major axis
        of the "elliptical" one (counter-clockwise from x).  Ignored for
        other analyzer types.  Default is 0.
    output_analyzer_axis_ratio : float, optional
        () ratio of the minor axis to the major axis of the "elliptical"
        output analyzer, its sign sets the handedness (see
        :func:`~SpinWaveToolkit.bls.polarization.jones_vector`).
        Ignored for other analyzer types.  Default is 1.0.
    full_output : bool, optional
        If True, returns additional intermediate results: polarizations
        with q-space grids and scattered electric field with real-space
        grids).  Default is False.

    Returns
    -------
    sigma : ndarray
        () calculated BLS spectrum.  1D real array with shape ``(Nf,)``.
    Px, Py, Pz : ndarray
        (V/m) induced polarization in the magnetic layer.  Corresponds
        to `P` in eq. (3) in Wojewoda et al. PRB 110, 224428 (2024).
        Each array has shape ``(Nf, 2*Nq-1, 2*Nq-1)``.
    Qx, Qy : ndarray
        (rad/m) k-space grids for polarizations `Px`, `Py`, `Pz`.
        Each array has shape ``(2*Nq-1, 2*Nq-1)``.
    Ex_scat, Ey_scat : ndarray
        (V/m) scattered real-space electric field components before the
        analyzer.  Each array has shape ``(Nf, 2*Nq-1, 2*Nq-1)``.
        Can be used for custom analyzer calculations and beam masking.
    x_scat, y_scat : ndarray
        (m ) real-space grids for the scattered electric field.
        Each array has shape ``(2*Nq-1, 2*Nq-1)``.

    See also
    --------
    get_signal_RT_focal, get_signal_RT_pupil

    Notes
    -----
    - The radiating polarization sheet is placed at the interface of
      the source layer with the layer above it (towards the
      superstrate).  The attenuation of light inside the source layer
      is accounted for by the volume factor in eq. (32) of the source
      paper.
    - The induced polarization is calculated only for the linear
      magneto-optic (Voigt) coupling with ``Q = 1`` (see
      :func:`~SpinWaveToolkit.bls.susceptibilities.mo_linear`), the
      quadratic effects are neglected.
    - The Fourier transform of `E` and the convolution with the Bloch
      functions are normalized as their continuous counterparts, so
      the signal does not depend on the sampling of `E` or on `Nq`
      (provided they are fine enough).  Its absolute scale is still
      given by the (arbitrary) normalization of `E` and `Bloch`.
    - The convolutions of the electric field with the Bloch functions
      are evaluated using the convolution theorem (zero-padded FFTs),
      which gives the same result as direct 2D convolutions, but is
      much faster for fine q-grids.

    """
    warn(
        "`get_signal_GF_focal` is an experimental function and may be subject to change."
        + " Please verify results carefully.",
        UserWarning,
        stacklevel=2,
    )

    k0 = 2 * np.pi / wavelength
    Nf = len(SweepBloch)

    # --- Set up q-space grid (qx and qy) ---
    qxHalf = np.linspace(0, 1.1, Nq) * k0
    qx = np.concatenate((-qxHalf[1:][::-1], qxHalf))

    # qy is taken identical to qx
    qy = qx.copy()
    Nqg = len(qx)  # = 2*Nq - 1
    dkx = qx[1] - qx[0]
    dky = qy[1] - qy[0]

    # Create the 2D grid using ndgrid convention (like Matlab)
    Qx, Qy = np.meshgrid(qx, qy, indexing="ij")
    Q = np.sqrt(Qx**2 + Qy**2)
    # Use complex square root to avoid NaNs for negative arguments
    Kzs = np.sqrt(DF[source_layer_index] * k0**2 - Q**2 + 0j)
    Kz = np.sqrt(k0**2 - Q**2 + 0j)
    # Prepare the target points as an (M,2) array where M = number of Qx points
    points = np.stack([Qx.ravel(), Qy.ravel()], axis=-1)
    # -------------------------------------------------------------

    # --- Suppose Exy is given as a tuple of 1D arrays (x, y) for the spatial coordinates ---
    EX, EY = Exy
    # Determine grid spacings (assuming uniform spacing)
    dx = EX[1] - EX[0]
    dy = EY[1] - EY[0]

    # --- Compute the Fourier transform of the electric field components ---
    # We assume E has shape (3, Ny, Nx) where E[0] is the X component, etc.
    # The factor dx*dy approximates the continuous Fourier transform
    # E(k) = int E(r) exp(-i k.r) d^2r, making it independent of the grid.
    # Apply ifftshift in both axes, then fft2, then fftshift back.
    fftEI = fftshift(
        spfft.fft2(ifftshift(np.asarray(E), axes=(-2, -1)), axes=(-2, -1)),
        axes=(-2, -1),
    ) * (dx * dy)
    # -------------------------------------------------------------

    # --- Compute the Fourier domain grid corresponding to the spatial grid ---
    # The FFT frequency bins (in radians per meter) are given by:
    kx_fft = fftshift(2 * np.pi * np.fft.fftfreq(len(EX), d=dx))
    ky_fft = fftshift(2 * np.pi * np.fft.fftfreq(len(EY), d=dy))

    # --- Interpolate the computed FFT of the E-field onto the Qx, Qy grid ---
    # Here fftEI has shape (3, Ny, Nx), i.e. it is defined on (KY_fft, KX_fft)
    # (all three components are interpolated at once)
    interp_func = RegularGridInterpolator(
        (ky_fft, kx_fft),
        np.moveaxis(fftEI, 0, -1),
        bounds_error=False,
        fill_value=0,
    )
    interp_fftEI = np.moveaxis(interp_func(points[:, ::-1]), -1, 0).reshape(3, Nqg, Nqg)

    # Compute a volume factor by integrating an exponential decay over the layer
    # thickness (both incident and scattered fields are attenuated), eq. (32)
    zs = np.linspace(0, d[source_layer_index - 1], 100)
    ExtinCoefMagLayer = np.sqrt(
        (abs(DF[source_layer_index]) - np.real(DF[source_layer_index])) / 2
    )
    Volume = np.exp(-2 * ExtinCoefMagLayer * k0 * zs)
    VolumeFac = trapezoid(Volume, zs)

    # Multiply the electric field by the volume factor and by the q-space
    # measure dqx*dqy/(2*pi)**2, so that the discrete convolutions below
    # approximate the continuous ones, eq. (18)
    interp_fftEI *= VolumeFac * dkx * dky / (2 * np.pi) ** 2

    # --- Prepare the convolutions using the convolution theorem ---
    # FFTs zero-padded to at least 2*Nqg-1 points give the full linear
    # convolution, from which the central (Nqg, Nqg) part is taken, i.e.
    # the same result as convolve2d(..., mode="same").
    fast_M = spfft.next_fast_len(2 * Nqg - 1)
    start = (Nqg - 1) // 2
    fftEI_conv = spfft.fft2(interp_fftEI, s=(fast_M, fast_M), axes=(-2, -1))
    # -------------------------------------------------------------

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

    # --- Calculate the p- and s-polarized electric field contributions ---
    # pGF and sGF are assumed to be 3×2 structures (lists of lists or similar).
    # The terms multiplying each polarization component do not depend on
    # frequency, so they are evaluated here: Ep = sum_c P_c * pTerms[c], etc.
    expMinus = np.exp(-1j * Kzs * d[source_layer_index - 1])
    expPlus = np.exp(1j * Kzs * d[source_layer_index - 1])
    pTerms = np.array([pGF[c][0] * expMinus + pGF[c][1] * expPlus for c in range(3)])
    sTerms = np.array([sGF[c][0] * expMinus + sGF[c][1] * expPlus for c in range(3)])

    # --- Convert to X and Y components in the laboratory frame ---
    # Avoid division by zero: when Q==0 set cosPhi=1 and sinPhi=0.
    cosPhi = np.divide(Qx, Q, out=np.ones_like(Qx), where=Q != 0)
    sinPhi = np.divide(Qy, Q, out=np.zeros_like(Qy), where=Q != 0)

    # --- Apply a polarization-dependent factor ---
    Factor = (
        (-2j * np.pi * np.sqrt(Kz * k0)) * np.exp(1j * k0 * focalLength) / focalLength
    )
    # Create a mask for Q values within k0*NA.
    mask = (Q <= k0 * NA).astype(float)
    # Ex_field = sum_c P_c * xTerms[c], Ey_field = sum_c P_c * yTerms[c]
    xTerms = (pTerms * cosPhi - sTerms * sinPhi) * Factor * mask
    yTerms = (pTerms * sinPhi + sTerms * cosPhi) * Factor * mask
    # -------------------------------------------------------------

    # --- Compute real-space grids and apply the point-spread filter ---
    # This represent limited ability to propagate the electric field to the detector.
    DXi = 2 * np.pi / dkx
    DYi = 2 * np.pi / dky
    dxi = DXi / Nqg
    dyi = DYi / Nqg
    xi = np.linspace(-(Nqg - 1) / 2, (Nqg - 1) / 2, Nqg) * dxi
    yi = np.linspace(-(Nqg - 1) / 2, (Nqg - 1) / 2, Nqg) * dyi
    Xi, Yi = np.meshgrid(xi, yi, indexing="ij")
    PSFFilter = np.exp(-(Xi**2 + Yi**2) / collectionSpot**2)
    # Compute a common scaling factor (note: np.size returns the total number of elements).
    factor_fft = (focalLength / k0) ** 2 * Xi.size / (4 * np.pi**2) * dkx * dky
    # -------------------------------------------------------------
    # --- Pre-compute analyzer coefficients on the real-space grid ---
    ax, ay = _analyzer_coefficients(
        output_analyzer, output_analyzer_angle_deg, output_analyzer_axis_ratio, Xi, Yi
    )
    # -------------------------------------------------------------

    # --- Prepare for frequency loop ---
    # Get the kx, ky grid for Bloch functions (assumed to be 1D arrays)
    kx_grid, ky_grid = KxKyBloch
    Bloch = np.asarray(Bloch)

    # Prepare arrays to store the results for each frequency
    sigma = np.zeros(Nf)
    if full_output:  # Preallocate polarization and scattered field.
        Px = np.empty((Nf, Nqg, Nqg), dtype=complex)
        Py = np.empty((Nf, Nqg, Nqg), dtype=complex)
        Pz = np.empty((Nf, Nqg, Nqg), dtype=complex)
        Ex_scat = np.empty((Nf, Nqg, Nqg), dtype=complex)
        Ey_scat = np.empty((Nf, Nqg, Nqg), dtype=complex)
    # Loop over frequencies in the Bloch function.
    # Here we assume that the first dimension of Bloch (after the component index)
    # corresponds to the sweep index (and that len(SweepBloch)==Nf).
    for i in range(Nf):
        # --- Interpolate Bloch function components onto the Qx-Qy grid ---
        # We assume Bloch has shape (3, Nf, Nkx, Nky)
        # (all three components are interpolated at once)
        interp_M = RegularGridInterpolator(
            (kx_grid, ky_grid),
            np.moveaxis(Bloch[:, i], 0, -1),
            bounds_error=False,
            fill_value=0,
        )
        Bloch_interp = np.moveaxis(interp_M(points), -1, 0).reshape(3, Nqg, Nqg)
        # -------------------------------------------------------------

        # --- Convolve the electric field with the (interpolated) Bloch components ---
        # P = i (E x M), i.e. Px = conv(Ez, i*My) + conv(Ey, -i*Mz), etc.
        fftM = spfft.fft2(Bloch_interp, s=(fast_M, fast_M), axes=(-2, -1))
        P_conv = 1j * np.array(
            [
                fftEI_conv[2] * fftM[1] - fftEI_conv[1] * fftM[2],
                fftEI_conv[0] * fftM[2] - fftEI_conv[2] * fftM[0],
                fftEI_conv[1] * fftM[0] - fftEI_conv[0] * fftM[1],
            ]
        )
        P_i = spfft.ifft2(P_conv, axes=(-2, -1))[
            :, start : start + Nqg, start : start + Nqg
        ]
        # -------------------------------------------------------------
        if full_output:  # save for output if requested
            Px[i], Py[i], Pz[i] = P_i

        # --- Transform back to real space with an applied numerical aperture mask ---
        # (the mask is included in xTerms and yTerms)
        Ex_real = factor_fft * fftshift(
            spfft.ifft2(ifftshift(np.sum(P_i * xTerms, axis=0)))
        )
        Ey_real = factor_fft * fftshift(
            spfft.ifft2(ifftshift(np.sum(P_i * yTerms, axis=0)))
        )
        # Apply the point-spread (PSF) filter in real space.
        Ex_real *= PSFFilter
        Ey_real *= PSFFilter
        # -------------------------------------------------------------
        if full_output:  # save for output if requested
            Ex_scat[i], Ey_scat[i] = Ex_real, Ey_real

        # --- Apply the output analyzer filtering ---
        # Analyzer projection: E_det = ax * Ex + ay * Ey.
        if ax is not None:
            E_det = ax * Ex_real + ay * Ey_real
            Ex_real = E_det
            Ey_real = np.zeros_like(E_det)
        # -------------------------------------------

        # --- Compute signal integrals over the image (spatial integration on the detector) ---
        if coherent_exc:
            ExS = dxi * dyi * np.sum(Ex_real)
            EyS = dxi * dyi * np.sum(Ey_real)
            sigma[i] = ExS.real**2 + ExS.imag**2 + EyS.real**2 + EyS.imag**2
        else:
            sigma[i] = (
                dxi
                * dyi
                * np.sum(
                    Ex_real.real**2
                    + Ex_real.imag**2
                    + Ey_real.real**2
                    + Ey_real.imag**2
                )
            )

    # Return the computed scattering cross-section (1D array over sweep)
    # and optionally other intermediate results for further analysis or custom processing.
    if full_output:
        return sigma, Px, Py, Pz, Qx, Qy, Ex_scat, Ey_scat, Xi, Yi
    else:
        return sigma


def get_signal_GF_pupil(
    KxKy,
    Ei_fields,
    Chi,
    DF,
    PM,
    d,
    NA,
    source_layer_index=1,
    output_layer_index=0,
    wavelength=532e-9,
    collectionSpot=1e-6,
    focalLength=1e-3,
    coherent_exc=False,
    output_analyzer="none",
    output_analyzer_angle_deg=0,
    output_analyzer_axis_ratio=1.0,
    full_output=False,
):
    """
    Compute Brillouin light scattering (BLS) spectrum using the
    Green function formalism, starting directly from the electric field
    in reciprocal (k) space.

    The incident field and the magneto-optic susceptibility are given on
    the same k-space grid, so no interpolation is needed and the
    calculation is faster than :func:`get_signal_GF_focal`.

    .. warning::

       This is an experimental function. Syntax and behavior may change
       in future releases. Please verify the results carefully.

    .. important::

       To maintain a valid physical representation of the convolution
       integral, the input k-space grid (`KxKy`) MUST be strictly
       equidistant and symmetric with respect to ``k = 0``.

    Source paper: https://doi.org/10.1103/PhysRevB.110.224428

    Parameters
    ----------
    KxKy : list[ndarray]
        (rad/m) list of two 1D arrays `(kx, ky)` with shapes ``(Nkx,)``
        and ``(Nky,)`` containing the reciprocal space coordinates.
        Must be a uniform/equidistant grid symmetric with respect to
        ``k = 0``, preferably with an odd number of points (see Notes).
    Ei_fields : list[ndarray]
        (V/m) list of the three reciprocal pupil field components
        `[Ekx, Eky, Ekz]` corresponding to the driving field E_dr
        (incident laser), as given by
        :meth:`~SpinWaveToolkit.bls.ObjectiveLens.getPupilField`.  Each
        must have shape ``(Nkx, Nky)``.
    Chi : ndarray
        () dynamic magneto-optic susceptibility tensor with shape
        ``(3, 3, Nf, Nkx, Nky)``, containing the tensor components
        `Chi_ij` for each frequency and k-space grid point, e.g. from
        :mod:`~SpinWaveToolkit.bls.susceptibilities`.
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
        Numerical aperture of the collecting optical system.
    source_layer_index : int, optional
        Index of the source layer in the stack.  Default is 1.
    output_layer_index : int, optional
        Index of the output layer in the stack.  Default is 0.
    wavelength : float, optional
        (m ) wavelength of the light.  Default is 532e-9.
    collectionSpot : float, optional
        (m ) waist of the Gaussian collection spot in the sample plane,
        i.e. the filter is ``h = exp(-(x**2 + y**2)/collectionSpot**2)``
        in amplitude (``1/e**2`` radius in intensity).  Default is 1e-6.
    focalLength : float, optional
        (m ) focal length of the lens.  Default is 1e-3.
    coherent_exc : bool, optional
        If True, calculates the coherent BLS signal (amplitudes sum
        first).  If False (default), calculates the non-coherent/thermal
        BLS signal (intensities sum first).
    output_analyzer : {"none", "linear", "rcp", "lcp", "elliptical", \
            "radial", "azimuthal"}, array_like or callable, optional
        Output polarization analyzer applied in real space before the
        detector.  The polarization types are the same as in
        :func:`~SpinWaveToolkit.bls.polarization.jones_vector` and
        :meth:`~SpinWaveToolkit.bls.ObjectiveLens.getPupilField`, and
        the analyzer transmits the given polarization state.

        - ``"none"`` (default): no analyzer (keeps both Ex and Ey).
        - ``"linear"``: linear analyzer at `output_analyzer_angle_deg`.
        - ``"rcp"``: right-hand circular analyzer.
        - ``"lcp"``: left-hand circular analyzer.
        - ``"elliptical"``: elliptical analyzer with major axis at
          `output_analyzer_angle_deg` and axis ratio
          `output_analyzer_axis_ratio`.
        - ``"radial"``: spatially varying radial analyzer.
        - ``"azimuthal"``: spatially varying azimuthal analyzer.

        If an array is provided, it is the Jones vector with shape
        ``(2,)``, or the Jones field with shape ``(2, Nkx, Nky)``
        defined on the real-space grid (see `x_scat`, `y_scat` in
        Returns), of the polarization transmitted by the analyzer.  The
        detected field is then ``conj(e[0])*Ex + conj(e[1])*Ey``.  Such
        arrays can be prepared using the
        :mod:`~SpinWaveToolkit.bls.polarization` module.  Note that
        optics with Jones matrix ``M`` followed by a polarizer
        transmitting ``e_p`` is equivalent to an analyzer transmitting
        ``e = M^H e_p`` (``M^H`` is the conjugate transpose of ``M``).
        The array is not normalized, i.e. it can also be used for
        amplitude masking.

        If a callable is provided, it must have signature
        ``f(x_scat, y_scat) -> (ax, ay)`` and return analyzer
        coefficients broadcastable to the shape of ``x_scat`` and
        ``y_scat`` (real space meshgrids - see Returns section).  The
        detected field is then ``ax*Ex + ay*Ey``.
    output_analyzer_angle_deg : float, optional
        (deg) angle of the "linear" output analyzer or of the major axis
        of the "elliptical" one (counter-clockwise from x).  Ignored for
        other analyzer types.  Default is 0.
    output_analyzer_axis_ratio : float, optional
        () ratio of the minor axis to the major axis of the "elliptical"
        output analyzer, its sign sets the handedness (see
        :func:`~SpinWaveToolkit.bls.polarization.jones_vector`).
        Ignored for other analyzer types.  Default is 1.0.
    full_output : bool, optional
        If True, returns additional intermediate results: polarizations
        with q-space grids and scattered electric field with real-space
        grids).  Default is False.

    Returns
    -------
    sigma : ndarray
        () calculated BLS spectrum.  1D real array with shape ``(Nf,)``.
    Px, Py, Pz : ndarray
        (V/m) induced polarization in the magnetic layer.  Corresponds
        to `P` in eq. (3) in Wojewoda et al. PRB 110, 224428 (2024).
        Each array has shape ``(Nf, Nkx, Nky)``.
    Qx, Qy : ndarray
        (rad/m) k-space grids (meshgrids of `KxKy`) for polarizations
        `Px`, `Py`, `Pz`.  Each array has shape ``(Nkx, Nky)``.
    Ex_scat, Ey_scat : ndarray
        (V/m) scattered real-space electric field components before the
        analyzer.  Each array has shape ``(Nf, Nkx, Nky)``.
        Can be used for custom analyzer calculations and beam masking.
    x_scat, y_scat : ndarray
        (m ) real-space grids for the scattered electric field.
        Each array has shape ``(Nkx, Nky)``.

    See also
    --------
    get_signal_GF_focal, get_signal_RT_pupil

    Notes
    -----
    - The pupil fields of
      :meth:`~SpinWaveToolkit.bls.ObjectiveLens.getPupilField` follow
      the angular spectrum representation
      ``E(r) = int E_k(k) exp(i k.r) d^2k``.  They are multiplied by
      ``(2*pi)**2`` to obtain the Fourier transform
      ``E(k) = int E(r) exp(-i k.r) d^2r`` used in
      :func:`get_signal_GF_focal`, so both functions give the same
      signal for the same objective lens.
    - The induced polarization is ``P = Chi . E``, i.e.
      ``P_u = sum_v Chi_uv E_v`` (convolution in k-space), eq. (18).
    - Light scattered by magnons with wavevectors up to ``2*k0*NA``
      can reach the detector.  Therefore, a note is issued if the
      k-grid limit is smaller than this value (contributions of the
      magnons outside the grid are neglected) or more than 10 times
      larger (most of the grid does not contribute to the signal).  A
      warning is issued if the incident field is not negligible at the
      boundary of the grid, i.e. if it is truncated.
    - If ``k = 0`` is not a grid point (even number of points), the
      convolution is shifted by half of the grid step.  This is
      negligible for dense grids, but a note is issued.
    - The radiating polarization sheet is placed at the interface of
      the source layer with the layer above it (towards the
      superstrate).  The attenuation of light inside the source layer
      is accounted for by the volume factor in eq. (32) of the source
      paper.
    - The convolution of the electric field with the susceptibility is
      normalized as its continuous counterpart, so the signal does not
      depend on the sampling of the k-grid (provided it is fine
      enough).  Its absolute scale is still given by the (arbitrary)
      normalization of `Ei_fields` and `Chi`.

    """
    warn(
        "`get_signal_GF_pupil` is an experimental function and may be subject to change."
        + " Please verify results carefully.",
        UserWarning,
        stacklevel=2,
    )

    k0 = 2 * np.pi / wavelength

    # --- K-space coordinates (ndgrid convention, like Matlab) ---
    kx, ky = KxKy
    kx, ky = np.asarray(kx), np.asarray(ky)
    Nkx, Nky = len(kx), len(ky)
    dkx, dky = _check_pupil_grid(kx, ky)
    Qx, Qy = np.meshgrid(kx, ky, indexing="ij")
    Q = np.sqrt(Qx**2 + Qy**2)
    # Use complex square root to avoid NaNs for negative arguments
    Kzs = np.sqrt(DF[source_layer_index] * k0**2 - Q**2 + 0j)
    Kz = np.sqrt(k0**2 - Q**2 + 0j)
    # -------------------------------------------------------------

    # --- Check the extent of the k-grid ---
    E_k = np.stack(Ei_fields)  # Shape (3, Nkx, Nky)
    absE = np.abs(E_k).max(axis=0)
    edge = max(absE[0].max(), absE[-1].max(), absE[:, 0].max(), absE[:, -1].max())
    if edge > 1e-3 * absE.max():
        warn(
            "The incident field is not negligible at the boundary of the k-grid "
            + f"({edge / absE.max():.1e} of its maximum), i.e. it is truncated. "
            + "Increase the k-grid limit.",
            UserWarning,
            stacklevel=2,
        )
    k_lim = min(kx[-1], ky[-1])
    if k_lim < 2 * k0 * NA:
        warn(
            f"Note: the k-grid limit ({k_lim:.3g} rad/m) is smaller than 2*k0*NA "
            + f"({2 * k0 * NA:.3g} rad/m).  Contributions of magnons with larger "
            + "wavevectors, which could still scatter light into the NA, are neglected.",
            UserWarning,
            stacklevel=2,
        )
    elif k_lim > 10 * 2 * k0 * NA:
        warn(
            f"Note: the k-grid limit ({k_lim:.3g} rad/m) is more than 10 times larger "
            + f"than 2*k0*NA ({2 * k0 * NA:.3g} rad/m), so most of the grid does not "
            + "contribute to the signal.  A smaller limit gives a finer resolution for "
            + "the same number of points.",
            UserWarning,
            stacklevel=2,
        )

    Chi = np.asarray(Chi)
    if Chi.ndim != 5 or Chi.shape[:2] != (3, 3) or Chi.shape[3:] != (Nkx, Nky):
        raise ValueError(
            f"Chi must have shape (3, 3, Nf, {Nkx}, {Nky}), got {Chi.shape}."
        )
    Nf = Chi.shape[2]
    # Skip components where susceptibility is zero
    chi_mask = np.any(Chi, axis=(2, 3, 4))

    # --- Fourier transform of the electric field from the pupil field ---
    # The pupil field follows E(r) = int E_k(k) exp(i k.r) d^2k, while the
    # Fourier transform is E(k) = int E(r) exp(-i k.r) d^2r = (2*pi)**2 E_k(k)
    E_k = E_k * (2 * np.pi) ** 2

    # Compute a volume factor by integrating an exponential decay over the layer
    # thickness (both incident and scattered fields are attenuated), eq. (32)
    zs = np.linspace(0, d[source_layer_index - 1], 100)
    ExtinCoefMagLayer = np.sqrt(
        (abs(DF[source_layer_index]) - np.real(DF[source_layer_index])) / 2
    )
    Volume = np.exp(-2 * ExtinCoefMagLayer * k0 * zs)
    VolumeFac = trapezoid(Volume, zs)

    # Multiply the electric field by the volume factor and by the q-space
    # measure dqx*dqy/(2*pi)**2, so that the discrete convolutions below
    # approximate the continuous ones, eq. (18)
    E_k *= VolumeFac * dkx * dky / (2 * np.pi) ** 2

    # --- Evaluate Fresnel coefficients and spherical Green functions ---
    # The function returns two objects (htp and hts) that can be evaluated on Q.
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

    # --- Calculate the p- and s-polarized electric field contributions ---
    # pGF and sGF are assumed to be 3×2 structures (lists of lists or similar).
    # The terms multiplying each polarization component do not depend on
    # frequency, so they are evaluated here: Ep = sum_c P_c * pTerms[c], etc.
    expMinus = np.exp(-1j * Kzs * d[source_layer_index - 1])
    expPlus = np.exp(1j * Kzs * d[source_layer_index - 1])
    pTerms = np.array([pGF[c][0] * expMinus + pGF[c][1] * expPlus for c in range(3)])
    sTerms = np.array([sGF[c][0] * expMinus + sGF[c][1] * expPlus for c in range(3)])

    # --- Convert to X and Y components in the laboratory frame ---
    # Avoid division by zero: when Q==0 set cosPhi=1 and sinPhi=0.
    cosPhi = np.divide(Qx, Q, out=np.ones_like(Qx), where=Q != 0)
    sinPhi = np.divide(Qy, Q, out=np.zeros_like(Qy), where=Q != 0)

    # --- Apply a polarization-dependent factor ---
    Factor = (
        (-2j * np.pi * np.sqrt(Kz * k0)) * np.exp(1j * k0 * focalLength) / focalLength
    )
    # Create a mask for Q values within k0*NA.
    mask = (Q <= k0 * NA).astype(float)
    # Ex_field = sum_c P_c * xTerms[c], Ey_field = sum_c P_c * yTerms[c]
    xTerms = (pTerms * cosPhi - sTerms * sinPhi) * Factor * mask
    yTerms = (pTerms * sinPhi + sTerms * cosPhi) * Factor * mask
    # -------------------------------------------------------------

    # --- Compute real-space grids and apply the point-spread filter ---
    # This represent limited ability to propagate the electric field to the detector.
    DXi = 2 * np.pi / dkx
    DYi = 2 * np.pi / dky
    dxi = DXi / Nkx
    dyi = DYi / Nky
    xi = np.linspace(-(Nkx - 1) / 2, (Nkx - 1) / 2, Nkx) * dxi
    yi = np.linspace(-(Nky - 1) / 2, (Nky - 1) / 2, Nky) * dyi
    Xi, Yi = np.meshgrid(xi, yi, indexing="ij")
    PSFFilter = np.exp(-(Xi**2 + Yi**2) / collectionSpot**2)
    # Compute a common scaling factor (note: np.size returns the total number of elements).
    factor_fft = (focalLength / k0) ** 2 * Xi.size / (4 * np.pi**2) * dkx * dky
    # -------------------------------------------------------------
    # --- Pre-compute analyzer coefficients on the real-space grid ---
    ax, ay = _analyzer_coefficients(
        output_analyzer, output_analyzer_angle_deg, output_analyzer_axis_ratio, Xi, Yi
    )
    # -------------------------------------------------------------

    # Prepare arrays to store the results for each frequency
    sigma = np.zeros(Nf)
    if full_output:  # Preallocate polarization and scattered field.
        Px = np.empty((Nf, Nkx, Nky), dtype=complex)
        Py = np.empty((Nf, Nkx, Nky), dtype=complex)
        Pz = np.empty((Nf, Nkx, Nky), dtype=complex)
        Ex_scat = np.empty((Nf, Nkx, Nky), dtype=complex)
        Ey_scat = np.empty((Nf, Nkx, Nky), dtype=complex)
    # Loop over frequencies in the susceptibility tensor.
    for i in range(Nf):
        # --- Convolve the electric field with the susceptibility ---
        # P_u = sum_v Chi_uv * E_v (convolution in k-space)
        P_i = np.zeros((3, Nkx, Nky), dtype=complex)
        for u in range(3):
            for v in range(3):
                if chi_mask[u, v]:
                    P_i[u] += fftconvolve(Chi[u, v, i], E_k[v], mode="same")
        # -------------------------------------------------------------
        if full_output:  # save for output if requested
            Px[i], Py[i], Pz[i] = P_i

        # --- Transform back to real space with an applied numerical aperture mask ---
        # (the mask is included in xTerms and yTerms)
        Ex_real = factor_fft * fftshift(
            spfft.ifft2(ifftshift(np.sum(P_i * xTerms, axis=0)))
        )
        Ey_real = factor_fft * fftshift(
            spfft.ifft2(ifftshift(np.sum(P_i * yTerms, axis=0)))
        )
        # Apply the point-spread (PSF) filter in real space.
        Ex_real *= PSFFilter
        Ey_real *= PSFFilter
        # -------------------------------------------------------------
        if full_output:  # save for output if requested
            Ex_scat[i], Ey_scat[i] = Ex_real, Ey_real

        # --- Apply the output analyzer filtering ---
        # Analyzer projection: E_det = ax * Ex + ay * Ey.
        if ax is not None:
            E_det = ax * Ex_real + ay * Ey_real
            Ex_real = E_det
            Ey_real = np.zeros_like(E_det)
        # -------------------------------------------

        # --- Compute signal integrals over the image (spatial integration on the detector) ---
        if coherent_exc:
            ExS = dxi * dyi * np.sum(Ex_real)
            EyS = dxi * dyi * np.sum(Ey_real)
            sigma[i] = ExS.real**2 + ExS.imag**2 + EyS.real**2 + EyS.imag**2
        else:
            sigma[i] = (
                dxi
                * dyi
                * np.sum(
                    Ex_real.real**2
                    + Ex_real.imag**2
                    + Ey_real.real**2
                    + Ey_real.imag**2
                )
            )

    # Return the computed scattering cross-section (1D array over sweep)
    # and optionally other intermediate results for further analysis or custom processing.
    if full_output:
        return sigma, Px, Py, Pz, Qx, Qy, Ex_scat, Ey_scat, Xi, Yi
    else:
        return sigma


def _check_pupil_grid(kx, ky):
    """
    Check the reciprocal-space grid used by the ``..._pupil`` functions.

    The discrete convolutions (``mode="same"``) represent the continuous
    ones only on an equidistant grid symmetric with respect to
    ``k = 0``, otherwise a ValueError is raised.  If ``k = 0`` is not a
    grid point (even number of points), the result of the convolution
    is shifted by half of the grid step and a note is issued.

    Returns
    -------
    dkx, dky : float
        (rad/m) grid steps (1.0 for grids with a single point).
    """
    steps = []
    for name, k in (("kx", np.asarray(kx)), ("ky", np.asarray(ky))):
        dk = k[1] - k[0] if len(k) > 1 else 1.0
        # (Required by FFT, and physically required by direct discrete convolution)
        if len(k) > 1 and not np.allclose(np.diff(k), dk):
            raise ValueError(
                f"The {name} grid must be strictly equidistant for valid discrete convolution."
            )
        if not np.isclose(k[0], -k[-1], rtol=0, atol=1e-3 * abs(dk)):
            raise ValueError(
                f"The {name} grid must be symmetric with respect to k = 0 (limits with the same "
                + f"magnitude) for valid discrete convolution, got limits "
                + f"{k[0]:.4g} and {k[-1]:.4g} rad/m."
            )
        if len(k) % 2 == 0:
            warn(
                f"Note: the {name} grid has an even number of points, i.e. k = 0 is not a grid "
                + "point, which shifts the convolution by half of the grid step (negligible "
                + "for dense grids).  Use an odd number of points to avoid it.",
                UserWarning,
                stacklevel=3,
            )
        steps.append(dk)
    return steps


def _analyzer_coefficients(output_analyzer, angle, axis_ratio, Xi, Yi):
    """
    Projection coefficients ``(ax, ay)`` of the output analyzer such
    that the detected field is ``ax*Ex + ay*Ey``.

    See the `output_analyzer` parameter of :func:`get_signal_GF_focal`.
    Returns ``(None, None)`` if no analyzer is used.
    """
    if callable(output_analyzer):
        ax, ay = output_analyzer(Xi, Yi)
        return np.asarray(ax), np.asarray(ay)
    if isinstance(output_analyzer, str):
        if output_analyzer == "none":
            return None, None
        try:
            e = jones_vector(output_analyzer, angle, axis_ratio, X=Xi, Y=Yi)
        except ValueError as err:
            raise ValueError(
                f"Invalid output_analyzer '{output_analyzer}'. Expected 'none', "
                "a polarization type of `polarization.jones_vector` ('linear', "
                "'rcp', 'lcp', 'elliptical', 'radial', 'azimuthal'), a Jones "
                "vector/field, or a callable f(Xi, Yi)->(ax, ay)."
            ) from err
    else:
        e = np.asarray(output_analyzer)
        if e.shape not in ((2,), (2, *Xi.shape)):
            raise ValueError(
                f"Analyzer Jones vector/field must have shape (2,) or "
                f"{(2, *Xi.shape)}, got {e.shape}."
            )
    # Projection onto the polarization state transmitted by the analyzer
    return np.conj(e[0]), np.conj(e[1])


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
    Compute Brillouin light scattering (BLS) spectrum using the
    Green function formalism.

    .. deprecated:: 1.3

        This function is deprecated and will be removed in
        :mod:`SpinWaveToolkit` v1.5.
        Please use :func:`get_signal_GF_focal` instead.

    Source paper: https://doi.org/10.1103/PhysRevB.110.224428

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

    See also
    --------
    get_signal_GF_focal : Replacement for this deprecated function.
    get_signal_RT_focal, get_signal_RT_pupil
    """
    warn(
        "`getBLSsignal` is deprecated and will be removed in SpinWaveToolkit v1.5."
        + " Please use `get_signal_GF_focal` instead.",
        DeprecationWarning,
        stacklevel=2,
    )

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
    for i, _ in enumerate(SweepBloch):
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
