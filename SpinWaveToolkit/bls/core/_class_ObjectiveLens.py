"""
Core (private) file for the `ObjectiveLens` class.
"""

import numpy as np
from scipy.interpolate import griddata
from scipy.special import jv  # Bessel function of first kind
from scipy.integrate import simpson  # Import simpson for numerical integration
from SpinWaveToolkit.bls.polarization import jones_vector

__all__ = ["ObjectiveLens"]


class ObjectiveLens:
    """
    Represents an objective lens with specific optical parameters.

    Module for calculating the electric field focused by an objective
    lens.  Calculations follow the method presented in book of Novotny
    and Hecht.

    Parameters
    ----------
    wavelength : float
        (m ) wavelength of the light.
    NA : float
        Numerical aperture of the objective lens.
    f0 : float
        Filling factor.
    f : float
        (m ) focal length of the objective lens.

    Attributes
    ----------
    same as Parameters

    Methods
    -------
    getFocalFieldRad
    getFocalFieldAzm
    getFocalField
    getPupilField

    See also
    --------
    get_signal_GF_focal, get_signal_RT_focal, get_signal_RT_pupil :
        Functions that use the electric fields for BLS signal
        calculations.
    SpinWaveToolkit.rotate_field :
        Function for rotating the vectorial electric field in xy plane.

    """

    def __init__(self, wavelength, NA, f0, f):
        self.wavelength = wavelength
        self.NA = NA
        self.f0 = f0
        self.f = f

    def _scattered_interpolant(self, x, y, z, XI, YI):
        """
        Interpolates scattered data ``(x, y, z)`` onto a regular grid
        ``(XI, YI)``.

        Uses linear interpolation with a nearest-neighbor fallback for
        undefined points.
        """
        points = np.column_stack((x, y))
        grid_z_linear = griddata(points, z, (XI, YI), method="linear")
        nan_mask = np.isnan(grid_z_linear)
        if np.any(nan_mask):
            grid_z_nearest = griddata(points, z, (XI, YI), method="nearest")
            grid_z_linear[nan_mask] = grid_z_nearest[nan_mask]
        return grid_z_linear

    def getFocalField(self, z, rho_max, N):
        """
        Compute the focal field using a general formulation.

        The field incident onto the objective is assumed to be linearly
        polarized along the x axis.

        Parameters
        ----------
        z : float
            (m ) defocus of the beam (``z = 0`` corresponds to the focal
            plane).
        rho_max : float
            (m ) maximum radial coordinate for evaluation.
        N : int
            Number of points in each direction for the output grid.

        Returns
        -------
        xi, yi : ndarray
            Vectors (1D numpy arrays) defining the interpolation grid.
        Exi, Eyi, Ezi : ndarray
            Complex electric field components on the grid.  Specified as
            2D arrays.
        """
        E0 = 1  # Amplitude of the incident electric field
        n1, n2 = (
            1,
            1,
        )  # Refractive indices of the medium and the lens - not properly implemented
        k0 = 2 * np.pi / self.wavelength * n2  # wavenumber of the light
        theta_max = np.arcsin(self.NA / n2)  # Maximum angle of the light cone

        theta = np.linspace(0, theta_max, 41)  # Angular coordinate
        fw = np.exp(
            -1 / (self.f0**2) * (np.sin(theta) ** 2) / (np.sin(theta_max) ** 2)
        )  # Apodization function
        phi = np.linspace(0, 2 * np.pi, 45)  # Azimuthal coordinate
        rho = np.linspace(1e-12, rho_max, 180)  # Radial coordinate

        # Initialize arrays for the integrals
        I00 = np.zeros(rho.shape, dtype=complex)
        I01 = np.zeros(rho.shape, dtype=complex)
        I02 = np.zeros(rho.shape, dtype=complex)
        Ex = np.zeros((len(rho), len(phi)), dtype=complex)
        Ey = np.zeros((len(rho), len(phi)), dtype=complex)
        Ez = np.zeros((len(rho), len(phi)), dtype=complex)

        for i, rhoi in enumerate(rho):
            # Compute the integrals for the electric field components
            I00[i] = simpson(
                fw
                * (np.cos(theta) ** (1 / 2))
                * np.sin(theta)
                * (1 + np.cos(theta))
                * jv(0, k0 * rhoi * np.sin(theta))
                * np.exp(1j * k0 * z * np.cos(theta)),
                x=theta,
            )
            I01[i] = simpson(
                fw
                * (np.cos(theta) ** (1 / 2))
                * (np.sin(theta) ** 2)
                * jv(1, k0 * rhoi * np.sin(theta))
                * np.exp(1j * k0 * z * np.cos(theta)),
                x=theta,
            )
            I02[i] = simpson(
                fw
                * (np.cos(theta) ** (1 / 2))
                * np.sin(theta)
                * (1 - np.cos(theta))
                * jv(2, k0 * rhoi * np.sin(theta))
                * np.exp(1j * k0 * z * np.cos(theta)),
                x=theta,
            )
            for j, phii in enumerate(phi):
                # Prefactor according to Novotny & Hecht, eq. (3.66)
                common_factor = (
                    1j
                    * k0
                    * self.f
                    / 2
                    * np.sqrt(n1 / n2)
                    * E0
                    * np.exp(-1j * k0 * self.f)
                )
                Ex[i, j] = common_factor * (I00[i] + I02[i] * np.cos(2 * phii))
                Ey[i, j] = common_factor * (I02[i] * np.sin(2 * phii))
                Ez[i, j] = common_factor * (-2j * I01[i] * np.cos(phii))
        # Create a grid for the interpolation
        PHI, RHO = np.meshgrid(phi, rho)
        X = RHO * np.cos(PHI)
        Y = RHO * np.sin(PHI)
        xi = np.linspace(np.min(X), np.max(X), N)
        yi = np.linspace(np.min(Y), np.max(Y), N)
        XI, YI = np.meshgrid(xi, yi)
        # Interpolate the electric field components
        Exi = self._scattered_interpolant(X.ravel(), Y.ravel(), Ex.ravel(), XI, YI)
        Eyi = self._scattered_interpolant(X.ravel(), Y.ravel(), Ey.ravel(), XI, YI)
        Ezi = self._scattered_interpolant(X.ravel(), Y.ravel(), Ez.ravel(), XI, YI)

        return xi, yi, Exi, Eyi, Ezi

    def getFocalFieldRad(self, z, rho_max, N):
        """
        Compute the focal field using a radial formulation.

        The field incident onto the objective is assumed to be a radially
        polarized doughnut beam (superposition of Hermite-Gaussian modes
        HG10 and HG01) with amplitude
        ``E0 * 2*rho/w0 * exp(-rho**2/w0**2)``, where
        ``w0 = f0 * f * sin(theta_max)`` is the beam waist given by the
        filling factor `f0`.  (Note that "radial" polarization in
        :meth:`getPupilField` assumes a Gaussian amplitude instead.)

        Parameters
        ----------
        z : float
            (m ) defocus of the beam (``z = 0`` corresponds to the focal
            plane).
        rho_max : float
            (m ) maximum radial coordinate for evaluation.
        N : int
            Number of points in each direction for the output grid.

        Returns
        -------
        xi, yi : 1D numpy arrays
            Vectors defining the interpolation grid.
        Exi, Eyi, Ezi : ndarray
            Complex electric field components on the grid.  Specified as
            2D arrays.
        """
        k0 = 2 * np.pi / self.wavelength
        E0 = 1
        theta_max = np.arcsin(self.NA)
        n1, n2 = 1, 1
        w0 = self.f0 * self.f * np.sin(theta_max)  # incident beam waist

        theta = np.linspace(0, theta_max, 41)
        fw = np.exp(-1 / (self.f0**2) * (np.sin(theta) ** 2) / (np.sin(theta_max) ** 2))

        phi = np.linspace(0, 2 * np.pi, 45)
        rho = np.linspace(0, rho_max, 180)

        Irad = np.zeros(rho.shape, dtype=complex)
        I10 = np.zeros(rho.shape, dtype=complex)
        Ex = np.zeros((len(rho), len(phi)), dtype=complex)
        Ey = np.zeros((len(rho), len(phi)), dtype=complex)
        Ez = np.zeros((len(rho), len(phi)), dtype=complex)

        for i, rhoi in enumerate(rho):
            integrand_rad = (
                fw
                * (np.cos(theta) ** (3 / 2))
                * (np.sin(theta) ** 2)
                * jv(1, k0 * rhoi * np.sin(theta))
                * np.exp(1j * k0 * z * np.cos(theta))
            )
            Irad[i] = simpson(integrand_rad, x=theta)

            integrand_I10 = (
                fw
                * (np.cos(theta) ** (1 / 2))
                * (np.sin(theta) ** 3)
                * jv(0, k0 * rhoi * np.sin(theta))
                * np.exp(1j * k0 * z * np.cos(theta))
            )
            I10[i] = simpson(integrand_I10, x=theta)

            for j, phii in enumerate(phi):
                # Prefactor according to Novotny & Hecht, eq. (3.70)
                common_factor = (
                    1j
                    * k0
                    * self.f**2
                    / (2 * w0)
                    * np.sqrt(n1 / n2)
                    * E0
                    * np.exp(-1j * k0 * self.f)
                )
                Ex[i, j] = common_factor * (4j * Irad[i] * np.cos(phii))
                Ey[i, j] = common_factor * (4j * Irad[i] * np.sin(phii))
                Ez[i, j] = common_factor * (-4 * I10[i])

        PHI, RHO = np.meshgrid(phi, rho)
        X = RHO * np.cos(PHI)
        Y = RHO * np.sin(PHI)
        xi = np.linspace(np.min(X), np.max(X), N)
        yi = np.linspace(np.min(Y), np.max(Y), N)
        XI, YI = np.meshgrid(xi, yi)

        Exi = self._scattered_interpolant(X.ravel(), Y.ravel(), Ex.ravel(), XI, YI)
        Eyi = self._scattered_interpolant(X.ravel(), Y.ravel(), Ey.ravel(), XI, YI)
        Ezi = self._scattered_interpolant(X.ravel(), Y.ravel(), Ez.ravel(), XI, YI)

        return xi, yi, Exi, Eyi, Ezi

    def getFocalFieldAzm(self, z, rho_max, N):
        """
        Compute the focal field using an azimuthal formulation
        (``E_z = 0``).

        The field incident onto the objective is assumed to be an
        azimuthally polarized doughnut beam with amplitude
        ``E0 * 2*rho/w0 * exp(-rho**2/w0**2)``, where
        ``w0 = f0 * f * sin(theta_max)`` is the beam waist given by the
        filling factor `f0`.  (Note that "azimuthal" polarization in
        :meth:`getPupilField` assumes a Gaussian amplitude instead.)

        Parameters
        ----------
        z : float
            (m ) defocus of the beam (``z = 0`` corresponds to the focal
            plane).
        rho_max : float
            (m ) maximum radial coordinate for evaluation.
        N : int
            Number of points in each direction for the output grid.

        Returns
        -------
        xi, yi : 1D numpy arrays
            Vectors defining the interpolation grid.
        Exi, Eyi, Ezi : ndarray
            Complex electric field components on the grid (with ``E_z``
            identically zero).  Specified as 2D arrays.
        """
        k0 = 2 * np.pi / self.wavelength
        E0 = 1
        theta_max = np.arcsin(self.NA)
        n1, n2 = 1, 1
        w0 = self.f0 * self.f * np.sin(theta_max)  # incident beam waist

        theta = np.linspace(0, theta_max, 41)
        fw = np.exp(-1 / (self.f0**2) * (np.sin(theta) ** 2) / (np.sin(theta_max) ** 2))
        phi = np.linspace(0, 2 * np.pi, 45)
        rho = np.linspace(0, rho_max, 180)

        Iazm = np.zeros(rho.shape, dtype=complex)
        Ex = np.zeros((len(rho), len(phi)), dtype=complex)
        Ey = np.zeros((len(rho), len(phi)), dtype=complex)
        Ez = np.zeros((len(rho), len(phi)), dtype=complex)  # Remains zero

        for i, rhoi in enumerate(rho):
            integrand_azm = (
                fw
                * (np.cos(theta) ** (1 / 2))
                * (np.sin(theta) ** 2)
                * jv(1, k0 * rhoi * np.sin(theta))
                * np.exp(1j * k0 * z * np.cos(theta))
            )
            Iazm[i] = simpson(integrand_azm, x=theta)
            for j, phii in enumerate(phi):
                # Prefactor according to Novotny & Hecht, eq. (3.72), here
                # with incident polarization along (-sin(phi), cos(phi))
                common_factor = (
                    1j
                    * k0
                    * self.f**2
                    / (2 * w0)
                    * np.sqrt(n1 / n2)
                    * E0
                    * np.exp(-1j * k0 * self.f)
                )
                Ex[i, j] = common_factor * (-4j * Iazm[i] * np.sin(phii))
                Ey[i, j] = common_factor * (4j * Iazm[i] * np.cos(phii))
                Ez[i, j] = 0

        PHI, RHO = np.meshgrid(phi, rho)
        X = RHO * np.cos(PHI)
        Y = RHO * np.sin(PHI)
        xi = np.linspace(np.min(X), np.max(X), N)
        yi = np.linspace(np.min(Y), np.max(Y), N)
        XI, YI = np.meshgrid(xi, yi)

        Exi = self._scattered_interpolant(X.ravel(), Y.ravel(), Ex.ravel(), XI, YI)
        Eyi = self._scattered_interpolant(X.ravel(), Y.ravel(), Ey.ravel(), XI, YI)
        Ezi = self._scattered_interpolant(X.ravel(), Y.ravel(), Ez.ravel(), XI, YI)

        return xi, yi, Exi, Eyi, Ezi

    def getPupilField(
        self, z, KX, KY, n=1.0, pol_type="linear", pol_angle=0, axis_ratio=1.0
    ):
        """
        Computes the complex electric field distribution in reciprocal space.

        This represents the field near the focus of a high-NA objective
        lens—including amplitude apodization, polarization
        transformation, and defocus phase, projected onto the kx-ky
        plane. This is the integrand for the vectorial Debye
        diffraction integral.

        Parameters
        ----------
        z : float
            (m ) defocus distance along optical axis.
        KX : ndarray
            (rad/m) 2D reciprocal-space grid (kx).
        KY : ndarray
            (rad/m) 2D reciprocal-space grid (ky).
        n : float, optional
            Refractive index of the focusing medium.
            Default is 1.0 (air/vacuum).
        pol_type : str or array_like, optional
            Polarization of the beam in the entrance pupil (before
            focusing).  Either one of the following strings:

            | "linear" - linearly polarized (angle set by `pol_angle`)
            | "radial" - radial polarization
            | "azimuthal" - azimuthal polarization
            | "rcp" - right-hand circular polarization
            | "lcp" - left-hand circular polarization
            | "elliptical" - elliptically polarized (uses `axis_ratio`
            |                and `pol_angle`)

            or a Jones vector with shape ``(2,)``, or a spatially
            varying Jones field with shape ``(2, *KX.shape)`` defined
            on the `KX`, `KY` grid.  Such Jones vectors/fields can be
            prepared using the :mod:`~SpinWaveToolkit.bls.polarization`
            module, e.g. to include wave plates, spiral phase plates or
            q-plates in the incident beam path.  Default is "linear".
        pol_angle : float, optional
            (deg) angle of linear polarization or the major axis of
            elliptical polarization.  Default is 0.  Ignored if
            `pol_type` is not a string.
        axis_ratio : float, optional
            Ratio of the minor axis to the major axis for elliptical
            polarization.  Can be positive or negative to dictate
            handedness.  Default is 1.0.  Ignored if `pol_type`
            is not "elliptical".

        Returns
        -------
        Ex_k, Ey_k, Ez_k : ndarray
            Complex electric field components in k-space (2D arrays).

        Notes
        -----
        The amplitude of the incident beam is always Gaussian, given by
        the filling factor `f0`, with the polarization state set by
        `pol_type`.  Therefore, "radial" and "azimuthal" polarizations
        here correspond to a Gaussian beam with a polarization
        singularity on the optical axis (e.g. directly after a q-plate),
        whereas :meth:`getFocalFieldRad` and :meth:`getFocalFieldAzm`
        assume a doughnut beam with amplitude
        ``E0 * 2*rho/w0 * exp(-rho**2/w0**2)``.  The doughnut beam can be
        obtained here by weighting the Jones field by ``2*rho/w0``,
        which with the sine condition ``rho = f*sin(theta)`` and
        ``w0 = f0*f*sin(theta_max)`` reads:

        .. code-block:: python

            sin_theta = np.hypot(KX, KY) / (2 * np.pi * n / wavelength)
            e_in = 2 * sin_theta / (f0 * NA / n) * pol.jones_vector(
                "radial", X=KX, Y=KY
            )
            Ex_k, Ey_k, Ez_k = lens.getPupilField(z, KX, KY, n=n, pol_type=e_in)

        Examples
        --------
        Focusing a vortex beam with topological charge 1, made from a
        circularly polarized beam by a spiral phase plate:

        .. code-block:: python

            import numpy as np
            import SpinWaveToolkit as SWT

            pol = SWT.bls.polarization
            lens = SWT.bls.ObjectiveLens(532e-9, 0.75, 2, 1e-3)
            kx = np.linspace(-15e6, 15e6, 201)
            KX, KY = np.meshgrid(kx, kx, indexing="ij")
            e_in = pol.apply_jones_matrix(
                pol.spiral_phase_plate(KX, KY, charge=1), pol.jones_vector("rcp")
            )
            Ex_k, Ey_k, Ez_k = lens.getPupilField(0, KX, KY, pol_type=e_in)
        """

        # --- CONSTANTS & PRELIMINARIES ---
        # Prevent math domain errors (arcsin(x) where x > 1)
        if self.NA > n:
            raise ValueError(
                f"Numerical aperture (NA={self.NA}) cannot exceed the "
                f"refractive index of the medium (n={n})."
            )

        k_vac = 2 * np.pi / self.wavelength     # Vacuum wave number
        k_medium = k_vac * n                    # Medium wave number
        theta_max = np.arcsin(self.NA / n)      # Max focusing angle

        # Radial k-vector and pupil mask (defines the physical aperture limit)
        K_rho = np.sqrt(KX**2 + KY**2)
        pupil_mask = K_rho <= (k_vac * self.NA)

        # Initialize output fields (complex)
        Ex_k = np.zeros_like(KX, dtype=complex)
        Ey_k = np.zeros_like(KX, dtype=complex)
        Ez_k = np.zeros_like(KX, dtype=complex)

        # --- ANGULAR COORDINATES ---
        # Select only points within the pupil for calculation
        k_rho = K_rho[pupil_mask]
        kx = KX[pupil_mask]
        ky = KY[pupil_mask]

        # Map transverse wavevectors to spherical angles inside the medium
        sin_theta = np.clip(k_rho / k_medium, -1, 1)
        cos_theta = np.sqrt(1 - sin_theta**2)

        # Avoid division by zero at k_rho = 0
        cos_phi = np.ones_like(k_rho)
        sin_phi = np.zeros_like(k_rho)
        nonzero = k_rho > 1e-12
        cos_phi[nonzero] = kx[nonzero] / k_rho[nonzero]
        sin_phi[nonzero] = ky[nonzero] / k_rho[nonzero]

        # --- APODIZATION (ILLUMINATION PROFILE) ---
        # Gaussian amplitude weighting (filling factor f0) and
        # 1/sqrt(cos) factor (sine condition + Cartesian Jacobian mapping)
        fw = np.exp(-((sin_theta / np.sin(theta_max)) ** 2) / self.f0**2)
        amplitude_factor = fw / np.sqrt(cos_theta)

        # Defocus propagator (phase term for defocus z in the medium)
        propagator = np.exp(1j * k_medium * z * cos_theta)

        # --- POLARIZATION BASIS TRANSFORMATION ---
        # Jones vector/field in the entrance pupil (before focusing)
        if isinstance(pol_type, str):
            e_in = jones_vector(pol_type, pol_angle, axis_ratio, X=KX, Y=KY)
        else:
            e_in = np.asarray(pol_type)
        if e_in.shape not in ((2,), (2, *KX.shape)):
            raise ValueError(
                f"Jones vector/field must have shape (2,) or {(2, *KX.shape)}, "
                f"got {e_in.shape}."
            )
        # Broadcast to the KX grid and select the points within the pupil
        if e_in.ndim == 1:
            e_in = e_in.reshape((2,) + (1,) * KX.ndim)
        e_in = np.broadcast_to(e_in, (2, *KX.shape))[:, pupil_mask]

        # Transformation according to Richards & Wolf (1959)
        ex = (cos_theta * cos_phi**2 + sin_phi**2) * e_in[0] + (
            cos_theta - 1
        ) * cos_phi * sin_phi * e_in[1]
        ey = (cos_theta - 1) * cos_phi * sin_phi * e_in[0] + (
            cos_theta * sin_phi**2 + cos_phi**2
        ) * e_in[1]
        ez = -sin_theta * (cos_phi * e_in[0] + sin_phi * e_in[1])

        # --- ASSEMBLE FINAL FIELD IN K-SPACE ---
        E0 = 1.0  # Input amplitude normalization
        # Angular spectrum representation, Novotny & Hecht eq. (3.47)
        prefactor = (
            1j * E0 * self.f / (2 * np.pi * k_medium) * np.exp(-1j * k_medium * self.f)
        )
        
        Ex_k[pupil_mask] = prefactor * amplitude_factor * propagator * ex
        Ey_k[pupil_mask] = prefactor * amplitude_factor * propagator * ey
        Ez_k[pupil_mask] = prefactor * amplitude_factor * propagator * ez

        return Ex_k, Ey_k, Ez_k
