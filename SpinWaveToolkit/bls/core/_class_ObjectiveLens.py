"""
Core (private) file for the `ObjectiveLens` class.
"""

import numpy as np
from scipy.interpolate import griddata
from scipy.special import jv  # Bessel function of first kind
from scipy.integrate import simpson  # Import simpson for numerical integration

__all__ = ["ObjectiveLens"]


class ObjectiveLens:
    """
    Represents an objective lens with specific optical parameters.

    Module for calculating the electric field focused by an objective
    lens.  Calculations follows the method presented in book of Novotny
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
                theta,
            )
            I01[i] = simpson(
                fw
                * (np.cos(theta) ** (1 / 2))
                * (np.sin(theta) ** 2)
                * jv(1, k0 * rhoi * np.sin(theta))
                * np.exp(1j * k0 * z * np.cos(theta)),
                theta,
            )
            I02[i] = simpson(
                fw
                * (np.cos(theta) ** (1 / 2))
                * np.sin(theta)
                * (1 - np.cos(theta))
                * jv(2, k0 * rhoi * np.sin(theta))
                * np.exp(1j * k0 * z * np.cos(theta)),
                theta,
            )
            for j, phii in enumerate(phi):
                common_factor = (
                    1j
                    * k0
                    * self.f
                    / 2
                    * np.sqrt(n1 / n2)
                    * E0
                    * np.exp(1j * k0 * self.f)
                )
                Ex[i, j] = common_factor * (I00[i] + I02[i] * np.cos(2 * phii))
                Ey[i, j] = common_factor * (I02[i] * np.sin(2 * phii))
                Ez[i, j] = common_factor * (-2j * I01[i] * np.sin(phii))
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
            Irad[i] = simpson(integrand_rad, theta)

            integrand_I10 = (
                fw
                * (np.cos(theta) ** (1 / 2))
                * (np.sin(theta) ** 3)
                * jv(0, k0 * rhoi * np.sin(theta))
                * np.exp(1j * k0 * z * np.cos(theta))
            )
            I10[i] = simpson(integrand_I10, theta)

            for j, phii in enumerate(phi):
                common_factor = (
                    1j
                    * k0
                    * self.f**2
                    / 2
                    * np.sqrt(n1 / n2)
                    * E0
                    * np.exp(-1j * k0 * self.f)
                )
                Ex[i, j] = common_factor * (1j * Irad[i] * np.cos(phii))
                Ey[i, j] = common_factor * (1j * Irad[i] * np.sin(phii))
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
            Iazm[i] = simpson(integrand_azm, theta)
            for j, phii in enumerate(phi):
                common_factor = (
                    1j
                    * k0
                    * self.f**2
                    / 2
                    * np.sqrt(n1 / n2)
                    * E0
                    * np.exp(-1j * k0 * self.f)
                )
                Ex[i, j] = common_factor * (1j * Iazm[i] * np.sin(phii))
                Ey[i, j] = common_factor * (-1j * Iazm[i] * np.cos(phii))
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
    
    def getPupilField(self, z, KX, KY, polarization_type='linear', polarization_angle_deg=0):
        """
        Computes the complex electric field distribution in reciprocal space.

        This represents the field near the focus of a high-NA objective
        lens—including amplitude apodization, polarization transformation,
        and defocus phase, projected onto the kx-ky plane. This is the
        integrand for the vectorial Debye diffraction integral.

        Parameters
        ----------
        z : float
            (m ) defocus distance along optical axis.
        KX : np.ndarray
            (1/m ) 2D reciprocal-space grid (kx).
        KY : np.ndarray
            (1/m ) 2D reciprocal-space grid (ky).
        polarization_type : str, optional
            'linear' — linearly polarized (angle set by polarization_angle_deg)
            'radial' — radial polarization
            'azimuthal' — azimuthal polarization
            'rcp' — right-hand circular polarization
            'lcp' — left-hand circular polarization
        polarization_angle_deg : float, optional
            Linear polarization angle in degrees (default 0, ignored otherwise).

        Returns
        -------
        Ex_k, Ey_k, Ez_k : np.ndarray
            Complex electric field components in k-space (2D arrays).
        """

        # --- CONSTANTS & PRELIMINARIES ---
        n = 1  # Refractive index (air or vacuum)
        k0 = 2 * np.pi * n / self.wavelength  # Wave number
        theta_max = np.arcsin(self.NA / n)   # Max focusing angle from NA

        # Radial k-vector and pupil mask (defines the aperture)
        K_rho = np.sqrt(KX**2 + KY**2)
        pupil_mask = K_rho <= k0 * self.NA

        # Initialize output fields (complex)
        Ex_k = np.zeros_like(KX, dtype=complex)
        Ey_k = np.zeros_like(KX, dtype=complex)
        Ez_k = np.zeros_like(KX, dtype=complex)

        # --- ANGULAR COORDINATES ---
        # Select only points within the pupil for calculation
        k_rho = K_rho[pupil_mask]
        kx = KX[pupil_mask]
        ky = KY[pupil_mask]

        sin_theta = np.clip(k_rho / k0, -1, 1)
        cos_theta = np.sqrt(1 - sin_theta**2)

        # Avoid division by zero at k_rho = 0
        cos_phi = np.ones_like(k_rho)
        sin_phi = np.zeros_like(k_rho)
        nonzero = k_rho > 1e-12
        cos_phi[nonzero] = kx[nonzero] / k_rho[nonzero]
        sin_phi[nonzero] = ky[nonzero] / k_rho[nonzero]

        # --- APODIZATION (ILLUMINATION PROFILE) ---
        # Gaussian amplitude weighting (filling factor f0) and
        # cosine factor accounting for obliquity (from Debye theory)
        fw = np.exp(- (sin_theta / np.sin(theta_max))**2 / self.f0**2)
        amplitude_factor = fw / np.sqrt(cos_theta)

        # Defocus propagator (phase term for defocus z)
        propagator = np.exp(1j * k0 * z * cos_theta)

        # --- POLARIZATION BASIS TRANSFORMATION ---
        if polarization_type == 'linear':
            # Linear polarization at arbitrary angle in the pupil plane
            angle_rad = np.deg2rad(polarization_angle_deg)

            # Jones vector in the entrance pupil (before focusing)
            e_in = np.array([np.cos(angle_rad), np.sin(angle_rad), 0])

            # Transformation according to Richards & Wolf (1959)
            ex = (cos_theta * cos_phi**2 + sin_phi**2) * e_in[0] \
               + (cos_theta - 1) * cos_phi * sin_phi * e_in[1]
            ey = (cos_theta - 1) * cos_phi * sin_phi * e_in[0] \
               + (cos_theta * sin_phi**2 + cos_phi**2) * e_in[1]
            ez = -sin_theta * (cos_phi * e_in[0] + sin_phi * e_in[1])

        elif polarization_type == 'radial':
            ex = sin_theta * cos_phi
            ey = sin_theta * sin_phi
            ez = cos_theta

        elif polarization_type == 'azimuthal':
            ex = -sin_phi
            ey = cos_phi
            ez = np.zeros_like(ex)

        elif polarization_type in ['rcp', 'lcp']:
            # Right/left circular polarization (RCP = +, LCP = -)
            sign = +1 if polarization_type == 'rcp' else -1

            # Local transverse basis
            e_theta = np.array([cos_theta * cos_phi, cos_theta * sin_phi, -sin_theta])
            e_phi = np.array([-sin_phi, cos_phi, np.zeros_like(sin_phi)])

            # Superposition of theta and phi unit vectors with ±i phase
            ex = (e_theta[0] + sign * 1j * e_phi[0]) / np.sqrt(2)
            ey = (e_theta[1] + sign * 1j * e_phi[1]) / np.sqrt(2)
            ez = (e_theta[2] + sign * 1j * e_phi[2]) / np.sqrt(2)

        else:
            raise ValueError(f"Polarization type '{polarization_type}' not recognized. "
                             "Use 'linear', 'radial', 'azimuthal', 'rcp', or 'lcp'.")

        # --- ASSEMBLE FINAL FIELD IN K-SPACE ---
        E0 = 1.0  # Input amplitude normalization
        prefactor = E0 * self.f
        Ex_k[pupil_mask] = prefactor * amplitude_factor * propagator * ex
        Ey_k[pupil_mask] = prefactor * amplitude_factor * propagator * ey
        Ez_k[pupil_mask] = prefactor * amplitude_factor * propagator * ez

        return Ex_k, Ey_k, Ez_k