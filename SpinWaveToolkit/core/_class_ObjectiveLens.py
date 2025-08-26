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
        (m) wavelength of the light.
    NA : float
        Numerical aperture of the objective lens.
    f0 : float
        Filling factor.
    f : float
        (m) focal length of the objective lens.

    Attributes
    ----------
    same as Parameters

    Methods
    -------
    getFocalFieldRad(z, rho_max, N)
        Compute the focal field using the radial formulation.
    getFocalFieldAzm(z, rho_max, N)
        Compute the focal field using the azimuthal formulation
        (with ``E_z = 0``).
    getFocalField(z, rho_max, N)
        Compute the focal field using a general formulation.
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
            (m) defocus of the beam (``z = 0`` corresponds to the focal
            plane).
        rho_max : float
            (m) maximum radial coordinate for evaluation.
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
            (m) defocus of the beam (``z = 0`` corresponds to the focal
            plane).
        rho_max : float
            (m) maximum radial coordinate for evaluation.
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
            (m) defocus of the beam (``z = 0`` corresponds to the focal
            plane).
        rho_max : float
            (m) maximum radial coordinate for evaluation.
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
