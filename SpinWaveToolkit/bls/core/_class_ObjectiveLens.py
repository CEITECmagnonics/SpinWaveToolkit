"""
Core (private) file for the `ObjectiveLens` class.
"""

from warnings import warn
import numpy as np
from scipy.special import j0, j1  # Bessel functions of first kind
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

    def _focal_field(self, z, rho_max, N, rtol, kernels, coef, assemble):
        """
        Focal field from integrals over the focusing angle with
        automatically chosen sampling.

        Common part of the focal field methods.  The integrals

            ``I_k(rho) = int_0^theta_max g_k(theta) J_{n_k}(k0 rho sin(theta))
            exp(i k0 z cos(theta)) dtheta``

        are evaluated by Simpson's rule on a radial grid and assembled
        into the field on the ``N x N`` output grid.  The numbers of
        points of the angular and radial grids are chosen such that the
        estimated maximum error of the field (relative to the maximum of
        its components on the output grid) is below `rtol`, half of
        which is assigned to each of the two discretizations:

        - Angular quadrature: the initial number of points follows from
          the number of oscillations of the integrand over the aperture.
          The error is estimated by comparing Simpson's rule with that on
          every other point (``|I_h - I_2h|/15``).
        - Radial grid: if the output grid has fewer distinct radii than
          needed for the interpolation, the integrals are evaluated
          exactly at these radii.  Otherwise, they are evaluated on an
          equidistant radial grid and linearly interpolated, with the
          error estimated from the second differences of the integrals
          (``max |I_i - (I_{i-1} + I_{i+1})/2| / 4``, i.e. the maximum
          interpolation error for a locally constant second
          derivative).

        If an estimate exceeds its part of `rtol`, the corresponding grid
        is refined accordingly and the calculation repeated.

        Parameters
        ----------
        z, rho_max, N, rtol :
            As in :meth:`getFocalField`.
        kernels : list[tuple]
            ``(n_k, g_k)`` for each integral, where ``n_k`` in
            ``{0, 1, 2}`` is the order of the Bessel function and
            ``g_k(theta)`` returns the angular weight (without the Bessel
            function and the defocus phase).
        coef : ndarray
            Array with shape ``(3, len(kernels))``, upper bounds of the
            magnitude of the coefficients of ``I_k`` in the field
            components (including the common prefactor), used to
            propagate the error estimates of the integrals to the field.
        assemble : callable
            ``assemble(I, cos_phi, sin_phi) -> (Ex, Ey, Ez)`` assembling
            the field from the integrals ``I`` (list of arrays with the
            shape of the output grid) and the cosine and sine of the
            azimuth of the output grid.

        Returns
        -------
        xi, yi, Exi, Eyi, Ezi : ndarray
            Output grid and field components as in :meth:`getFocalField`.
        """
        if not 1e-10 <= rtol < 1:
            # (smaller values approach the round-off errors and could not be reached)
            raise ValueError(f"`rtol` must be between 1e-10 and 1, got {rtol}.")
        k0 = 2 * np.pi / self.wavelength
        theta_max = np.arcsin(self.NA)
        tol = rtol / 2  # error budget for each of the two discretizations

        xi = np.linspace(-rho_max, rho_max, N)
        yi = np.linspace(-rho_max, rho_max, N)
        RHO = np.sqrt(xi[:, np.newaxis] ** 2 + yi**2)
        # Azimuth of the output grid (phi = 0 on the optical axis)
        cos_phi = np.divide(
            xi[:, np.newaxis], RHO, out=np.ones_like(RHO), where=RHO > 0
        )
        sin_phi = np.divide(yi, RHO, out=np.zeros_like(RHO), where=RHO > 0)
        # Distinct radii of the output grid: x_i = q_i * rho_max/(N-1) with
        # integers q_i, so they follow from the distinct values of
        # q_i**2 + q_j**2 (found on one octant of the grid)
        if N > 1:
            q = 2 * np.arange(N) - (N - 1)
            q2 = np.unique(q**2)
            iu, ju = np.triu_indices(len(q2))
            s = np.unique(q2[iu] + q2[ju])
            rho_exact = rho_max / (N - 1) * np.sqrt(s)
        else:
            rho_exact = RHO.ravel()
        rho_end = rho_exact[-1]
        idx_exact = None  # index of the radius of each grid point in rho_exact

        # Initial sampling from error models calibrated to the estimates
        # below, so that usually no refinement is needed: Simpson's rule
        # ~ 0.03*p**-4 for p points per oscillation period of the integrand
        # (at least 6, so that the estimate on every other point is
        # reliable), interpolation ~ 0.05*(drho*k0*NA)**2
        n_osc = (
            k0 * (rho_end * self.NA + abs(z) * (1 - np.cos(theta_max))) / (2 * np.pi)
        )
        Ntheta = _simpson_points(max(6, (0.03 / tol) ** 0.25) * n_osc)
        Nrho = int(np.ceil(rho_end * k0 * self.NA / np.sqrt(tol / 0.05))) + 1

        for _ in range(5):
            exact = len(rho_exact) <= Nrho
            rho = rho_exact if exact else np.linspace(0, rho_end, Nrho)
            I, dI_theta = self._radial_integrals(rho, Ntheta, kernels, z)
            if exact:
                if idx_exact is None:
                    idx_exact = (
                        np.searchsorted(s, np.add.outer(q**2, q**2))
                        if N > 1
                        else np.zeros(RHO.shape, dtype=int)
                    )
                E = assemble([Ik[idx_exact] for Ik in I], cos_phi, sin_phi)
                err_rho = 0.0
            else:
                # Linear interpolation on the equidistant radial grid
                t = RHO * ((Nrho - 1) / rho_end)
                i = np.minimum(t.astype(int), Nrho - 2)
                t -= i
                E = assemble([Ik[i] + np.diff(Ik)[i] * t for Ik in I], cos_phi, sin_phi)
                d2 = np.abs(I[:, 1:-1] - (I[:, :-2] + I[:, 2:]) / 2).max(axis=1) / 4
                err_rho = (coef @ d2).max()
            err_theta = (coef @ dI_theta).max()
            E_max = max(np.abs(c).max() for c in E)
            if err_theta <= tol * E_max and err_rho <= tol * E_max:
                break
            # Refine the grid(s) with too large error estimates
            if err_theta > tol * E_max:
                ratio = (err_theta / (tol * E_max)) ** 0.25
                Ntheta = _simpson_points(1.2 * ratio * (Ntheta - 1))
            if err_rho > tol * E_max:
                ratio = np.sqrt(err_rho / (tol * E_max))
                Nrho = int(np.ceil(1.2 * ratio * (Nrho - 1))) + 1
        else:
            warn(
                f"The focal field did not reach the requested accuracy rtol = {rtol:.1e} "
                + f"(estimated errors {err_theta / E_max:.1e} of the angular integration "
                + f"and {err_rho / E_max:.1e} of the radial interpolation).",
                UserWarning,
                stacklevel=3,
            )
        return (xi, yi, *E)

    def _radial_integrals(self, rho, Ntheta, kernels, z):
        """
        Integrals over the focusing angle at radii `rho` (Simpson's rule
        with `Ntheta` points, ``Ntheta - 1`` divisible by 4).

        Returns the integrals with shape ``(len(kernels), len(rho))`` and
        the error estimates ``max_rho |I_h - I_2h|/15`` of each of them.
        """
        k0 = 2 * np.pi / self.wavelength
        theta = np.linspace(0, np.arcsin(self.NA), Ntheta)
        sin_theta = np.sin(theta)
        phase = np.exp(1j * k0 * z * np.cos(theta))
        # Simpson weights on the full grid and on every other point
        w_fine = _simpson_weights(Ntheta, theta[1] - theta[0])
        w_coarse = _simpson_weights((Ntheta + 1) // 2, 2 * (theta[1] - theta[0]))
        G = [g(theta) * phase for _, g in kernels]
        v_fine = [w_fine * Gk for Gk in G]
        v_coarse = [w_coarse * Gk[::2] for Gk in G]
        orders = {n for n, _ in kernels}

        I = np.empty((len(kernels), len(rho)), dtype=complex)
        I_coarse = np.empty_like(I)
        chunk = max(1, 2**21 // Ntheta)  # limits the memory of the Bessel arrays
        for start in range(0, len(rho), chunk):
            sl = slice(start, start + chunk)
            x = k0 * np.outer(rho[sl], sin_theta)
            J = {}
            if orders & {0, 2}:
                J[0] = j0(x)
            if orders & {1, 2}:
                J[1] = j1(x)
            if 2 in orders:  # recurrence J2 = 2*J1/x - J0, with J2(0) = 0
                J[2] = np.divide(2 * J[1], x, out=J[0].copy(), where=x != 0) - J[0]
            for k, (n, _) in enumerate(kernels):
                # real matrix times complex vector, avoiding a complex copy of J
                I[k, sl] = J[n] @ v_fine[k].real + 1j * (J[n] @ v_fine[k].imag)
                Jc = J[n][:, ::2]
                I_coarse[k, sl] = Jc @ v_coarse[k].real + 1j * (Jc @ v_coarse[k].imag)
        return I, np.abs(I - I_coarse).max(axis=1) / 15

    def getFocalField(self, z, rho_max, N, rtol=1e-4):
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
            (m ) maximum coordinate for evaluation, i.e. the output grid
            spans ``[-rho_max, rho_max]`` in both x and y.
        N : int
            Number of points in each direction for the output grid.  An
            odd number is recommended, so that the grid contains the focus
            ``x = y = 0``.  (Otherwise, the Fourier transform of the field
            acquires a phase ramp corresponding to a shift by half of the
            grid step.)
        rtol : float, optional
            () requested accuracy, i.e. the maximum error of the field
            components on the output grid relative to their maximum.  The
            numerical sampling is chosen automatically to reach it (see
            Notes).  Must be between 1e-10 and 1.  Default is 1e-4.

        Returns
        -------
        xi, yi : ndarray
            Vectors (1D numpy arrays) defining the output grid.
        Exi, Eyi, Ezi : ndarray
            Complex electric field components on the grid.  Specified as
            2D arrays with shape ``(N, N)``, indexed as ``[ix, iy]`` (same
            as :meth:`getPupilField` on a grid with ``indexing="ij"``).

        Notes
        -----
        The field is given by integrals over the focusing angle, which
        depend only on the radial coordinate, while the azimuthal
        dependence of the field is evaluated exactly.  The integrals are
        evaluated by Simpson's rule and, if the output grid has many
        distinct radii, on an equidistant radial grid from which they
        are linearly interpolated.  Both discretizations are chosen
        such that their estimated errors are below ``rtol/2``:

        - The number of angular points follows from the number of
          oscillations of the integrand over the aperture, which grows
          with `rho_max` and `z`, and the error is estimated by
          comparison with Simpson's rule on every other point.
        - The radial step follows from the wavelength and `NA` (about
          ``wavelength/50`` for ``rtol = 1e-4`` and ``NA = 0.75``), and
          the interpolation error is estimated from the second
          differences of the integrals.  If the output grid has fewer
          distinct radii than the radial grid would need, the integrals
          are evaluated exactly at these radii instead (no
          interpolation), which is typical for small `N`.

        If an estimate is too large, the sampling is refined and the
        calculation repeated.  The estimates are asymptotic (not
        rigorous) bounds; a warning is issued if the accuracy is not
        reached after several refinements.  The computational time
        grows only weakly with `N` and approximately as
        ``rho_max**2 * rtol**(-3/4)`` for large `rho_max` (or with the
        number of distinct radii of the output grid, if smaller).
        """
        E0 = 1  # Amplitude of the incident electric field
        n1, n2 = (
            1,
            1,
        )  # Refractive indices of the medium and the lens - not properly implemented
        k0 = 2 * np.pi / self.wavelength * n2  # wavenumber of the light
        theta_max = np.arcsin(self.NA / n2)  # Maximum angle of the light cone

        def fw(theta):  # Apodization function
            return np.exp(
                -1 / (self.f0**2) * (np.sin(theta) ** 2) / (np.sin(theta_max) ** 2)
            )

        # Angular weights of the integrals I00, I01, I02 (without the Bessel
        # functions and the defocus phase)
        kernels = [
            (0, lambda t: fw(t) * np.sqrt(np.cos(t)) * np.sin(t) * (1 + np.cos(t))),
            (1, lambda t: fw(t) * np.sqrt(np.cos(t)) * np.sin(t) ** 2),
            (2, lambda t: fw(t) * np.sqrt(np.cos(t)) * np.sin(t) * (1 - np.cos(t))),
        ]

        # Prefactor according to Novotny & Hecht, eq. (3.66)
        common_factor = (
            1j * k0 * self.f / 2 * np.sqrt(n1 / n2) * E0 * np.exp(-1j * k0 * self.f)
        )

        def assemble(I, cos_phi, sin_phi):
            I00, I01, I02 = I
            Exi = common_factor * (I00 + I02 * (cos_phi**2 - sin_phi**2))  # cos(2 phi)
            Eyi = common_factor * (I02 * 2 * sin_phi * cos_phi)  # sin(2 phi)
            Ezi = common_factor * (-2j * I01 * cos_phi)
            return Exi, Eyi, Ezi

        coef = np.abs(common_factor) * np.array([[1, 0, 1], [0, 0, 1], [0, 2, 0]])
        return self._focal_field(z, rho_max, N, rtol, kernels, coef, assemble)

    def getFocalFieldRad(self, z, rho_max, N, rtol=1e-4):
        """
        Compute the focal field using a radial formulation.

        The field incident onto the objective is assumed to be a
        radially polarized doughnut beam (superposition of
        Hermite-Gaussian modes HG10 and HG01) with amplitude
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
            (m ) maximum coordinate for evaluation, i.e. the output grid
            spans ``[-rho_max, rho_max]`` in both x and y.
        N : int
            Number of points in each direction for the output grid.  An
            odd number is recommended, so that the grid contains the focus
            ``x = y = 0``.  (Otherwise, the Fourier transform of the field
            acquires a phase ramp corresponding to a shift by half of the
            grid step.)
        rtol : float, optional
            () requested accuracy, i.e. the maximum error of the field
            components on the output grid relative to their maximum.  The
            numerical sampling is chosen automatically to reach it (see
            Notes of :meth:`getFocalField`).  Must be between 1e-10 and
            1.  Default is 1e-4.

        Returns
        -------
        xi, yi : 1D numpy arrays
            Vectors defining the output grid.
        Exi, Eyi, Ezi : ndarray
            Complex electric field components on the grid.  Specified as
            2D arrays with shape ``(N, N)``, indexed as ``[ix, iy]`` (same
            as :meth:`getPupilField` on a grid with ``indexing="ij"``).

        Notes
        -----
        The integrals over the focusing angle depend only on the radial
        coordinate, while the azimuthal dependence of the field is
        evaluated exactly.  The numerical sampling is the same as in
        :meth:`getFocalField`.
        """
        k0 = 2 * np.pi / self.wavelength
        E0 = 1
        theta_max = np.arcsin(self.NA)
        n1, n2 = 1, 1
        w0 = self.f0 * self.f * np.sin(theta_max)  # incident beam waist

        def fw(theta):  # Apodization function
            return np.exp(
                -1 / (self.f0**2) * (np.sin(theta) ** 2) / (np.sin(theta_max) ** 2)
            )

        # Angular weights of the integrals Irad and I10 (without the Bessel
        # functions and the defocus phase)
        kernels = [
            (1, lambda t: fw(t) * np.cos(t) ** (3 / 2) * np.sin(t) ** 2),
            (0, lambda t: fw(t) * np.sqrt(np.cos(t)) * np.sin(t) ** 3),
        ]

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

        def assemble(I, cos_phi, sin_phi):
            Irad, I10 = I
            Exi = common_factor * (4j * Irad * cos_phi)
            Eyi = common_factor * (4j * Irad * sin_phi)
            Ezi = common_factor * (-4 * I10)
            return Exi, Eyi, Ezi

        coef = np.abs(common_factor) * np.array([[4, 0], [4, 0], [0, 4]])
        return self._focal_field(z, rho_max, N, rtol, kernels, coef, assemble)

    def getFocalFieldAzm(self, z, rho_max, N, rtol=1e-4):
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
            (m ) maximum coordinate for evaluation, i.e. the output grid
            spans ``[-rho_max, rho_max]`` in both x and y.
        N : int
            Number of points in each direction for the output grid.  An
            odd number is recommended, so that the grid contains the focus
            ``x = y = 0``.  (Otherwise, the Fourier transform of the field
            acquires a phase ramp corresponding to a shift by half of the
            grid step.)
        rtol : float, optional
            () requested accuracy, i.e. the maximum error of the field
            components on the output grid relative to their maximum.  The
            numerical sampling is chosen automatically to reach it (see
            Notes of :meth:`getFocalField`).  Must be between 1e-10 and
            1.  Default is 1e-4.

        Returns
        -------
        xi, yi : 1D numpy arrays
            Vectors defining the output grid.
        Exi, Eyi, Ezi : ndarray
            Complex electric field components on the grid (with ``E_z``
            identically zero).  Specified as 2D arrays with shape
            ``(N, N)``, indexed as ``[ix, iy]`` (same as
            :meth:`getPupilField` on a grid with ``indexing="ij"``).

        Notes
        -----
        The integral over the focusing angle depends only on the radial
        coordinate, while the azimuthal dependence of the field is
        evaluated exactly.  The numerical sampling is the same as in
        :meth:`getFocalField`.
        """
        k0 = 2 * np.pi / self.wavelength
        E0 = 1
        theta_max = np.arcsin(self.NA)
        n1, n2 = 1, 1
        w0 = self.f0 * self.f * np.sin(theta_max)  # incident beam waist

        def fw(theta):  # Apodization function
            return np.exp(
                -1 / (self.f0**2) * (np.sin(theta) ** 2) / (np.sin(theta_max) ** 2)
            )

        # Angular weight of the integral Iazm (without the Bessel function
        # and the defocus phase)
        kernels = [(1, lambda t: fw(t) * np.sqrt(np.cos(t)) * np.sin(t) ** 2)]

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

        def assemble(I, cos_phi, sin_phi):
            (Iazm,) = I
            Exi = common_factor * (-4j * Iazm * sin_phi)
            Eyi = common_factor * (4j * Iazm * cos_phi)
            Ezi = np.zeros_like(Exi)  # Remains zero
            return Exi, Eyi, Ezi

        coef = np.abs(common_factor) * np.array([[4], [4], [0]])
        return self._focal_field(z, rho_max, N, rtol, kernels, coef, assemble)

    def getPupilField(
        self, z, KX, KY, n=1.0, pol_type="linear", pol_angle=0, axis_ratio=1.0
    ):
        """
        Computes the complex electric field distribution in reciprocal
        space.

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
            (rad/m) 2D reciprocal-space grid (kx).  Use
            ``np.meshgrid(kx, ky, indexing="ij")`` to get the ``[ix, iy]``
            indexing used throughout the :mod:`~SpinWaveToolkit.bls`
            module.
        KY : ndarray
            (rad/m) 2D reciprocal-space grid (ky).
        n : float, optional
            Refractive index of the focusing medium.
            Default is 1.0 (air/vacuum).
        pol_type : str or array_like, optional
            Polarization of the beam in the entrance pupil (before
            focusing).  Either one of the following strings:

            - "linear" - linearly polarized (angle set by `pol_angle`)
            - "radial" - radial polarization
            - "azimuthal" - azimuthal polarization
            - "rcp" - right-hand circular polarization
            - "lcp" - left-hand circular polarization
            - "elliptical" - elliptically polarized (uses `axis_ratio`
              and `pol_angle`)

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
            Complex electric field components in k-space (2D arrays with
            the same shape as `KX` and `KY`).

        Notes
        -----
        The output is the angular spectrum representation of the focal
        field (Novotny & Hecht, eq. (3.47)), i.e. the field near the
        focus is ``E(r) = int E_k(k) exp(i k.r) d^2k``.  It therefore
        differs by a factor ``(2*pi)**2`` from the Fourier transform
        ``E(k) = int E(r) exp(-i k.r) d^2r``, which is taken into account
        in the ``..._pupil`` BLS signal functions.

        The amplitude of the incident beam is always Gaussian, given by
        the filling factor `f0`, with the polarization state set by
        `pol_type`.  Therefore, "radial" and "azimuthal" polarizations
        here correspond to a Gaussian beam with a polarization
        singularity on the optical axis (e.g. directly after a q-plate),
        whereas :meth:`getFocalFieldRad` and :meth:`getFocalFieldAzm`
        assume a doughnut beam with amplitude
        ``E0 * 2*rho/w0 * exp(-rho**2/w0**2)``.  The doughnut beam can
        be obtained here by weighting the Jones field by ``2*rho/w0``,
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

        k_vac = 2 * np.pi / self.wavelength  # Vacuum wave number
        k_medium = k_vac * n  # Medium wave number
        theta_max = np.arcsin(self.NA / n)  # Max focusing angle

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


def _simpson_points(n):
    """
    Number of points for Simpson's rule with error estimate: at least
    `n` (and at least 21), with the number of intervals divisible by 4,
    so that Simpson's rule can also be applied on every other point.
    """
    return 4 * int(np.ceil(max(n, 20) / 4)) + 1


def _simpson_weights(n, h):
    """Weights of the composite Simpson's rule for `n` (odd) points with step `h`."""
    w = np.full(n, 2.0)
    w[1::2] = 4.0
    w[0] = w[-1] = 1.0
    return w * h / 3
