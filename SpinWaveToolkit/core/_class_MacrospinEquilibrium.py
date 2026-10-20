"""
Core (private) file for the `MacrospinEquilibrium` class.
"""

import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
from SpinWaveToolkit.helpers import MU0, wrapAngle, sphr2cart, cart2sphr
from SpinWaveToolkit.core._sphere import (
    angle_frame,
    angle_hessian,
    classify,
    local_frame,
    memoize_last,
    minimize_kwargs,
    move,
    newton_spheres,
    project,
    tangent_bases,
)

__all__ = ["MacrospinEquilibrium"]


class MacrospinEquilibrium:
    """
    Compute magnetization equilibrium direction in a macrospin
    approximation.

    Can be used to find static equilibrium position before calculating
    the spin-wave dispersion relation in other classes.  See
    :doc:`/_example_nbs/macrospin_equilibrium_histloop` or
    :doc:`/_example_nbs/FMR_with_PMA` examples.

    Usually searches for a local equilibrium based on the initial
    position, see :meth:`minimize`.  (The equilibrium is a minimum in
    the energy density landscape.)  All equilibria, including the global
    minimum, can be found by :meth:`find_equilibria`.

    Includes the effect of the thin film's demagnetizing field (dipolar
    energy), external field (Zeeman energy) and any number of uniaxial
    anisotropies with arbitrary directions of the anisotropy axis.

    All angles are given in the laboratory (lab) frame of reference,
    i.e. `z || thin film normal`, and `x || IP projection of spin wave
    wavevector`.

    .. note::

       Cubic anisotropy is not implemented for now, as its usage is not
       very common in our experiments.

    Parameters
    ----------
    Ms : float
        (A/m) saturation magnetization of the magnetic material.
    Bext : float
        (T ) magnitude of external field.
    theta_H, phi_H : float
        (rad) polar and azimuthal angles of the external field
        direction.
    theta, phi : float or None
        (rad) polar and azimuthal angle of magnetization.  Serves as the
        starting point for ``minimize``.  If None, the values are
        inferred from `theta_H` and `phi_H`, respectively.  Default is
        None.
    demag : (3, 3) array or None, optional
        The demagnetization tensor in the lab frame.  If None,
        ``np.diag([0.0, 0.0, 1.0])`` is used, which corresponds to an
        infinite thin film in the xy plane.  Default is None.
    verbose: bool, optional
        Additional informative output to console?  Default is True.

    Attributes
    ----------
    Ms : float
        (A/m) saturation magnetization of the magnetic material.
    d : float
        (m ) thickness of the magnetic film.
    Bext : dict
        Dictionary containing the spherical coordinates of the external
        magnetic field.  It has keys {"Bext", "theta_H", "phi_H"}.
    M : dict
        Dictionary containing the spherical coordinates of
        magnetization.  It has keys {"theta", "phi"}.
    b : (3,) array
        Unit vector of external magnetic field.  Depends on `Bext`.
    demag : (3, 3) array
        The demagnetization tensor in the lab frame.
    anis : dict
        Dictionary of anisotropies.  Each (uniaxial) anisotropy has
        a unique name key, which can be then used to access its
        properties, which are stored as another dictionary with the
        following keys:

        - "Ku" - uniax. anisotropy strength (J/m^3)
        - "theta" - polar angle of anisotropy axis (rad)
        - "phi" - azimuthal angle of anisotropy axis (rad)
        - "Na" - corresponding tensor calculated from the three
          parameters above.

    Na_tot : (3, 3) array
        Sum of all anisotropy tensors.  Depends on `anis`.
    eden_zeeman, eden_demag, eden_anis_uni : float
        (J/m^3) Zeeman, dipolar, and uniax. anisotropy energy density,
        respectively.  Stored as results of the last `minimize()` call.
    res : scipy.optimize.OptimizeResult
        Full output of the `scipy.optimize.minimize` function from the
        last calculation (see Notes of :meth:`minimize`).  (That is,
        given as None until first run.)
    verbose : bool
        Additional informative output to console?

    Methods
    -------
    add_uniaxial_anisotropy
    recalc_params
    minimize
    find_equilibria
    eval_energy
    getHeff
    hysteresis

    Examples
    --------

    .. code-block:: python

        maceq = MacrospinEquilibrium(
            Ms=800e3, Bext=150e-3, theta_H=np.deg2rad(10),
            phi_H=np.deg2rad(60), theta=0, phi=np.deg2rad(30)
        )
        maceq.add_uniaxial_anisotropy("uni0", Ku=15e3, theta=0, phi=0)
        maceq.add_uniaxial_anisotropy("uni1", Ku=10e3,
            theta=np.deg2rad(70), phi=np.pi/2)
        maceq.minimize()
        print(maceq.M)

    See also
    --------
    SingleLayer, SingleLayerNumeric, Material

    """

    def __init__(
        self,
        Ms,
        Bext,
        theta_H,
        phi_H,
        theta=None,
        phi=None,
        demag=None,
        verbose=True,
    ):
        self.Ms = Ms

        self.M = {
            "theta": theta if theta is not None else theta_H,
            "phi": phi if phi is not None else phi_H,
        }
        self.Bext = {
            "Bext": Bext,
            "theta_H": theta_H,
            "phi_H": phi_H,
        }
        self.b = sphr2cart(theta_H, phi_H)  # unit vector of Bext in lab frame
        self.demag = demag if demag is not None else np.diag([0.0, 0.0, 1.0])
        self.verbose = verbose
        # preallocations
        self.anis = {}
        self.Na_tot = np.zeros((3, 3), dtype=np.float64)
        self.eden_zeeman, self.eden_demag, self.eden_anis_uni = 0.0, 0.0, 0.0
        self.res = None  # full `scipy.optimize.minimize` output
        self.__check_angle(
            (
                theta_H,
                phi_H,
                theta if theta is not None else 0,
                phi if phi is not None else 0,
            )
        )

    def recalc_params(self):
        """Recalculate/update the values of dependent variables, e.g.
        field unit vector and total anistropy tensor.
        """
        self.Na_tot = np.sum([i["Na"] for _, i in self.anis.items()], 0)
        self.b = sphr2cart(self.Bext["theta_H"], self.Bext["phi_H"])

    def add_uniaxial_anisotropy(
        self, name, Ku=0.0, theta=0.0, phi=0.0, Bani=None, Na=None
    ):
        """Add uniaxial anisotropy to the system.  Only the first order
        constant is assumed.

        Can be used to update an already defined anisotropy by referring
        to the same name.

        Parameters
        ----------
        name : str
            String name to use as key in the dictionary of anisotropies.
            If an anisotropy with the same name already exists, it is
            silently overwritten.
        Ku : float, optional
            (J/m^3) uniaxial anisotropy constant.  Easy plane anisotropy
            for Ku < 0 and easy axis for Ku > 0.  Unused if `Na` or
            `Bani` is specified.
        theta : float, optional
            (rad) polar angle of the anisotropy axis in the lab frame.
            Unused if `Na` is specified.
        phi : float, optional
            (rad) azimuthal angle of the anisotropy axis in the lab
            frame.  Unused if `Na` is specified.
        Bani : float, optional
            (T ) uniaxial anisotropy field.  If specified (and non-zero),
            `Ku` input is not used, but rather is recalculated from
            `Bani` as ``Ku = Ms*Bani/2``.  Default is None.
        Na : (3, 3) array or None, optional
            () uniaxial anisotropy tensor.  Can be used for direct
            assignment.  However, when used, the other parameters are
            not recalculated into the anisotropy dict.  Default is None.

        """
        self.__check_angle((theta, phi))
        u = sphr2cart(theta, phi)
        Ku = Ku if (Bani is None or Bani == 0) else Bani * self.Ms / 2
        # The minus is here (below), bcs Na has to have opposite sign than Ku,
        # see Skomski (2008) Simple models of magnetism, sec. 3.1.2, pp. 77-78.
        Na = -np.outer(u, u) * (2 * Ku / (MU0 * self.Ms**2)) if Na is None else Na
        self.anis[name] = {"Ku": Ku, "theta": theta, "phi": phi, "Na": Na}
        self.recalc_params()

    def minimize(self, scipy_kwargs=None, verbose=None, global_minimum=False):
        """Find the equilibrium (local minimum of the energy density)
        closest to the current magnetization direction.

        Uses the :py:func:`scipy.optimize.minimize` function with the
        analytical gradient and Hessian of the energy density, see
        Notes.  The current magnetization direction (attribute `M`) is
        the starting point.

        Result is saved in the attributes `M` (just magnetization
        angles) and `res` (full output).

        Parameters
        ----------
        scipy_kwargs : dict or None, optional
            Dictionary with settings passed to
            :py:func:`scipy.optimize.minimize`, e.g. ``{"method":
            "Nelder-Mead"}``.  Cannot contain ``bounds`` (the angles are
            periodic and are not bounded).  The analytical gradient
            (``jac``) and Hessian (``hess``) are passed automatically to
            the methods that use them.  If ``tol`` and ``options`` are
            not given, tolerances for the precision of about 1e-8 rad
            are used.  If None, ``{"method": "trust-exact"}`` is used
            with the trust radius limited to 0.3 rad (see Notes).
            Default is None.  Try changing the optimization method if
            you have concern about the results (see documentation of
            :py:func:`scipy.optimize.minimize`).
        verbose : bool or None, optional
            Additional informative output to console?  If None, the
            value is inferred from the ``verbose`` attribute.  Default
            is None.
        global_minimum : bool, optional
            If True, the global minimum of the energy density is found
            instead (using :meth:`find_equilibria`, independent of the
            current magnetization direction and of `scipy_kwargs`), the
            magnetization angles are saved to that minimum and `res`
            is set to None.  Default is False.

        Notes
        -----
        The energy density is a quadratic function of the magnetization
        unit vector `m`, ``E = -Ms*Bext*(b.m) + MU0*Ms**2/2 * m.N.m``,
        where ``N`` is the sum of the demagnetization and anisotropy
        tensors.  Its gradient and Hessian are therefore evaluated
        analytically.  The minimization runs on the energy divided by
        ``MU0*Ms**2``, so that the tolerances do not depend on the
        material.

        The default trust-region method with a small trust radius (the
        steps are at most 0.3 rad, initially 0.1 rad) follows the energy
        landscape from the starting point to the local minimum of its
        basin, similarly to the (overdamped) magnetization dynamics.
        This is important for hysteresis loops, where the magnetization
        has to stay in a metastable minimum until it vanishes.  Methods
        with line searches (e.g. L-BFGS-B or BFGS) are similarly fast,
        but their first steps can jump over an energy barrier into
        another minimum, i.e. the magnetization switches too early (by
        about 20 % for the Stoner-Wohlfarth model).

        To avoid the singularity of the spherical coordinates at the
        poles (``theta = 0`` and ``pi``), the angles used in the
        minimization are measured in a local frame, in which the
        starting point lies on the equator.  (If the minimum is close to
        a pole of the local frame, the minimization is repeated in a new
        local frame.)  Therefore, the ``x``, ``jac`` and ``hess``
        attributes of `res` refer to these local angles, except
        ``res.x``, which is converted back to ``(theta, phi)`` in the
        lab frame.  If SciPy stops at the round-off level of the
        gradient before reaching its tolerance, ``res.success`` is set
        to True (with a note in ``res.message``).

        After the minimization, the stability of the result is checked
        by the Hessian.  If the result is a saddle point or a maximum
        (labile equilibrium, e.g. when starting exactly in a symmetric
        position), the magnetization is displaced by 1 mrad along the
        direction of the negative curvature and the minimization is
        repeated.  The displacement is chosen with a positive z
        component (or y, x if the z component is zero), so that the
        result is deterministic.
        """
        if verbose is None:
            verbose = self.verbose
        self.recalc_params()

        if global_minimum:
            theta, phi, _, eq_type = self.find_equilibria()
            i = np.argmax(eq_type == 0)  # lowest energy local minimum
            self.M["theta"], self.M["phi"] = theta[i], phi[i]
            self.res = None
            if verbose:
                print("Global minimum found.")
            self.__save_energies()
            return

        h, A = self.__reduced_params()
        scale = max(1.0, np.abs(A).max(), np.linalg.norm(h))
        grad_hess = _sphere_functions(h, A)
        m = sphr2cart(self.M["theta"], self.M["phi"])
        for _ in range(10):  # restarts in a new local frame or from saddle points
            R = local_frame(m)
            fun, jac, hess = _energy_functions(R.T @ h, R.T @ A @ R)
            kwargs = minimize_kwargs(scipy_kwargs, jac, hess)
            res = minimize(fun, np.array([np.pi / 2, 0.0]), **kwargs)
            m = R @ sphr2cart(*res.x)
            # polish by Newton steps on the sphere (SciPy stops at the
            # round-off level of the energy, ~1e-7 rad)
            (mn,), gnorm = newton_spheres([m[:, None]], grad_hess, 5, scale)
            if gnorm[0] < 1e-9 * scale and mn[:, 0] @ m > np.cos(1e-6):
                m = mn[:, 0]
            if abs(res.x[0] - np.pi / 2) > np.pi / 3:
                continue  # close to a pole of the local frame, repeat around m
            # stability check (Hessian on the sphere)
            T = tangent_bases(m[:, None])
            g, H = grad_hess([m[:, None]], [T])
            lam, vec = np.linalg.eigh(H[:, :, 0])
            if lam[0] >= -1e-9 * scale:
                break
            s = T[:, :, 0] @ vec[:, 0]  # direction of negative curvature
            for c in (2, 1, 0):  # deterministic sign
                if abs(s[c]) > 1e-9:
                    s *= np.sign(s[c])
                    break
            m = move(m, s, 1e-3)
        else:  # (no stable minimum within the restarts: gradient at the last m)
            g = tangent_bases(m[:, None])[:, :, 0].T @ (A @ m - h)
        res.x = np.array(cart2sphr(*m)[:2])
        # SciPy may stop at the round-off level of the gradient before
        # reaching its tolerance, such a result is accepted
        if not res.success and np.linalg.norm(g) < 1e-7 * scale:
            # (limited by the round-off of the energy)
            res.success = True
            res.message += " (Accepted, the gradient is at the round-off level.)"
        self.res = res

        if self.res.success and verbose:
            print("Minimum successfully found.")
        elif verbose:
            print(f"Not converged.\n{self.res.message}")

        # save final state
        self.M["theta"], self.M["phi"] = wrapAngle(res.x)
        self.__save_energies()

    def find_equilibria(self):
        """Find all equilibrium directions of the magnetization.

        The equilibria are the stationary points of the energy density
        on the unit sphere, i.e. the directions where the torque of the
        effective field vanishes.  For the quadratic energy density used
        here (see Notes of :meth:`minimize`), there are at most six
        isolated ones, which are found directly (without iterations from
        a starting point).  Does not change the state of the object.

        Returns
        -------
        theta, phi : ndarray
            (rad) polar and azimuthal angles of the equilibria in the
            lab frame, sorted by increasing energy density, i.e. the
            first one is the global minimum.
        eden : ndarray
            (J/m^3) energy density of the equilibria.
        type : ndarray[int]
            Type of the equilibria: 0 for (local) minima, i.e. stable
            equilibria, 1 for maxima and 2 for saddle points.

        Notes
        -----
        A stationary point ``m`` satisfies ``(N - lam*I) m = h`` for a
        Lagrange multiplier ``lam``, where ``N`` is the sum of the
        demagnetization and anisotropy tensors and
        ``h = Bext*b/(MU0*Ms)``.  Together with ``|m| = 1``, it gives a
        polynomial equation of the 6th degree in ``lam``, which is
        solved as an eigenvalue problem of a 6x6 matrix (Gander,
        Golub & von Matt, Linear Algebra Appl. 114, 815 (1989)).
        The solutions are refined by Newton iterations on the sphere and
        classified by the eigenvalues of the Hessian (both positive for
        a minimum, both negative for a maximum).  In degenerate cases (e.g. a field
        along the normal of a film without in-plane anisotropy), the
        equilibria are not isolated and only some representatives are
        returned.
        """
        self.recalc_params()
        h, A = self.__reduced_params()
        scale = max(1.0, np.abs(A).max(), np.linalg.norm(h))
        grad_hess = _sphere_functions(h, A)

        # candidates from the quadratic eigenvalue problem
        # (A - lam)^2 g = h h^T g, linearized with d = (A - lam) g
        M = np.block([[A, -np.eye(3)], [-np.outer(h, h), A]])
        lams, vecs = np.linalg.eig(M)
        cands = []
        for lam, v in zip(lams, vecs.T):
            if abs(lam.imag) > 1e-6 * scale:
                continue
            v = (v / v[np.argmax(np.abs(v))]).real  # remove the complex phase
            d = v[3:]  # m is parallel to d (both signs are refined below)
            if np.linalg.norm(d) > 1e-12:
                cands += [d, -d]
        # eigenvectors of A (the equilibria for zero field) as more seeds
        _, V = np.linalg.eigh(A)
        cands += list(V.T) + list(-V.T)
        if np.linalg.norm(h) > 0:
            cands += [h, -h]
        cands = np.array(cands).T
        cands /= np.linalg.norm(cands, axis=0)

        # refine all candidates at once and remove the duplicates
        (m,), gnorm = newton_spheres([cands], grad_hess, 50, scale)
        found = []
        for c in m[:, gnorm < 1e-9 * scale].T:
            if not any(c @ f > 1 - 1e-12 for f in found):
                found.append(c)
        found = np.array(found).T  # (3, M)

        eden_red = -h @ found + 0.5 * np.einsum("in,ij,jn->n", found, A, found)
        _, H = grad_hess([found], [tangent_bases(found)])
        eq_type = classify(H, scale)
        order = np.argsort(eden_red)
        theta, phi = cart2sphr(*found[:, order])[:2]
        return (
            theta,
            wrapAngle(phi),
            eden_red[order] * MU0 * self.Ms**2,
            eq_type[order],
        )

    def eval_energy(self, m, components=False):
        """Evaluate the energy density for the magnetization direction
        angles ``m = (theta, phi)``.

        Returns
        -------
        eden : float or list[float]
            (J/m^3) energy density. If `components` is False, given as
            a sum of all components (float). Returns a list of
            components otherwise.
        """
        m = sphr2cart(*m)
        # Zeeman
        eZ = -self.Ms * self.Bext["Bext"] * float(np.dot(m, self.b))

        # demag
        ed = 0.5 * MU0 * self.Ms**2 * float(m @ self.demag @ m)

        # uniaxial anisotropies
        if not self.anis:  # when self.anis is an empty dict
            ea_uni = 0
        else:
            ea_uni = 0.5 * MU0 * self.Ms**2 * float(m @ self.Na_tot @ m)

        return [eZ, ed, ea_uni] if components else eZ + ed + ea_uni

    def getHeff(self):
        """Calculate effective field with the current magnetization
        direction.

        The effective field is the negative derivative of the energy
        density with respect to the magnetization vector,
        ``Beff = Bext*b - MU0*Ms*N.m``, where ``N`` is the sum of the
        demagnetization and anisotropy tensors (evaluated
        analytically).

        Returns
        -------
        theta_Heff, phi_Heff : float
            (rad) polar and azimuthal angle of the effective field
            direction.
        Heff : float
            (T) effective field magnitude.
        """
        self.recalc_params()
        m = sphr2cart(self.M["theta"], self.M["phi"])
        N = self.demag + (self.Na_tot if self.anis else 0)
        N = 0.5 * (N + N.T)  # (only the symmetric part contributes to the energy)
        heff_cart = self.Bext["Bext"] * self.b - MU0 * self.Ms * (N @ m)

        # calculate magnitude and spherical angles
        theta_Heff, phi_Heff, heff_mag = cart2sphr(*heff_cart)

        return theta_Heff, phi_Heff, heff_mag

    def hysteresis(self, Bext, theta_H, phi_H, scipy_kwargs=None):
        """Calculate a hysteresis curve from a vector of swept field
        values.

        Any of the input (field-related) values can be swept, but the
        other two have to be either of same shape or floats.

        Parameters
        ----------
        Bext : float or (N,) array
            (T ) amplitude of external magnetic field (can be negative).
        theta_H : float or (N,) array
            (rad) polar angle of external magnetic field.
        phi_H : float or (N,) array
            (rad) azimuthal angle of external magnetic field.
        scipy_kwargs : dict or None, optional
            Dictionary with settings passed to
            :py:func:`scipy.optimize.minimize`, see :meth:`minimize`.
            If None, the default of :meth:`minimize` is used.  Default
            is None.  This is not a sweepable parameter!


        Returns
        -------
        theta, phi : (N,) array
            (rad) angles related to magnetization direction at each
            point of the sweep.


        Notes
        -----
        To get a projection of magnetization in a certain direction,
        you can just put the output of this method to the `sphr2cart`
        function provided by `SpinWaveToolkit`.

        For sweeping other parameters, the user is encouraged to write
        their own script similarly to this method.  Currently, we
        do not plan to implement a general sweep.
        """
        n = np.shape(Bext + theta_H + phi_H)[0]
        ones = np.ones(n)
        Bext, theta_H, phi_H = ones * Bext, ones * theta_H, ones * phi_H
        theta, phi = np.empty(n), np.empty(n)

        for i in tqdm(range(n), ascii=True, ncols=80):
            self.Bext["Bext"] = Bext[i]
            self.Bext["theta_H"] = theta_H[i]
            self.Bext["phi_H"] = phi_H[i]
            self.minimize(scipy_kwargs=scipy_kwargs, verbose=False)
            theta[i], phi[i] = self.__getM()

        return theta, phi

    def __reduced_params(self):
        """Reduced field ``h`` and tensor ``A`` of the energy density
        divided by ``MU0*Ms**2``, i.e.
        ``E/(MU0*Ms**2) = -h.m + m.A.m/2``.
        """
        h = self.Bext["Bext"] * self.b / (MU0 * self.Ms)
        A = self.demag + (self.Na_tot if self.anis else 0)
        return h, 0.5 * (A + A.T)

    def __save_energies(self):
        """Save the energy density components of the current state."""
        eden = self.eval_energy(self.__getM(), True)
        self.eden_zeeman, self.eden_demag, self.eden_anis_uni = eden

    def __check_angle(self, angle):
        """Function for checking the angles are in correct units.

        The motivation for this is that users often forget to convert
        to radians and input ridiculously large values.

        Parameters
        ----------
        angle : float or array_like
            (rad) angles to be checked.
        """
        angle = np.atleast_1d(angle)
        if self.verbose and np.any(angle > 2 * np.pi):
            print(
                "Input angle larger than 2pi rad. Make sure"
                + " all input angle values are in radians!"
            )

    def __getM(self):
        """Returns magnetization vector data as ``(theta, phi)``."""
        return self.M["theta"], self.M["phi"]


def _energy_functions(h, A):
    """Reduced energy density ``-h.m + m.A.m/2`` of ``m(theta, phi)``
    and its analytical gradient and Hessian with respect to the angles.
    """

    frame = memoize_last(lambda x: angle_frame(*x))

    def fun(x):
        m = frame(x)[0]
        return -h @ m + 0.5 * m @ A @ m

    def jac(x):
        m, J, _ = frame(x)
        return J.T @ (A @ m - h)

    def hess(x):
        m, J, D = frame(x)
        return angle_hessian(J, D, A @ m - h, A)

    return fun, jac, hess


def _sphere_functions(h, A):
    """Gradient and Hessian of the reduced energy ``-h.m + m.A.m/2`` on
    the unit sphere in the tangent bases, as used by `newton_spheres`."""

    def grad_hess(ms, Ts):
        return project(ms, Ts, [A @ ms[0] - h[:, None]], [[A]])

    return grad_hess
