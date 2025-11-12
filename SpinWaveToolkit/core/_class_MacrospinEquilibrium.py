"""
Core (private) file for the `MacrospinEquilibrium` class.
"""

import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
from SpinWaveToolkit.helpers import MU0, wrapAngle, sphr2cart, cart2sphr

__all__ = ["MacrospinEquilibrium"]


class MacrospinEquilibrium:
    """
    Compute magnetization equilibrium direction in a macrospin
    approximation.

    Can be used to find static equilibrium position before calculating
    the spin-wave dispersion relation in other classes.  See
    :doc:`/examples` for more.
        ### update exact example (e.g. a layer with PMA)

    Usually searches for a local equilibrium based on the initial
    position.  (The equilibrium is a minimum in the energy density
    landscape.)

    .. caution::

       The model might get stuck in labile equilibria, therefore it is
       encouraged to slightly perturb the angles (e.g. by 1 µrad) to get
       more reliable results from the calculations.

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
        last calculation.  (That is, given as None until first run.)
    verbose : bool
        Additional informative output to console?

    Methods
    -------
    add_uniaxial_anisotropy
    recalc_params
    minimize
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

    def minimize(self, scipy_kwargs=None, verbose=None):
        """Evaluate the minimization problem.

        Uses the `scipy.optimize.minimize` function to find the minimum
        in the 2D energy landscape E(theta, phi).

        Result is saved in the attributes `M` (just magnetization
        angles) and `res` (full output).

        Parameters
        ----------
        scipy_kwargs : dict or None, optional
            Dictionary with settings passed to
            :py:func:`scipy.optimize.minimize`.  Cannot contain ``tol``
            and ``bounds`` keywords, as they are fixly set here.
            If None, ``{"method": "Nelder-Mead"}`` is used.
            Defualt is None. Try changing the
            optimization method is you have concern about the results
            (see documentation of :py:func:`scipy.optimize.minimize`).
        verbose : bool or None, optional
            Additional informative output to console?  If None, the
            value is inferred from the ``verbose`` attribute.  Default
            is None.


        Notes
        -----
        The used minimize function usually finds a local minimum
        based on the initial conditions. This can be used for
        calculating hysteresis loops and field sweeps. For global
        minimum finding, a brute force method could be used, but it
        is not implemented here.

        """
        if verbose is None:
            verbose = self.verbose

        def fun(_x):
            """placeholder for energy evaluations"""
            return self.eval_energy(_x)

        m0 = (self.M["theta"] * 0.999 + 1e-3, self.M["phi"] * 0.999 + 1e-3)
        self.recalc_params()
        if scipy_kwargs is None:
            scipy_kwargs = {"method": "Nelder-Mead"}

        self.res = minimize(
            fun,
            m0,
            tol=1e-20,
            **scipy_kwargs,
            bounds=((-np.pi, 2 * np.pi), (-2 * np.pi, 4 * np.pi)),
        )
        if self.res.success and verbose:
            # print(f"Minimum successfully found after {self.res.nit} iterations.")
            print("Minimum successfully found.")
        elif verbose:
            print(f"Not converged.\n{self.res.message}")

        # save final state
        self.M["theta"], self.M["phi"] = wrapAngle(
            cart2sphr(*sphr2cart(*self.res.x))[:2]
        )
        eden = self.eval_energy(self.res.x, True)
        self.eden_zeeman, self.eden_demag, self.eden_anis_uni = eden

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

        Performs a numerical derivative of the energy density wrt.
        magnetization to get the effective field vector.

        Returns
        -------
        theta_Heff, phi_Heff : float
            (rad) polar and azimuthal angle of the effective field
            direction.
        Heff : float
            (T) effective field magnitude.
        """
        # define relative and aboslute (in radians) step in numerical derivative
        rstep, astep_rad = 1e-5, 1e-4
        # get central values
        Ms0 = float(self.Ms)
        theta0 = float(self.M["theta"])
        phi0 = float(self.M["phi"])

        # radial derivative
        self.Ms = Ms0 * (1 - rstep)
        eden_ms_i = self.eval_energy(self.__getM())
        self.Ms = Ms0 * (1 + rstep)
        eden_ms_f = self.eval_energy(self.__getM())
        self.Ms = Ms0  # reset Ms
        heff_ms = (eden_ms_i - eden_ms_f) / (Ms0 * 2 * rstep)  # negative sign included

        # theta derivative
        theta_i = theta0 * (1 - rstep) - astep_rad
        theta_f = theta0 * (1 + rstep) + astep_rad
        eden_th_i = self.eval_energy((theta_i, phi0))
        eden_th_f = self.eval_energy((theta_f, phi0))
        heff_th = 1 / Ms0 * (eden_th_i - eden_th_f) / (theta_f - theta_i)

        # phi derivative
        phi_i = phi0 * (1 - rstep) - astep_rad
        phi_f = phi0 * (1 + rstep) + astep_rad
        eden_ph_i = self.eval_energy((theta0, phi_i))
        eden_ph_f = self.eval_energy((theta0, phi_f))
        heff_ph = 1 / (Ms0 * np.sin(theta0)) * (eden_ph_i - eden_ph_f) / (phi_f - phi_i)

        # convert (heff_ms, heff_th, heff_ph) to cartesian coordinates
        heff_cart = sphr2cart(theta0, phi0) * heff_ms
        heff_cart += (
            np.array(
                [
                    np.cos(theta0) * np.cos(phi0),
                    np.cos(theta0) * np.sin(phi0),
                    -np.sin(theta0),
                ]
            )
            * heff_th
        )
        heff_cart += np.array([-np.sin(phi0), np.cos(phi0), 0.0]) * heff_ph

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
            :py:func:`scipy.optimize.minimize`.  Cannot contain ``tol``
            and ``bounds`` keywords, as they are fixly set here.
            If None, ``{"method": "Nelder-Mead"}`` is used.
            Defualt is None. Try changing the
            optimization method is you have concern about the results
            (see documentation of :py:func:`scipy.optimize.minimize`).
            This is not a sweepable parameter!


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
