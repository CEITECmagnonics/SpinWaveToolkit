"""
Core (private) file for the `MacrospinEquilibrium` class.
"""

import numpy as np
from SpinWaveToolkit.helpers import MU0, wrapAngle, sphr2cart, cart2sphr
from scipy.optimize import minimize

__all__ = ["MacrospinEquilibrium"]


class MacrospinEquilibrium:
    """
    Compute magnetization equilibrium direction in a macrospin 
    approximation.

    Usually searches for a local equilibrium based on the initial 
    position.

    Includes the effect of the thin film's demagnetizing field (dipolar 
    energy), external field (Zeeman energy) and any number of uniaxial 
    anisotropies with arbitrary directions of the anisotropy axis.

    All angles are given in the laboratory frame of reference, i.e.
    `z || thin film normal`, and `x || IP projection of spin wave 
    wavevector`.
    
    .. note:: 
    
       Cubic anisotropy is not implemented for now, as its usage is not 
       very common in our experiments.

    Parameters
    ----------
    Ms : float
        (A/m) saturation magnetization of the magnetic material.
    d : float
        (m) thickness of the magnetic film.
    Bext : float
        (T) magnitude of external field.
    theta_H, phi_H : float
        (rad) polar and azimuthal angles of the external field 
        direction.
    theta, phi : float

    Attributes
    ----------
    Ms : float
        (A/m) saturation magnetization of the magnetic material.
    d : float
        (m) thickness of the magnetic film.
    Bext : dict
        Dictionary containing the spherical coordinates of the external 
        magnetic field.  It has keys {"Bext", "theta_H", "phi_H"}.
    M : dict

    Methods
    -------

    Examples
    --------

    See also
    --------
    SingleLayer, SingleLayerNumeric, Material

    """

    def __init__(
            self,
            Ms,
            d, 
            Bext, 
            theta_H, 
            phi_H,
            theta=None,
            phi=None,
            demag=None
        ):
        self.Ms = Ms
        self.d = d

        self.M = {
            "theta": theta if theta is not None else theta_H,
            "phi": phi if phi is not None else phi_H,
        }
        self.Bext = {
            "Bext": Bext,
            "theta_H": theta_H,
            "phi_H": phi_H,
        }
        self.b = sphr2cart(theta_H, phi_H)  # unit vector of Bext in lab ### remake this similarly to Na_tot, so that it is always calculated from the current Bext (maybe too computationally demanding. Ask chatgpt what to do.)
        self.demag = demag if demag is not None else np.diag([0.0, 0.0, 1.0])
        # preallocations
        self.anis = {}
        self._Na_tot = np.zeros((3, 3), dtype=np.float64)
        self.eden_zeeman, self.eden_demag, self.eden_anis_uni = 0.0, 0.0, 0.0
        self.res = None  # full `scipy.optimize.minimize` output
    
    @property
    def Na_tot(self):
        """Summed tensors of all anisotropies."""
        return np.sum([self.anis[i]["Na"] for i in self.anis.keys()], 0)
    
    @Na_tot.setter
    def Na_tot(self, val):
        raise TypeError("Na_tot cannot be assigned any values! "
                        + "Define individual anisotropies instead.")
    
    def add_uniaxial_anisotropy(self, name, Ku=0.0, theta=0.0, phi=0.0, Na=None):
        """Add uniaxial anisotropy to the system.  Only the first order 
        constant is assumed.
        
        Parameters
        ----------
        name : str
            String name to use as key in the dictionary of anisotropies.
            If an anisotropy with the same name already exists, it is 
            silently overwritten.
        Ku : float
            (J/m^3) uniaxial anisotropy constant.  Easy plane anisotropy 
            for Ku > 0 and easy axis for Ku < 0.  Unused if `tensor` is 
            specified.
        theta : float
            (rad) polar angle of the anisotropy axis in the lab frame.
            Unused if `tensor` is specified.
        phi : float
            (rad) azimuthal angle of the anisotropy axis in the lab 
            frame.  Unused if `tensor` is specified.
        Na : (3, 3) array or None, optional
            Uniaxial anisotropy tensor.  Can be used for direct 
            assignment.  However, when used, the other parameters are 
            not recalculated into the anisotropy dict.  Default is None.

        """
        u = sphr2cart(theta, phi)
        Na = np.outer(u, u) * (2 * Ku / (MU0 * self.Ms**2))
        self.anis[name] = {"Ku": Ku, "theta": theta, "phi": phi, "Na": Na}

    def minimize(self, scipy_kwargs={}):
        """Evaluate the minimization problem.
         
        Uses the `scipy.optimize.minimize` function to find the minimum
        in the 2D energy landscape E(theta, phi).

        Result is saved in the attributes `M` (just magnetization 
        angles) and `res` (full output).

        Parameters
        ----------
        scipy_kwargs : dict, optional
            dictionary with settings passed to 
            `scipy.optimize.minimize`.  If a ``"constraints"`` setting 
            key is used, its value must be a list of constraints (see
            documentation of :py:func:`scipy.optimize.minimize`).
        
        .. note::
           
           The used minimize function usually finds a local minimum 
           based on the initial conditions. This can be used for 
           calculating hysteresis loops and field sweeps. For global 
           minimum finding, a brute force method could be used, but it 
           is not implemented here.
        
        """

        def fun(_x):
            """placeholder for energy evaluations"""
            return self.eval_energy(_x)
        
        m0 = sphr2cart(self.M["theta"], self.M["phi"])
        cons = [{'type': 'eq', 'fun': lambda _m: np.dot(_m, _m) - 1.0}]
        if "constraints" in scipy_kwargs.keys():
            scipy_kwargs["constraints"] += cons
        else:
            scipy_kwargs["constraints"] = cons

        self.res = minimize(fun, m0, constraints=cons, **scipy_kwargs)
        if self.res.success:
            # print(f"Minimum successfully found after {self.res.nit} iterations.")
            print(f"Minimum successfully found.")
        else:
            print(f"Not converged.\n{self.res.message}")

        # save final state
        self.M["theta"], self.M["phi"], _ = cart2sphr(*self.res.x)
        eden = self.eval_energy(self.res.x, True)
        self.eden_zeeman, self.eden_demag, self.eden_anis_uni = eden
    
    def eval_energy(self, m, components=False):
        """Evaluate the energy density for the magnetization unit 
        vector `m`.

        Returns 
        -------
        eden : float or list[float]
            (J/m^3) energy density. If `components` is False, given as 
            a sum of all components (float). Returns a list of 
            components otherwise.
        """
        Bext = self.Bext["Bext"]

        # Zeeman
        eZ = - self.Ms * Bext * float(np.dot(m, self.b))

        # demag
        ed = 0.5 * MU0 * self.Ms**2 * float(m @ self.demag @ m)

        # uniaxial anisotropies
        ea_uni = 0.5 * MU0 * self.Ms**2 * float(m @ self.Na_tot @ m)

        return [eZ, ed, ea_uni] if components else eZ + ed + ea_uni
