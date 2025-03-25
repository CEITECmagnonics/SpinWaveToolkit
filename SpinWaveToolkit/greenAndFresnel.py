"""
functions: fresnel_and_green.py

This module implements:

1. fresnel_coefficients: Computes Fresnel transmission coefficients for 
   p- and s-polarized waves as a function of lateral wavevector q.
   
   The function accepts:
     - lambda_     : wavelength (in meters)
     - DF          : array of dielectric functions (complex) for each layer,
                     ordered from the top (superstrate) to bottom.
     - PM          : array of permeabilities for each layer (usually ones)
     - d           : array of thicknesses for the layers between the superstrate
                     and substrate (length should be len(DF)-2)
     - source_layer_index: (1-indexed) index of the layer where the source is located.
     - output_layer_index: (1-indexed) index of the layer where the output is desired.
     
   It returns two functions, htp and hts, which when called with a lateral wavevector q
   (scalar or numpy array) return the corresponding Fresnel transmission coefficient(s)
   for p- and s-polarization, respectively.
   
2. sph_green_function: Computes the spherical Green’s functions for p- and s-polarized fields,
   given the lateral wavevector components (Kx, Ky), the dielectric function of a target layer,
   wavelength, and the Fresnel coefficients (tp and ts). The outputs pGF and sGF are provided
   as 3×2 lists (each entry a numpy array of the same shape as Kx and Ky).
   
Note:
    - The code uses numpy for numerical operations.
    - The matrices (2×2) for propagation and interface relations are represented as NumPy arrays.
    - Indices in the code assume that the user supplies layer indices in a 1-indexed manner,
      similar to the MATLAB version.
"""

import numpy as np

def fresnel_coefficients(lambda_, DF, PM, d, source_layer_index, output_layer_index):
    """
    Compute Fresnel coefficients for p- and s-polarized waves as a function of lateral wavevector q.

    Parameters
    ----------
    lambda_ : float
        Wavelength of the calculated light (in meters).
    DF : array_like
        Array of dielectric functions (complex) for each layer, ordered from top (superstrate) downward.
    PM : array_like
        Array of permeabilities for each layer (typically ones).
    d : array_like
        Array of thicknesses for the layers between the superstrate and substrate.
        (Length should be len(DF) - 2.)
    source_layer_index : int
        (1-indexed) Index of the layer where the source (induced polarization) is located.
    output_layer_index : int
        (1-indexed) Index of the layer where the output is desired.

    Returns
    -------
    htp : function
        A function that computes the p-polarized Fresnel transmission coefficients given q.
    hts : function
        A function that computes the s-polarized Fresnel transmission coefficients given q.
    """
    # Convert inputs to numpy arrays (ensure complex type for DF and PM)
    DF = np.asarray(DF, dtype=complex)
    PM = np.asarray(PM, dtype=complex)
    d = np.asarray(d, dtype=complex)

    N = len(DF)
    # Compute the wavenumber in each layer: kn = 2π/λ * sqrt(DF * PM)
    kn = 2 * np.pi / lambda_ * np.sqrt(DF * PM)

    # ------------------------ p-polarization ------------------------
    def htp(q):
        """
        Compute p-polarized Fresnel coefficients for a given lateral wavevector q.

        Parameters
        ----------
        q : float or array_like
            Lateral wavevector component(s).

        Returns
        -------
        tp_out : np.ndarray
            Fresnel transmission coefficient(s) for p-polarization.
            The shape (either a 1×2 array or a 2×2 array) depends on the chosen output layer.
        """
        # Compute the z-component in each layer (as a function of q)
        knz = [np.sqrt(kn[i]**2 - q**2) for i in range(N)]
        
        # Compute reflection (rp) and transmission (tp) coefficients at interfaces (from layer i to i+1)
        rpij = []
        tpij = []
        rpji = []
        tpji = []
        for i in range(N-1):
            tmp   = DF[i+1] * knz[i]
            tmp2  = DF[i]   * knz[i+1]
            tmp3  = tmp + tmp2
            rpij.append((tmp - tmp2) / tmp3)
            tpij.append(2 * tmp / tmp3 * np.sqrt((PM[i+1] * DF[i]) / (PM[i] * DF[i+1])))
            rpji.append(-rpij[-1])
            tpji.append(2 * tmp2 / tmp3 * np.sqrt((PM[i] * DF[i+1]) / (PM[i+1] * DF[i])))

        # Build propagation matrices for layers 2 to N-1 (indices 1 to N-2)
        Pj = []
        PjInv = []
        for i in range(1, N-1):
            if np.isinf(d[i-1]):
                # For infinite thickness, force interface reflection to zero and use identity matrix.
                rpij[i] = 0
                P = np.array([[1, 0],
                              [0, 1]], dtype=complex)
                Pj.append(P)
                PjInv.append(P)
            else:
                phase = np.exp(1j * knz[i] * d[i-1])
                P = np.array([[phase, 0],
                              [0, 1/phase]], dtype=complex)
                Pj.append(P)
                # Inverse propagation matrix: swap the diagonal entries.
                P_inv = np.array([[1/phase, 0],
                                  [0, phase]], dtype=complex)
                PjInv.append(P_inv)

        # Compute interface matrices Mij and Mji at each interface
        Mij = []
        Mji = []
        for i in range(N-1):
            tmp_val = tpij[i] * tpji[i] - rpij[i] * rpji[i]
            Mji.append(np.array([[1, -rpij[i]],
                                 [rpji[i], tmp_val]], dtype=complex))
            Mij.append(np.array([[tmp_val, rpij[i]],
                                 [-rpji[i], 1]], dtype=complex))

        # Upward propagation: from top layer to source layer.
        MUp = np.eye(2, dtype=complex)
        FactorUp = 1
        for i in range(source_layer_index - 1):
            MUp = MUp @ Mji[i] @ PjInv[i]
            FactorUp *= tpji[i]

        # Downward propagation: from bottom (last interface) upward to source layer.
        MDown = Mij[-1]
        FactorDown = tpij[-1]
        # Loop indices: from (N-2) downto source_layer_index (adjusted for 0-indexing)
        for i in range(N-2 - 1, source_layer_index - 1 - 1, -1):
            MDown = MDown @ Pj[i] @ Mij[i]
            FactorDown *= tpij[i]

        tmp_val = MDown[1, 1] * MUp[0, 0] - MDown[0, 1] * MUp[1, 0]

        # Branch according to the desired output layer.
        if output_layer_index == 1:
            # Output at the superstrate (top layer)
            tp_out = np.array([FactorUp * MDown[1, 1] / tmp_val,
                               FactorUp * MDown[0, 1] / tmp_val], dtype=complex)
        elif output_layer_index == N:
            # Output at the bottom substrate
            tp_out = np.array([FactorDown * MUp[1, 0] / tmp_val,
                               FactorDown * MUp[0, 0] / tmp_val], dtype=complex)
        elif output_layer_index == source_layer_index:
            tp_out = np.array([[MUp[1, 0] * MDown[0, 1] / tmp_val, MUp[0, 0] * MDown[0, 1] / tmp_val],
                               [MUp[1, 0] * MDown[1, 1] / tmp_val, MUp[1, 0] * MDown[0, 1] / tmp_val]], dtype=complex)
        else:
            if output_layer_index < source_layer_index:
                L = np.eye(2, dtype=complex)
                for i in range(output_layer_index - 1):
                    L = L @ Mji[i] @ PjInv[i]
                Factor = 1
                for i in range(output_layer_index - 1, source_layer_index - 1):
                    Factor *= tpji[i]
                tp_out = np.array([[Factor * L[0, 0] * MDown[1, 1] / tmp_val, Factor * L[0, 0] * MDown[0, 1] / tmp_val],
                                   [Factor * L[1, 0] * MDown[1, 1] / tmp_val, Factor * L[1, 0] * MDown[0, 1] / tmp_val]], dtype=complex)
            elif output_layer_index > source_layer_index:
                L = Mij[-1]
                for i in range(N-2 - 1, output_layer_index - 1 - 1, -1):
                    L = L @ Pj[i] @ Mij[i]
                Factor = 1
                for i in reversed(range(source_layer_index - 1, output_layer_index - 1)):
                    Factor *= tpij[i]
                tp_out = np.array([[Factor * L[0, 1] * MUp[1, 0] / tmp_val, Factor * L[0, 1] * MUp[0, 0] / tmp_val],
                                   [Factor * L[1, 1] * MUp[1, 0] / tmp_val, Factor * L[1, 1] * MUp[0, 0] / tmp_val]], dtype=complex)
            else:
                tp_out = None  # Should not occur
        return tp_out

    # ------------------------ s-polarization ------------------------
    def hts(q):
        """
        Compute s-polarized Fresnel coefficients for a given lateral wavevector q.

        Parameters
        ----------
        q : float or array_like
            Lateral wavevector component(s).

        Returns
        -------
        ts_out : np.ndarray
            Fresnel transmission coefficient(s) for s-polarization.
            The shape (either a 1×2 array or a 2×2 array) depends on the chosen output layer.
        """
        knz = [np.sqrt(kn[i]**2 - q**2) for i in range(N)]
        
        rsij = []
        tsij = []
        rsji = []
        tsji = []
        for i in range(N-1):
            tmp   = PM[i+1] * knz[i]
            tmp2  = PM[i]   * knz[i+1]
            tmp3  = tmp + tmp2
            rsij.append((tmp - tmp2) / tmp3)
            tsij.append(2 * tmp / tmp3)
            rsji.append(-rsij[-1])
            tsji.append(2 * tmp2 / tmp3)
        
        Pj = []
        PjInv = []
        for i in range(1, N-1):
            if np.isinf(d[i-1]):
                rsij[i] = 0
                P = np.array([[1, 0],
                              [0, 1]], dtype=complex)
                Pj.append(P)
                PjInv.append(P)
            else:
                phase = np.exp(1j * knz[i] * d[i-1])
                P = np.array([[phase, 0],
                              [0, 1/phase]], dtype=complex)
                Pj.append(P)
                P_inv = np.array([[1/phase, 0],
                                  [0, phase]], dtype=complex)
                PjInv.append(P_inv)
        
        Mij = []
        Mji = []
        for i in range(N-1):
            tmp_val = tsij[i] * tsji[i] - rsij[i] * rsji[i]
            Mji.append(np.array([[1, -rsij[i]],
                                 [rsji[i], tmp_val]], dtype=complex))
            Mij.append(np.array([[tmp_val, rsij[i]],
                                 [-rsji[i], 1]], dtype=complex))
        
        MUp = np.eye(2, dtype=complex)
        FactorUp = 1
        for i in range(source_layer_index - 1):
            MUp = MUp @ Mji[i] @ PjInv[i]
            FactorUp *= tsji[i]
        
        MDown = Mij[-1]
        FactorDown = tsij[-1]
        for i in range(N-2 - 1, source_layer_index - 1 - 1, -1):
            MDown = MDown @ Pj[i] @ Mij[i]
            FactorDown *= tsij[i]
        
        tmp_val = MDown[1, 1] * MUp[0, 0] - MDown[0, 1] * MUp[1, 0]
        
        if output_layer_index == 1:
            ts_out = np.array([FactorUp * MDown[1, 1] / tmp_val,
                               FactorUp * MDown[0, 1] / tmp_val], dtype=complex)
        elif output_layer_index == N:
            ts_out = np.array([FactorDown * MUp[1, 0] / tmp_val,
                               FactorDown * MUp[0, 0] / tmp_val], dtype=complex)
        elif output_layer_index == source_layer_index:
            ts_out = np.array([[MUp[1, 0] * MDown[0, 1] / tmp_val, MUp[0, 0] * MDown[0, 1] / tmp_val],
                               [MUp[1, 0] * MDown[1, 1] / tmp_val, MUp[1, 0] * MDown[0, 1] / tmp_val]], dtype=complex)
        else:
            if output_layer_index < source_layer_index:
                L = np.eye(2, dtype=complex)
                for i in range(output_layer_index - 1):
                    L = L @ Mji[i] @ PjInv[i]
                Factor = 1
                for i in range(output_layer_index - 1, source_layer_index - 1):
                    Factor *= tsji[i]
                ts_out = np.array([[Factor * L[0, 0] * MDown[1, 1] / tmp_val, Factor * L[0, 0] * MDown[0, 1] / tmp_val],
                                   [Factor * L[1, 0] * MDown[1, 1] / tmp_val, Factor * L[1, 0] * MDown[0, 1] / tmp_val]], dtype=complex)
            elif output_layer_index > source_layer_index:
                L = Mij[-1]
                for i in range(N-2 - 1, output_layer_index - 1 - 1, -1):
                    L = L @ Pj[i] @ Mij[i]
                Factor = 1
                for i in reversed(range(source_layer_index - 1, output_layer_index - 1)):
                    Factor *= tsij[i]
                ts_out = np.array([[Factor * L[0, 1] * MUp[1, 0] / tmp_val, Factor * L[0, 1] * MUp[0, 0] / tmp_val],
                                   [Factor * L[1, 1] * MUp[1, 0] / tmp_val, Factor * L[1, 1] * MUp[0, 0] / tmp_val]], dtype=complex)
            else:
                ts_out = None
        return ts_out

    return htp, hts

def sph_green_function(Kx, Ky, DFNiFe, lambda_, tp, ts):
    """
    Compute the spherical Green's functions for p- and s-polarized fields.

    Parameters
    ----------
    Kx, Ky : np.ndarray
        Lateral wavevector components.
    DFNiFe : float
        Dielectric function (permitivity) of the NiFe (or similar) layer.
    lambda_ : float
        Wavelength of the light.
    tp, ts : list or array_like
        Fresnel coefficients for p- and s-polarization, respectively.
        Each is expected to have two elements (e.g. tp[0] and tp[1]).

    Returns
    -------
    pGF : list of lists
        A 3×2 list containing the p-polarized Green's function components.
    sGF : list of lists
        A 3×2 list containing the s-polarized Green's function components.
    """
    # Constants
    c = 3e9
    mu0 = 4 * np.pi * 1e-7

    k0 = 2 * np.pi / lambda_
    w = c * k0
    ks = k0 * np.sqrt(DFNiFe)

    # Initialize Green's function containers as 3x2 lists.
    pGF = [[None, None] for _ in range(3)]
    sGF = [[None, None] for _ in range(3)]

    # Calculate the radial wavevector and angles.
    Kr = np.sqrt(Kx**2 + Ky**2)
    # Avoid division by zero: define cosPhi=1 and sinPhi=0 where Kr==0.
    cosPhi = np.where(Kr == 0, 1, Kx / Kr)
    sinPhi = np.where(Kr == 0, 0, Ky / Kr)

    # Calculate the z-component in the substrate.
    Kzs = np.sqrt(DFNiFe * k0**2 - Kr**2) + np.finfo(float).eps

    # --- s-polarized Green's functions ---
    sGF[0][0] = -1j * w**2 * mu0 / 2 * sinPhi * ts[0] / Kzs
    sGF[1][0] =  1j * w**2 * mu0 / 2 * cosPhi * ts[0] / Kzs
    sGF[2][0] = np.zeros_like(Kr)

    sGF[0][1] = -1j * w**2 * mu0 / 2 * sinPhi * ts[1] / Kzs
    sGF[1][1] =  1j * w**2 * mu0 / 2 * cosPhi * ts[1] / Kzs
    sGF[2][1] = np.zeros_like(Kr)

    # --- p-polarized Green's functions ---
    pGF[0][0] =  1j * w**2 * mu0 / 2 * cosPhi * tp[0] / ks
    pGF[1][0] =  1j * w**2 * mu0 / 2 * sinPhi * tp[0] / ks
    pGF[2][0] = -1j * w**2 * mu0 / 2 * tp[0] * Kr / (Kzs * ks)

    pGF[0][1] = -1j * w**2 * mu0 / 2 * cosPhi * tp[1] / ks
    pGF[1][1] = -1j * w**2 * mu0 / 2 * sinPhi * tp[1] / ks
    pGF[2][1] = -1j * w**2 * mu0 / 2 * tp[1] * Kr / (Kzs * ks)

    return pGF, sGF
