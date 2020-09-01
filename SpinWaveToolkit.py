# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 \n
This module provides analytical tools in Spin wave physics

Available classes are:
    SpinWaveCharacteristic -- Compute spin wave characteristic in dependance to k-vector \n
    Material -- Class for magnetic materials used in spin wave resreach
    
Available constants: \n
    mu0 -- Magnetic permeability
    
Available functions: \n
    wavenumberToWavelength -- Convert wavelength to wavenumber \n
    wavelengthTowavenumber -- Convert wavenumber to wavelength \n
    
Example code for obtaining propagation lenght and dispersion charactetristic: \n

import SpinWaveToolkit as SWT
import numpy as np

\~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n
#Here is an example of code    \n

import SpinWaveToolkit as SWT \n
import numpy as np \n

kxi = np.linspace(1e-12, 10e6, 150) \n

NiFeChar = SWT.SpinWaveCharacteristic(kxi = kxi, theta = np.pi/2, phi = np.pi/2,n =  0, d = 10e-9, weff = 2e-6, nT = 1, boundaryCond = 2, Bext = 20e-3, material = SWT.NiFe) \n
DispPy = NiFeChar.GetDispersion()*1e-9/(2*np.pi) #GHz \n
vgPy = NiFeChar.GetGroupVelocity()*1e-3 # km/s \n
lifetimePy = NiFeChar.GetLifetime()*1e9 #ns \n
propLen = NiFeChar.GetPropLen()*1e6 #um \n
\~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n
@author: Ondrej Wojewoda
"""
import numpy as np

mu0 = 4*np.pi*1e-7; #Magnetic permeability

class SpinWaveCharacteristic:
    """Compute spin wave characteristic in dependance to k-vector (wavenumber) such as frequency, group velocity, lifetime and propagation length
    #The model uses famous Slavin-Kalinikos equation from https://doi.org/10.1088%2F0022-3719%2F19%2F35%2F014
    Keyword arguments: \n
    kxi -- k-vector (wavenumber) in rad/m, usually vector (default linspace(1e-12, 25e6, 200)) \n
    theta -- out of plane angle in rad, pi/2 is totally inplane magnetization (default pi/2) \n
    phi -- in-plane angle in rad, pi/2 is DE geometry (default pi/2) \n
    n -- the quantization number in z (out-of-plane) direction (default 0) \n
    d -- thickness of layer in m (in z direction) \n
    weff -- effective width of the waveguide in um (optional, default 3e-6 um) \n
    n -- the transversal quantization number in y  direction (default 0) \n
    boundaryCond -- 1 is is totally unpinned and 2 is totally pinned boundary condition \n
    
    w0 -- parameter in Slavin-Kalinikos equation in rad*Hz/T w0 = mu0*gamma*Hext \n
    wM -- parameter in Slavin-Kalinikos equation in rad*Hz/T w0 = mu0*gamma*Ms \n
    A -- parameter in Slavin-Kalinikos equation A = Aex*2/(Ms**2*mu0) \n
    
    Availible methods: \n
    GetDisperison \n
    GetLifetime \n
    GetGroupVelocity \n
    GetPropLen \n
    Code example: \n
    #Here is an example of code
    kxi = np.linspace(1e-12, 150e6, 150) \n
     \n
    NiFeChar = SpinWaveCharacteristic(kxi = kxi, theta = np.pi/2, phi = np.pi/2,n =  0, d = 30e-9, weff = 2e-6, nT = 0, boundaryCond = 2, Bext = 20e-3, material = NiFe) \n
    DispPy = NiFeChar.GetDispersion()*1e-9/(2*np.pi) #GHz \n
    vgPy = NiFeChar.GetGroupVelocity()*1e-3 # km/s \n
    lifetimePy = NiFeChar.GetLifetime()*1e9 #ns \n
    propLen = NiFeChar.GetPropLen()*1e6 #um \n
    """
    def __init__(self, Bext, material, d, kxi = np.linspace(1e-12, 25e6, 200), theta = np.pi/2, phi = np.pi/2, n = 0, weff = 3e-6, nT = 0, boundaryCond = 2):
        self.kxi = kxi
        self.theta = theta
        self.phi= phi
        self.n = n
        self.d = d
        self.weff = weff
        self.nT = nT
        self.boundaryCond = boundaryCond
        self.alpha = material.alpha
        #Compute Slavin-Kalinikos parameters wM, w0 A
        self.wM = material.Ms*material.gamma*mu0
        self.w0 = material.gamma*Bext
        self.A = material.Aex*2/(material.Ms**2*mu0)
    def GetDispersion(self):
        """ Gives frequencies for defined k (Dispersion relation) \n
        The returned value is in the same units as w0, wM"""
        k = np.sqrt(np.power(self.kxi,2) + np.power(self.n*np.pi/self.d,2) + np.power(self.nT*np.pi/self.weff,2))
        # Totally unpinned boundary condition
        if self.boundaryCond == 1:
            if self.n == 0:
                Pnn = (self.kxi**2)/(k**2) - (self.kxi**4)/(k**4)*(1/2)*(2/(self.kxi*self.d)*(1-np.exp(-self.kxi*self.d)))
            else:
                Pnn = (self.kxi**2)/(k**2) - (self.kxi**4)/(k**4)*(2/(self.kxi*self.d)*(1-np.exp(-self.kxi*self.d)))
        # Totally pinned boundary condition
        else:
            Pnn = (self.kxi**2)/(k**2) - (self.kxi**2)/(k**2)*(1 - np.exp(-k*self.d))/(k*self.d);
        # Deprecated
            # Pnn = np.power(kxi,2)/np.power(k,2) - np.power(kxi,2)/np.power(k,2)*(1 - np.exp(-k*d))/(k*d)
        Pnn[0] = 0
        Fnn = Pnn + np.power(np.sin(self.theta),2)*(1-Pnn*(1+np.power(np.cos(self.phi),2)) + self.wM*(Pnn*(1 - Pnn)*np.power(np.sin(self.phi),2))/(self.w0 + self.A*self.wM*np.power(k,2)))
        f = np.sqrt((self.w0 + self.A*self.wM*np.power(k,2))*(self.w0 + self.A*self.wM*np.power(k,2) + self.wM*Fnn))
        return f
    def GetGroupVelocity(self):
        """ Gives group velocities for defined k \n
        The group velocity is computed as vg = dw/dk
        """
        f = self.GetDispersion()
        vg = np.diff(f)/(self.kxi[2]-self.kxi[1])
        return(vg)
    def GetLifetime(self):
        """ Gives lifetimes for defined k \n
        lifetime is computed as tau = (alpha*w*dw/dw0)^-1"""
        w0Ori = self.w0
        self.w0 = w0Ori*0.9999999
        dw0p999 = self.GetDispersion()
        self.w0 = w0Ori*1.0000001
        dw0p001 = self.GetDispersion()
        self.w0 = w0Ori
        lifetime = (self.alpha*self.GetDispersion()*(dw0p001 - dw0p999)/(w0Ori*1.0000001 - w0Ori*0.9999999))**-1
        return lifetime
    def GetPropLen(self):
        """ Give propagation lengths for defined k \n
        propagation length is computed as lambda = vg*tau"""
        propLen = self.GetLifetime()[0:-1]*self.GetGroupVelocity()
        return propLen
    
def wavenumberToWavelength(wavenumber):
    """ Convert wavelength to wavenumber
    lambda = 2*pi/k     
    Arguments: \n
    wavenumber -- wavenumber of the wave (rad/m)
    Return: \n
    wavelength (m)"""
    return 2*np.pi/wavenumber
def wavelengthTowavenumber(wavelength):
    """ Convert wavenumber to wavelength
    k = 2*pi/lambda 
    Arguments: \n
    wavelength -- wavelength of the wave (m)
    Return: \n
    wavenumber (rad/m)"""
    return 2*np.pi/wavelength
      
class Material:
    """Class for magnetic materials used in spin wave resreach \n
    To define custom material please type MyNewMaterial = Material(Ms = MyMS, Aex = MyAex, alpha = MyAlpha, gamma = MyGamma) \n
    Keyword arguments: \n
    Ms -- Saturation magnetization (A/m) \n
    Aex -- Exchange constant (J/m) \n
    alpha -- Gilbert damping ()\n
    gamma -- Gyromagnetic ratio (rad*GHz/T) (default 28.1*pi) \n \n
    Predefined materials are: \n
    NiFe (Permalloy)\n
    CoFeB\n
    FeNi (Metastable iron)"""
    def __init__(self, Ms, Aex, alpha, gamma = 28.1*np.pi*1e9):
        self.Ms = Ms
        self.Aex = Aex
        self.alpha = alpha
        self.gamma = gamma
#Predefined materials
NiFe = Material(Ms = 800e3, Aex = 16e-12, alpha = 70e-4)
CoFeB = Material(Ms = 1250e3, Aex = 15e-12, alpha = 40e-4, gamma=30*np.pi*1e9)
FeNi = Material(Ms = 1410e3, Aex = 11e-12, alpha = 80e-4)