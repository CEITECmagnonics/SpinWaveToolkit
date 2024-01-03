# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 \n
This module provides analytical tools in Spin wave physics
Available classes are:
    DispersionCharacteristic -- Compute spin wave characteristic in dependance to k-vector \n
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
NiFeChar = SWT.DispersionCharacteristic(kxi = kxi, theta = np.pi/2, phi = np.pi/2,n =  0, d = 10e-9, weff = 2e-6, nT = 1, boundaryCond = 2, Bext = 20e-3, material = SWT.NiFe) \n
DispPy = NiFeChar.GetDispersion()*1e-9/(2*np.pi) #GHz \n
vgPy = NiFeChar.GetGroupVelocity()*1e-3 # km/s \n
lifetimePy = NiFeChar.GetLifetime()*1e9 #ns \n
propLen = NiFeChar.GetPropLen()*1e6 #um \n
\~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n
@author: Ondrej Wojewoda, ondrej.wojewoda@ceitec.vutbr.cz
"""
import numpy as np
from scipy.optimize import fsolve, minimize
from numpy import linalg as LA
# from scipy.integrate import trapz

mu0 = 4*np.pi*1e-7; #Magnetic permeability

class DispersionCharacteristic:
    """Compute spin wave characteristic in dependance to k-vector (wavenumber) such as frequency, group velocity, lifetime and propagation length
    #The model uses famous Slavin-Kalinikos equation from https://doi.org/10.1088%2F0022-3719%2F19%2F35%2F014
    Keyword arguments: \n
    kxi -- k-vector (wavenumber) in rad/m, usually vector (default linspace(1e-12, 25e6, 200)) \n
    theta -- out of plane angle in rad, pi/2 is totally inplane magnetization (default pi/2) \n
    phi -- in-plane angle in rad, pi/2 is DE geometry (default pi/2) \n
    n -- the quantization number in z (out-of-plane) direction (default 0) \n
    d -- thickness of layer in m (in z direction) \n
    weff -- effective width of the waveguide in um (optional, default 3e-6 um) \n
    boundaryCond -- 1 is is totally unpinned and 2 is totally pinned boundary condition, 3 is a long wave limit, 4 is partially pinned \n
    dp -- for 4 BC, pinning parameter ranges from 0 to inf. 0 means totally unpinned \n
    
    w0 -- parameter in Slavin-Kalinikos equation in rad*Hz/T w0 = mu0*gamma*Hext \n
    wM -- parameter in Slavin-Kalinikos equation in rad*Hz/T w0 = mu0*gamma*Ms \n
    A -- parameter in Slavin-Kalinikos equation A = Aex*2/(Ms**2*mu0) \n
    
    Availible methods: \n
    GetDisperison \n
    GetLifetime \n
    GetGroupVelocity \n
    GetPropLen \n
    GetPropagationVector \n
    GetPropagationQVector \n
    GetSecondPerturbation \n
    GetDensityOfStates \n
    Code example: \n
    #Here is an example of code
    kxi = np.linspace(1e-12, 150e6, 150) \n
     \n
    NiFeChar = DispersionCharacteristic(kxi = kxi, theta = np.pi/2, phi = np.pi/2,n =  0, d = 30e-9, weff = 2e-6, nT = 0, boundaryCond = 2, Bext = 20e-3, material = NiFe) \n
    DispPy = NiFeChar.GetDispersion()*1e-9/(2*np.pi) #GHz \n
    vgPy = NiFeChar.GetGroupVelocity()*1e-3 # km/s \n
    lifetimePy = NiFeChar.GetLifetime()*1e9 #ns \n
    propLen = NiFeChar.GetPropLen()*1e6 #um \n
    """
    def __init__(self, Bext, material, d, kxi = np.linspace(1e-12, 25e6, 200), theta = np.pi/2, phi = np.pi/2, weff = 3e-6, boundaryCond = 1, dp=0, Ku = 0, Ku2 = 0,  Jbl=0, Jbq=0, s=0, d2=0, material2=0, JblDyn=1, JbqDyn=1, phiAnis1=np.pi/2, phiAnis2=np.pi/2):
        self.kxi = np.array(kxi)
        self.theta = theta
        self.phi= phi
        self.d = d
        self.d1 = d
        self.weff = weff
        self.boundaryCond = boundaryCond
        self.alpha = material.alpha
        #Compute Slavin-Kalinikos parameters wM, w0 A
        self.wM = material.Ms*material.gamma*mu0
        self.w0 = material.gamma*Bext
        self.wU = material.gamma*2*Ku/material.Ms
        self.A = material.Aex*2/(material.Ms**2*mu0)
        self.dp = dp
        self.gamma = material.gamma
        self.mu0dH0 = material.mu0dH0
        
        self.Ms = material.Ms
        self.Hani = 2*Ku/material.Ms/mu0
        self.phiAnis1 = phiAnis1
        self.phiAnis2 = phiAnis2
        if d2==0:
            self.d2=d
        else:
            self.d2=d2
        if material2==0:
            self.Ms2 = material.Ms
            self.Hani2 = 2*Ku/material.Ms/mu0
            self.A2 = material.Aex*2/(material.Ms**2*mu0)
        else:
            self.Ms2 = material2.Ms
            self.Hani2 = 2*Ku2/material2.Ms/mu0
            self.A2 = material2.Aex*2/(material2.Ms**2*mu0)
        self.s = s
        self.Jbl = Jbl
        self.Jbq = Jbq
        self.Bext = Bext
        self.Ku = Ku
        self.Ku2 = Ku2
        if JblDyn==1:
            JblDyn=Jbl
        if JbqDyn==1:
            JbqDyn=Jbq
        self.JblDyn = JblDyn
        self.JbqDyn = JbqDyn
        
    def GetPropagationVector(self, n = 0, nc = -1, nT = 0):
        """ Gives dimensionless propagation vector \n
        The boundary condition is chosen based on the object property \n
        Arguments: \n
        n -- Quantization number \n
        nc(optional) -- Second quantization number. Used for hybridization \n
        nT(optional) -- Waveguide (transversal) quantization number
        """
        if nc == -1:
            nc = n
        kxi = np.sqrt(self.kxi**2 + (nT*np.pi/self.weff)**2)
        kappa = n*np.pi/self.d
        kappac = nc*np.pi/self.d
        k = np.sqrt(np.power(kxi,2) + kappa**2)
        kc = np.sqrt(np.power(kxi,2) + kappac**2 )
        # Totally unpinned boundary condition
        if self.boundaryCond == 1:
            Fn = 2/(kxi*self.d)*(1-(-1)**n*np.exp(-kxi*self.d))
            if n == 0 and nc == 0:         
                Pnn = (kxi**2)/(kc**2) - (kxi**4)/(k**2*kc**2)*1/2*((1+(-1)**(n+nc))/2)*Fn
            elif n == 0 and nc != 0 or nc == 0 and n != 0 :
                Pnn =  - (kxi**4)/(k**2*kc**2)*1/np.sqrt(2)*((1+(-1)**(n+nc))/2)*Fn
            elif n == nc:
                Pnn = (kxi**2)/(kc**2) - (kxi**4)/(k**2*kc**2)*((1+(-1)**(n+nc))/2)*Fn
            else:
                Pnn = - (kxi**4)/(k**2*kc**2)*((1+(-1)**(n+nc))/2)*Fn
        # Totally pinned boundary condition
        elif self.boundaryCond == 2:
            if n == nc:
                Pnn = (kxi**2)/(kc**2) + (kxi**2)/(k**2)*(kappa*kappac)/(kc**2)*(1+(-1)**(n+nc)/2)*2/(kxi*self.d)*(1-(-1)**n*np.exp(-kxi*self.d));
            else:
                Pnn = (kxi**2)/(k**2)*(kappa*kappac)/(kc**2)*(1+(-1)**(n+nc)/2)*2/(kxi*self.d)*(1-(-1)**n*np.exp(-kxi*self.d));
        # Totally unpinned condition - long wave limit         
        elif self.boundaryCond == 3:
                if n == 0:
                    Pnn = kxi*self.d/2
                else:
                    Pnn = (kxi*self.d)**2/(n**2*np.pi**2)
        # Partially pinned boundary condition
        elif self.boundaryCond == 4:
            dp = self.dp
            kappa = self.GetPartiallyPinnedKappa(n) #We have to get correct kappa from transversal eq.
            kappac = self.GetPartiallyPinnedKappa(nc)
            if kappa == 0:
                kappa = 1e1
            if kappac == 0:
                kappac = 1e1
            k = np.sqrt(np.power(kxi,2) + kappa**2)
            kc = np.sqrt(np.power(kxi,2) + kappac**2 )
            An = np.sqrt(2*((kappa**2 + dp**2)/kappa**2 + np.sin(kappa*self.d)/(kappa*self.d) * ((kappa**2 - dp**2)/kappa**2*np.cos(kappa*self.d) + 2*dp/kappa*np.sin(kappa*self.d)))**-1)
            Anc = np.sqrt(2*((kappac**2 + dp**2)/kappac**2 + np.sin(kappac*self.d)/(kappac*self.d) * ((kappac**2 - dp**2)/kappac**2*np.cos(kappac*self.d) + 2*dp/kappac*np.sin(kappac*self.d)))**-1)
            Pnnc = kxi*An*Anc/(2*self.d*k**2*kc**2)*((kxi**2 - dp**2)*np.exp(-kxi*self.d)*(np.cos(kappa*self.d) + 
                                                          np.cos(kappac*self.d)) + (kxi - dp)*np.exp(-kxi*self.d)*((dp*kxi - kappa**2)*np.sin(kappa*self.d)/kappa + 
                                                          (dp*kxi - kappac**2)*np.sin(kappac*self.d)/kappac) - (kxi**2 - dp**2)*(1 + np.cos(kappa*self.d)*np.cos(kappac*self.d)) + 
                                                          (kappa**2*kappac**2 - dp**2*kxi**2)*np.sin(kappa*self.d)/kappa*np.sin(kappac*self.d)/kappac - 
                                                          dp*(k**2*np.cos(kappac*self.d)*np.sin(kappa*self.d)/kappa + kc**2*np.cos(kappa*self.d)*np.sin(kappac*self.d)/kappac))
            if n == nc:
                Pnn = kxi**2/kc**2 + Pnnc
            else:
                Pnn = Pnnc
        else:
             raise Exception("Sorry, there is no boundary condition with this number.") 
            
        return(Pnn)
    def GetPropagationQVector(self, n = 0, nc = -1, nT = 0):
        """ Gives dimensionless propagation vector Q \n
        This vector accounts for interaction between odd and even spin wave mode \n
        The boundary condition is chosen based on the object property \n
        Arguments: \n
        n -- Quantization number \n
        nc(optional) -- Second quantization number. Used for hybridization \n
        nT(optional) -- Waveguide (transversal) quantization number
        """
        if nc == -1:
            nc = n
        kxi = np.sqrt(self.kxi**2 + (nT*np.pi/self.weff)**2)
        kappa = n*np.pi/self.d
        kappac = nc*np.pi/self.d
        if kappa == 0:
            kappa = 1
        if kappac == 0:
            kappac = 1
        k = np.sqrt(np.power(kxi,2) + kappa**2)
        kc = np.sqrt(np.power(kxi,2) + kappac**2 )
        # Totally unpinned boundary condition
        if self.boundaryCond == 1:
            Fn = 2/(kxi*self.d)*(1-(-1)**n*np.exp(-kxi*self.d))
            Qnn = kxi**2/kc**2*(kappac**2/(kappac**2-kappa**2)*2/(kxi*self.d) - kxi**2/(2*k**2)*Fn)*((1-(-1)**(n + nc))/2)
        elif self.boundaryCond == 4:
            dp = self.dp
            kappa = self.GetPartiallyPinnedKappa(n)
            kappac = self.GetPartiallyPinnedKappa(nc)
            if kappa == 0:
                kappa = 1
            if kappac == 0:
                kappac = 1
            An = np.sqrt(2*((kappa**2 + dp**2)/kappa**2 + np.sin(kappa*self.d)/(kappa*self.d) * ((kappa**2 - dp**2)/kappa**2*np.cos(kappa*self.d) + 2*dp/kappa*np.sin(kappa*self.d)))**-1)
            Anc = np.sqrt(2*((kappac**2 + dp**2)/kappac**2 + np.sin(kappac*self.d)/(kappac*self.d) * ((kappac**2 - dp**2)/kappac**2*np.cos(kappac*self.d) + 2*dp/kappac*np.sin(kappac*self.d)))**-1)
            Qnn = kxi*An*Anc/(2*self.d*k**2*kc**2)*((kxi**2-dp**2)*np.exp(-kxi*self.d)*(np.cos(kappa*self.d)-np.cos(kappac*self.d)) + (kxi - dp)*np.exp(-kxi*self.d)*((dp*kxi - 
                            kappa**2)*np.sin(kappa*self.d)/kappa - (dp*kxi - kappac**2)*np.sin(kappac*self.d)/kappac) + (kxi - dp)*((dp*kxi - 
                               kappac**2)*np.cos(kappa*self.d)*np.sin(kappac*self.d)/kappac - (dp*kxi - 
                                 kappa**2)*np.cos(kappac*self.d)*np.sin(kappa*self.d)/kappa) + (1 - 
                                    np.cos(kappac*self.d)*np.cos(kappa*self.d)*2*(kxi**2*dp**2 + 
                                          kappa**2*kappac**2 + (kappac**2 + kappa**2)*(kxi**2 + dp**2))/(kappac**2-kappa**2)
                                            - np.sin(kappa*self.d)*np.sin(kappac**2*self.d)/(kappa*kappac*(kappac**2-kappa**2))*(dp*kxi*(kappa**4+kappac**4) +
                                                 (dp**2*kxi**2 - kappa**2*kappac**2)*(kappa**2 + kappac**2)-2*kappa**2*kappac**2*(dp**2 + kxi**2 - dp*kxi))))
        else:
             raise Exception("Sorry, there is no boundary condition with this number.") 
        return Qnn
#    def GetTVector(self, n, nc, kappan, kappanc):
#        zeta = np.linspace(-self.d/2, self.d/2, 500)
#        An = 1
#        Phin = An*(np.cos(kappan)*(zeta + self.d/2) + self.dp/kappan*np.sin(kappan)*(zeta + self.d/2))
#        Phinc = An*(np.cos(kappanc)*(zeta + self.d/2) + self.dp/kappanc*np.sin(kappanc)*(zeta + self.d/2))
#        Tnn = 1/self.d*trapz(y = Phin*Phinc, x = zeta)
#        return Tnn
    def GetPartiallyPinnedKappa(self, n):
        """ Gives kappa from the transverse equation \n
        Arguments: \n
        n -- Quantization number \n
        """
        def transEq(kappa, d, dp):
            e = (kappa**2 - dp**2)*np.tan(kappa*d) - kappa*dp*2 
            return e
        #The classical thickness mode is given as starting point
        kappa = fsolve(transEq, x0=(n*np.pi/self.d), args = (self.d, self.dp), maxfev=10000, epsfcn=1e-10, factor=0.1)
        return(kappa)
    def GetDispersion(self, n=0, nc=-1, nT=0):
        """ Gives frequencies for defined k (Dispersion relation) \n
        The returned value is in the rad Hz \n
        Arguments: \n
        n -- Quantization number \n
        nc(optional) -- Second quantization number. Used for hybridization \n
        nT(optional) -- Waveguide (transversal) quantization number """
        if nc == -1:
            nc = n
        if self.boundaryCond == 4:
            kappa = self.GetPartiallyPinnedKappa(n)
        else:
            kappa = n*np.pi/self.d
        kxi = np.sqrt(self.kxi**2 + (nT*np.pi/self.weff)**2)
        k = np.sqrt(np.power(kxi,2) + kappa**2)
        phi = np.arctan((nT*np.pi/self.weff)/self.kxi) - self.phi
        Pnn = self.GetPropagationVector(n = n, nc = nc, nT = nT)
        Fnn = Pnn + np.power(np.sin(self.theta),2)*(1-Pnn*(1+np.power(np.cos(phi),2)) + self.wM*(Pnn*(1 - Pnn)*np.power(np.sin(phi),2))/(self.w0 + self.A*self.wM*np.power(k,2)))
        f = np.sqrt((self.w0 + self.A*self.wM*np.power(k,2))*(self.w0 + self.A*self.wM*np.power(k,2) + self.wM*Fnn))
        return f
    def GetDispersionTacchi(self):
        """ Gives frequencies for defined k (Dispersion relation) \n
        The returned value is in the rad Hz \n
        Arguments: \n
        n -- Quantization number \n
        nc(optional) -- Second quantization number. Used for hybridization \n
        nT(optional) -- Waveguide (transversal) quantization number """
        ks = np.sqrt(np.power(self.kxi,2))
        phi = self.phi
        wV = np.zeros((6, np.size(ks,0)))
        for idx, k in enumerate(ks):
            Ck = np.array([[-(self.ankTacchi(0, k) + self.CnncTacchi(0, 0, k, phi)), -(self.bTacchi() + self.pnncTachci(0, 0, k, phi)), 0, -self.qnncTacchi(1, 0, k, phi), -self.CnncTacchi(2, 0, k, phi), -self.pnncTachci(2, 0, k, phi)],
                           [(self.bTacchi() + self.pnncTachci(0, 0, k, phi)), (self.ankTacchi(0, k) + self.CnncTacchi(0, 0, k, phi)), -self.qnncTacchi(1, 0, k, phi), 0, self.pnncTachci(2, 0, k, phi), self.CnncTacchi(2, 0, k, phi)],
                           [0, -self.qnncTacchi(0, 1, k, phi), -(self.ankTacchi(1, k) + self.CnncTacchi(1, 1, k, phi)), -(self.bTacchi() + self.pnncTachci(1, 1, k, phi)), 0, -self.qnncTacchi(2, 1, k, phi)],
                           [-self.qnncTacchi(0, 1, k, phi), 0, (self.bTacchi() + self.pnncTachci(1, 1, k, phi)), (self.ankTacchi(1, k) + self.CnncTacchi(1, 1, k, phi)), -self.qnncTacchi(2, 1, k, phi), 0],
                           [-self.CnncTacchi(0, 2, k, phi), -self.pnncTachci(0, 2, k, phi), 0, -self.qnncTacchi(1, 2, k, phi), -(self.ankTacchi(2, k) + self.CnncTacchi(2, 2, k, phi)), -(self.bTacchi() + self.pnncTachci(2, 2, k, phi))],
                           [self.pnncTachci(0, 2, k, phi), self.CnncTacchi(0, 2, k, phi), -self.qnncTacchi(1, 2, k, phi), 0, (self.bTacchi() + self.pnncTachci(2, 2, k, phi)), (self.ankTacchi(2, k) + self.CnncTacchi(2, 2, k, phi))]],
                          dtype=float)
            w, v = LA.eig(Ck)
            wV[:, idx] = np.sort(w)
        return wV
    def CnncTacchi(self, n, nc, k, phi):
        Cnnc = -self.wM/2*(1 - np.sin(phi)**2)*self.PnncTacchi(n,nc,k)
        return Cnnc
    def pnncTachci(self, n, nc, k, phi):
        pnnc = -self.wM/2*(1 + np.sin(phi)**2)*self.PnncTacchi(n,nc,k)
        return pnnc
    def qnncTacchi(self, n, nc, k, phi):
        qnnc = -self.wM/2*np.sin(phi)*self.QnncTacchi(n,nc,k)
        return qnnc
    def OmegankTacchi(self, n, k):
        Omegank = self.w0 + self.wM*self.A*(k**2 + (n*np.pi/self.d)**2)
        return Omegank
    def ankTacchi(self, n, k):
        ank = self.OmegankTacchi(n, k) + self.wM/2 - self.wU/2
        return ank
    def bTacchi(self):
        b = self.wM/2 - self.wU/2
        return b
    def PnncTacchi(self, n, nc,kxi):
        """ Gives dimensionless propagation vector \n
        The boundary condition is chosen based on the object property \n
        Arguments: \n
        n -- Quantization number \n
        nc(optional) -- Second quantization number. Used for hybridization \n
        nT(optional) -- Waveguide (transversal) quantization number
        """
        kappa = n*np.pi/self.d
        kappac = nc*np.pi/self.d
        k = np.sqrt(np.power(kxi,2) + kappa**2)
        kc = np.sqrt(np.power(kxi,2) + kappac**2 )
        # Totally unpinned boundary condition
        if self.boundaryCond == 1:
            Fn = 2/(kxi*self.d)*(1-(-1)**n*np.exp(-kxi*self.d))
            if n == 0 and nc == 0:         
                Pnn = (kxi**2)/(kc**2) - (kxi**4)/(k**2*kc**2)*1/2*((1+(-1)**(n+nc))/2)*Fn
            elif n == 0 and nc != 0 or nc == 0 and n != 0 :
                Pnn =  - (kxi**4)/(k**2*kc**2)*1/np.sqrt(2)*((1+(-1)**(n+nc))/2)*Fn
            elif n == nc:
                Pnn = (kxi**2)/(kc**2) - (kxi**4)/(k**2*kc**2)*((1+(-1)**(n+nc))/2)*Fn
            else:
                Pnn = - (kxi**4)/(k**2*kc**2)*((1+(-1)**(n+nc))/2)*Fn
        # Totally pinned boundary condition
        elif self.boundaryCond == 2:
            if n == nc:
                Pnn = (kxi**2)/(kc**2) + (kxi**2)/(k**2)*(kappa*kappac)/(kc**2)*(1+(-1)**(n+nc)/2)*2/(kxi*self.d)*(1-(-1)**n*np.exp(-kxi*self.d));
            else:
                Pnn = (kxi**2)/(k**2)*(kappa*kappac)/(kc**2)*(1+(-1)**(n+nc)/2)*2/(kxi*self.d)*(1-(-1)**n*np.exp(-kxi*self.d));
        else:
             raise Exception("Sorry, there is no boundary condition with this number for Tacchi numeric solution.") 
            
        return(Pnn)
    def QnncTacchi(self, n, nc, kxi):
        """ Gives dimensionless propagation vector Q \n
        This vector accounts for interaction between odd and even spin wave mode \n
        The boundary condition is chosen based on the object property \n
        Arguments: \n
        n -- Quantization number \n
        nc(optional) -- Second quantization number. Used for hybridization \n
        nT(optional) -- Waveguide (transversal) quantization number
        """
        # The totally pinned BC should be added
        kappa = n*np.pi/self.d
        kappac = nc*np.pi/self.d
        if kappa == 0:
            kappa = 1
        if kappac == 0:
            kappac = 1
        k = np.sqrt(np.power(kxi,2) + kappa**2)
        kc = np.sqrt(np.power(kxi,2) + kappac**2 )
        # Totally unpinned boundary condition
        if self.boundaryCond == 1:
            Fn = 2/(kxi*self.d)*(1-(-1)**n*np.exp(-kxi*self.d))
            Qnn = kxi**2/kc**2*(kappac**2/(kappac**2-kappa**2)*2/(kxi*self.d) - kxi**2/(2*k**2)*Fn)*((1-(-1)**(n + nc))/2)
        elif self.boundaryCond == 4:
            dp = self.dp
            kappa = self.GetPartiallyPinnedKappa(n)
            kappac = self.GetPartiallyPinnedKappa(nc)
            if kappa == 0:
                kappa = 1
            if kappac == 0:
                kappac = 1
            An = np.sqrt(2*((kappa**2 + dp**2)/kappa**2 + np.sin(kappa*self.d)/(kappa*self.d) * ((kappa**2 - dp**2)/kappa**2*np.cos(kappa*self.d) + 2*dp/kappa*np.sin(kappa*self.d)))**-1)
            Anc = np.sqrt(2*((kappac**2 + dp**2)/kappac**2 + np.sin(kappac*self.d)/(kappac*self.d) * ((kappac**2 - dp**2)/kappac**2*np.cos(kappac*self.d) + 2*dp/kappac*np.sin(kappac*self.d)))**-1)
            Qnn = kxi*An*Anc/(2*self.d*k**2*kc**2)*((kxi**2-dp**2)*np.exp(-kxi*self.d)*(np.cos(kappa*self.d)-np.cos(kappac*self.d)) + (kxi - dp)*np.exp(-kxi*self.d)*((dp*kxi - 
                            kappa**2)*np.sin(kappa*self.d)/kappa - (dp*kxi - kappac**2)*np.sin(kappac*self.d)/kappac) + (kxi - dp)*((dp*kxi - 
                               kappac**2)*np.cos(kappa*self.d)*np.sin(kappac*self.d)/kappac - (dp*kxi - 
                                 kappa**2)*np.cos(kappac*self.d)*np.sin(kappa*self.d)/kappa) + (1 - 
                                    np.cos(kappac*self.d)*np.cos(kappa*self.d)*2*(kxi**2*dp**2 + 
                                          kappa**2*kappac**2 + (kappac**2 + kappa**2)*(kxi**2 + dp**2))/(kappac**2-kappa**2)
                                            - np.sin(kappa*self.d)*np.sin(kappac**2*self.d)/(kappa*kappac*(kappac**2-kappa**2))*(dp*kxi*(kappa**4+kappac**4) +
                                                 (dp**2*kxi**2 - kappa**2*kappac**2)*(kappa**2 + kappac**2)-2*kappa**2*kappac**2*(dp**2 + kxi**2 - dp*kxi))))
        else:
             raise Exception("Sorry, there is no boundary condition with this number for Tacchi numeric solution.") 
        return Qnn
    
    
    def GetDispersionSAFM(self, n=0):
        """ Gives frequencies for defined k (Dispersion relation) \n
        The returned value is in the rad Hz \n
        Seems that this model has huge aproximation and I recomend to not use it \n
        Arguments: \n
        n -- number of mode - 0 - acoustic """
        Zet = np.sinh(self.kxi*self.d/2)/(self.kxi*self.d/2)*np.exp(-abs(self.kxi)*self.d/2)
        g = mu0*self.Ms*Zet**2*np.exp(-abs(self.kxi)*self.s)*self.kxi*self.d/2
        p = mu0*self.Hani + mu0*self.Ms*self.kxi**2*self.A + mu0*self.Ms*(1 - Zet)
        q = mu0*self.Hani + mu0*self.Ms*self.kxi**2*self.A + mu0*self.Ms*Zet
        Cj = (self.Jbl - 2*self.Jbq)/(self.Ms*self.d)
        
        if n==0:       
            f = self.gamma*(g + np.sqrt((p - g)*(q - g - 2*Cj)))
        elif n==1:
            f = self.gamma*(-g + np.sqrt((q + g)*(p + g - 2*Cj)))
        return f
    def GetDispersionSAFMNumeric(self):
        """ Gives frequencies for defined k (Dispersion relation) \n
        The returned value is in the rad Hz \n
        Arguments: \n
        n -- number of mode - 0 - acoustic """
        
        Ms1 = self.Ms
        Ms2 = self.Ms2
        A1 = self.A
        A2 = self.A2
        Hu1 = self.Hani
        Hu2 = self.Hani2
        d1 = self.d1
        d2 = self.d2
        phiAnis1 = self.phiAnis1
        phiAnis2 = self.phiAnis2
        
        Hs1 = 0 #Surface anisotropy of the first layer
        Hs2 = 0 #Surface anisotropy of the second layer
        
        phi1, phi2 = wrapAngle(self.GetPhisSAFM())
        
        ks = self.kxi
        wV = np.zeros((4, np.size(ks,0)))
        for idx, k in enumerate(ks):
            Zet1 = np.sinh(k*self.d1/2)/(k*self.d1/2)*np.exp(-abs(k)*self.d1/2)
            Zet2 = np.sinh(k*self.d2/2)/(k*self.d2/2)*np.exp(-abs(k)*self.d2/2)
            
            Hz1e0 = self.Bext/mu0*np.cos(wrapAngle(self.phi - phi1)) + Hu1*np.cos(phi1-phiAnis1)**2 + (self.JblDyn*np.cos(wrapAngle(phi1-phi2)) + 2*self.JbqDyn*np.cos(wrapAngle(phi1-phi2))**2)/(d1*Ms1*mu0)
            Hz2e0 = self.Bext/mu0*np.cos(wrapAngle(self.phi - phi2)) + Hu2*np.cos(phi2-phiAnis2)**2 + (self.JblDyn*np.cos(wrapAngle(phi1-phi2)) + 2*self.JbqDyn*np.cos(wrapAngle(phi1-phi2))**2)/(d2*Ms2*mu0)
            
            AX1Y1 = -Ms1*Zet1-Ms1*A1*k**2 - Hz1e0 - Hs1
            AX1X2 = 1j*Ms1*np.sin(phi2)*k*d2/2*Zet1*Zet2*np.exp(-abs(k)*self.s)
            AX1Y2 = Ms1*abs(k)*d2/2*Zet1*Zet2*np.exp(-abs(k)*self.s)+(self.JblDyn+2*self.JbqDyn*np.cos(wrapAngle(phi1-phi2)))/(d1*Ms2*mu0)
            AY1X1 = Ms1*np.sin(phi1)**2*(1-Zet1)+Ms1*A1*k**2-Hu1*np.sin(phi1-phiAnis1)**2+Hz1e0-2*self.JbqDyn/(d1*Ms1*mu0)*np.sin(wrapAngle(phi1-phi2))**2
            AY1X2 = Ms1*np.sin(phi1)*np.sin(phi2)*abs(k)*d2/2*Zet1*Zet2*np.exp(-abs(k)*self.s) - (self.JblDyn*np.cos(wrapAngle(phi2-phi1))+2*self.JbqDyn*np.cos(wrapAngle(2*(phi1-phi2))))/(d1*Ms2*mu0) #mozna tady
            AY1Y2 = -1j*Ms1*np.sin(phi1)*k*d2/2*Zet1*Zet2*np.exp(-abs(k)*self.s)
            AX2X1 = -1j*Ms2*np.sin(phi1)*k*d1/2*Zet1*Zet2*np.exp(-abs(k)*self.s)
            AX2Y1 = Ms2*abs(k)*d1/2*Zet1*Zet2*np.exp(-abs(k)*self.s)+(self.JblDyn+2*self.JbqDyn*np.cos(wrapAngle(phi1-phi2)))/(d2*Ms1*mu0)
            AX2Y2 = -Ms2*Zet2-Ms2*A2*k**2-Hz2e0+Hs2
            AY2X1 = Ms2*np.sin(phi1)*np.sin(phi2)*abs(k)*d1/2*Zet1*Zet2*np.exp(-abs(k)*self.s)-(self.JblDyn*np.cos(wrapAngle(phi1-phi2))+2*self.JbqDyn*np.cos(wrapAngle(2*(phi1-phi2))))/(d2*Ms1*mu0) #a tady
            AY2Y1 = 1j*Ms2*np.sin(phi2)*k*d1/2*Zet1*Zet2*np.exp(-abs(k)*self.s)
            AY2X2 = Ms2*np.sin(phi2)**2*(1-Zet2)+Ms2*A2*k**2-Hu2*np.sin(phi2-phiAnis2)**2+Hz2e0-2*self.JbqDyn/(d2*Ms2*mu0)*np.sin(wrapAngle(phi1-phi2))**2   
            

            A = np.array([[0    , AX1Y1, AX1X2, AX1Y2],
                          [AY1X1, 0    , AY1X2, AY1Y2],
                          [AX2X1, AX2Y1, 0    , AX2Y2],
                          [AY2X1, AY2Y1, AY2X2, 0    ]], dtype=complex)
            w, v = LA.eig(A)
            wV[:, idx] = np.sort(np.imag(w)*self.gamma*mu0)
        return wV
    
    def GetDispersionSAFMNumericRezende(self):
        """ Gives frequencies for defined k (Dispersion relation) \n
        The returned value is in the rad Hz \n
        Arguments: \n
        n -- number of mode - 0 - acoustic """
        
        Ms1 = self.Ms
        Ms2 = self.Ms2
        A1 = self.A
        A2 = self.A2
        Hau1 = self.Hani
        Hau2 = self.Hani2
        d1 = self.d1
        d2 = self.d2
        H0 = self.Bext/mu0
        Hex1bl = self.JblDyn/(d1*Ms1*mu0)
        Hex2bl = self.JblDyn/(d2*Ms2*mu0)
        Hex1bq = self.JbqDyn/(d1*Ms1*mu0)
        Hex2bq = self.JbqDyn/(d2*Ms2*mu0)
        gamma1=self.gamma
        gamma2=self.gamma
        
        Has1 = 0 #Surface anisotropy of the first layer
        Has2 = 0 #Surface anisotropy of the second layer
        
        Hac1 = 0 #Cubi anisotropy of the first layer
        Hac2 = 0 #Cubic anisotropy of the second layer
        
        phi1, phi2 = wrapAngle(self.GetPhisSAFM())
        
        ks = self.kxi
        wV = np.zeros((4, np.size(ks,0)))
        for idx, k in enumerate(ks):
            H1=H0*np.cos(phi1-self.phi)+Hac1/4*(3+np.cos(4*phi1))+Hau1*np.cos(phi1-self.phiAnis1)**2-Has1+Ms1*(1-k*d1/2)+A1*k**2+Hex1bl*np.cos(phi1-phi2)-2*Hex1bq*np.cos(phi1-phi2)**2
            G1=H0*np.cos(phi2-self.phi)+Hac2/4*(3+np.cos(4*phi2))+Hau2*np.cos(phi2-self.phiAnis2)**2-Has2+Ms2*(1-k*d2/2)+A2*k**2+Hex2bl*np.cos(phi2-phi1)-2*Hex2bq*np.cos(phi2-phi1)**2
            H2=-Hex2bl-1/2*Ms1*k*d1*(1-k*d2/2)*np.exp(-k*self.s)+2*Hex2bq*np.cos(phi1-phi2)
            G2=-Hex1bl-1/2*Ms2*k*d2*(1-k*d1/2)*np.exp(-k*self.s)+2*Hex1bq*np.cos(phi2-phi1)
            H3=H0*np.cos(phi1-self.phi)+Hac1*np.cos(4*phi1)+Hau1*np.cos(2*(phi1-self.phiAnis1))+1/2*Ms1*k*d1*np.cos(phi1-self.phi)**2+A1*k**2+Hex1bl*np.cos(phi1-phi2)-2*Hex1bq*np.cos(2*(phi1-phi2))
            G3=H0*np.cos(phi2-self.phi)+Hac2*np.cos(4*phi2)+Hau2*np.cos(2*(phi2-self.phiAnis2))+1/2*Ms2*k*d2*np.cos(phi2-self.phi)**2+A2*k**2+Hex2bl*np.cos(phi2-phi1)-2*Hex2bq*np.cos(2*(phi2-phi1))
            H4=Hex2bl*np.cos(phi1-phi2)-1/2*Ms1*k*d1*(1-k*d2/2)*np.exp(-k*self.s)*np.cos(phi1-self.phi)*np.cos(phi2-self.phi)-2*Hex2bq*np.cos(2*(phi1-phi2))
            G4=Hex1bl*np.cos(phi2-phi1)-1/2*Ms2*k*d2*(1-k*d1/2)*np.exp(-k*self.s)*np.cos(phi2-self.phi)*np.cos(phi1-self.phi)-2*Hex1bq*np.cos(2*(phi2-phi1))
            H5=-1/2*Ms1*k*d1*(1-k*d2/2)*np.exp(-k*self.s)*np.cos(phi2-self.phi)
            G5=-1/2*Ms2*k*d2*(1-k*d1/2)*np.exp(-k*self.s)*np.cos(phi1-self.phi)
            H6=-1/2*Ms1*k*d1*(1-k*d2/2)*np.exp(-k*self.s)*np.cos(phi1-self.phi)
            G6=-1/2*Ms2*k*d2*(1-k*d1/2)*np.exp(-k*self.s)*np.cos(phi2-self.phi)
            
            A = np.array([[0     ,H1      ,1j*H5,H2   ],
                          [-H3   ,0       ,H4   ,1j*H6],
                          [-1j*G5,G2      ,0    ,G1   ],
                          [G4    ,-1j*G6  ,-G3  ,0    ]], dtype=complex)
            w, v = LA.eig(A)
            wV[:, idx] = np.sort(np.sqrt(w*np.conj(w))*self.gamma*mu0)
            
            # a=(G2*H4+G4*H2+G5*H5+G6*H6)/gamma1/gamma2-H1*H3/(gamma2**2)-G1*G3/(gamma1**2)
            # b=(G3*G5*H2+G1*G4*H5-G1*G6*H4-G2*G3*H6)/(gamma1)+(G6*H2*H3+G4*H1*H6-G5*H1*H4-G2*H3*H5)/gamma2
            # c=G1*G6*H3*H5+G1*G3*H1*H3+G5*G6*H5*H6+G5*G6*H2*H4+G3*G5*H1*H6+G2*G4*H2*H4-G2*G3*H2*H3-G1*G4*H1*H4+G2*G4*H5*H6
            # wV[:, idx] = np.roots([1/(gamma1**2*gamma2**2), 0, a*mu0**2, b*mu0**3, c*mu0**4])
        return wV
    
    def GetPhisSAFM(self):  
        phi1x0 = wrapAngle(self.phiAnis1 + 0.1)
        phi2x0 = wrapAngle(self.phiAnis2 + 0.1)
        result = minimize(self.GetFreeEnergySAFM, x0=[phi1x0, phi2x0], tol=1e-20, method='Nelder-Mead', bounds=((0, 2*np.pi), (0, 2*np.pi)))
        phis = wrapAngle(result.x)
        return phis
    def GetFreeEnergySAFM(self, phis):
        phiAnis1 = self.phiAnis1 #EA along x direction
        phiAnis2 = self.phiAnis2 #EA along x direction
        theta1 = np.pi/2 #No OOP magnetization
        theta2 = np.pi/2
        Ks1 = 0  #No surface anisotropy
        Ks2 = 0
        
        phi1, phi2 = phis
        H = self.Bext/mu0
        EJ1 = -self.Jbl*(np.sin(theta1)*np.sin(theta2)*np.cos(wrapAngle(phi1-phi2))+np.cos(theta1)*np.cos(theta2)) - self.Jbq*(np.sin(theta1)*np.sin(theta2)*np.cos(wrapAngle(phi1-phi2))+np.cos(theta1)*np.cos(theta2))**2
        
        Eaniso1 = -(2*np.pi*self.Ms**2-Ks1)*np.sin(theta1)**2-self.Ku*np.sin(theta1)**2*np.cos(wrapAngle(phi1-phiAnis1))**2
        Eaniso2 = -(2*np.pi*self.Ms2**2-Ks2)*np.sin(theta2)**2-self.Ku*np.sin(theta2)**2*np.cos(wrapAngle(phi2-phiAnis2))**2
        
        E = EJ1 + self.d1*(-self.Ms*mu0*H*(np.sin(theta1)*np.sin(self.theta)*np.cos(wrapAngle(phi1-self.phi)) + np.cos(theta1)*np.cos(self.theta)) + Eaniso1) + self.d2*(-self.Ms2*mu0*H*(np.sin(theta2)*np.sin(self.theta)*np.cos(wrapAngle(phi2-self.phi)) + np.cos(theta2)*np.cos(self.theta)) + Eaniso2)
        return E
    def GetFreeEnergySAFMOOP(self, thetas):
        phiAnis = np.pi/2 #EA along x direction
        phi1 = np.pi/2 #No OOP magnetization
        phi2 = -np.pi/2
        Ks1 = 0#No surface anisotropy
        Ks2 = 0
        
        theta1, theta2 = thetas
        H = self.Bext/mu0
        EJ1 = -self.Jbl*(np.sin(theta1)*np.sin(theta2)*np.cos(wrapAngle(phi1-phi2))+np.cos(theta1)*np.cos(theta2)) - self.Jbq*(np.sin(theta1)*np.sin(theta2)*np.cos(wrapAngle(phi1-phi2))+np.cos(theta1)*np.cos(theta2))**2
        
        Eaniso1 = -(0.5*mu0*self.Ms**2-Ks1)*np.sin(theta1)**2-self.Ku*np.sin(theta1)**2*np.cos(wrapAngle(phi1-phiAnis))**2
        Eaniso2 = -(0.5*mu0*self.Ms2**2-Ks2)*np.sin(theta2)**2-self.Ku*np.sin(theta2)**2*np.cos(wrapAngle(phi2-phiAnis))**2
        
        E = EJ1 + self.d1*(-self.Ms*mu0*H*(np.sin(theta1)*np.sin(self.theta) + np.cos(theta1)*np.cos(self.theta)) + Eaniso1) + self.d2*(-self.Ms2*mu0*H*(np.sin(theta2)*np.sin(self.theta) + np.cos(theta2)*np.cos(self.theta)) + Eaniso2)
        return E
        
        # bounds=((-np.pi, np.pi), (-np.pi, np.pi))
    
    
        
#    def GetDispersionNumeric(self, n=0, nc=-1, nT=0):
#        """ Gives frequencies for defined k (Dispersion relation) \n
#        The returned value is in the rad Hz \n
#        Arguments: \n
#        n -- Quantization number \n
#        nc(optional) -- Second quantization number. Used for hybridization \n
#        nT(optional) -- Waveguide (transversal) quantization number """
#        if nc == -1:
#            nc = n
#        if self.boundaryCond == 4:
#            kappa = self.GetPartiallyPinnedKappa(n)
#        else:
#            kappa = n*np.pi/self.d
#        kxi = np.sqrt(self.kxi**2 + (nT*np.pi/self.weff)**2)
#        k = np.sqrt(np.power(kxi,2) + kappa**2)
#        phi = np.arctan((nT*np.pi/self.weff)/self.kxi) - self.phi
#        Pnn = self.GetPropagationVector(n = n, nc = nc, nT = nT)
#        
#        A = np.cos(self.phi)**2 - np.sin(self.theta)**2*(1 - np.cos(self.phi)**2)
#        B = -2*np.cos(self.phi)*np.sin(2*self.theta)
#        C = np.cos(self.theta)*np.sin(self.phi)*np.cos(self.phi)
#        D = -2*np.sin(self.theta)*np.sin(self.phi)
#        E = np.sin(self.phi)**2
#        
#        Fnn = Pnn + np.power(np.sin(self.theta),2)*(1-Pnn*(1+np.power(np.cos(phi),2)) + self.wM*(Pnn*(1 - Pnn)*np.power(np.sin(phi),2))/(self.w0 + self.A*self.wM*np.power(k,2)))
#        f = np.sqrt((self.w0 + self.A*self.wM*np.power(k,2))*(self.w0 + self.A*self.wM*np.power(k,2) + self.wM*Fnn))
#        return f
    def GetGroupVelocity(self, n=0, nc=-1, nT=0):
        """ Gives group velocities for defined k \n
        The group velocity is computed as vg = dw/dk
        Arguments: \n
        n -- Quantization number \n
        nc(optional) -- Second quantization number. Used for hybridization \n
        nT(optional) -- Waveguide (transversal) quantization number 
        """
        if nc == -1:
            nc = n
        f = self.GetDispersion(n = n, nc = nc, nT = nT)
        vg = np.diff(f)/(self.kxi[2]-self.kxi[1])
        return(vg)
    def GetLifetime(self, n=0, nc=-1, nT=0):
        """ Gives lifetimes for defined k \n
        lifetime is computed as tau = (alpha*w*dw/dw0)^-1
        Arguments: \n
        n -- Quantization number \n
        nc(optional) -- Second quantization number. Used for hybridization \n
        nT(optional) -- Waveguide (transversal) quantization number """
        if nc == -1:
            nc = n
        w0Ori = self.w0
        self.w0 = w0Ori*0.9999999
        dw0p999 = self.GetDispersion(n = n, nc = nc, nT = nT)
        self.w0 = w0Ori*1.0000001
        dw0p001 = self.GetDispersion(n = n, nc = nc, nT = nT)
        self.w0 = w0Ori
        lifetime = ((self.alpha*self.GetDispersion(n = n, nc = nc, nT = nT) + self.gamma*self.mu0dH0)*(dw0p001 - dw0p999)/(w0Ori*1.0000001 - w0Ori*0.9999999))**-1
        return lifetime
    def GetLifetimeSAFM(self, n):
        """ Gives lifetimes for defined k \n
        lifetime is computed as tau = (alpha*w*dw/dw0)^-1
        Arguments: \n
        n -- Quantization number \n
        """
        BextOri = self.Bext
        self.Bext = BextOri-0.001
        dw0p999 = self.GetDispersionSAFMNumeric()
        self.Bext = BextOri+0.001
        dw0p001 = self.GetDispersionSAFMNumeric()
        self.Bext = BextOri
        w = self.GetDispersionSAFMNumeric()
        lifetime = ((self.alpha*w[n] + self.gamma*self.mu0dH0)*(dw0p001[n] - dw0p999[n])/(0.2))**-1
        return lifetime
    def GetPropLen(self, n=0, nc=-1, nT=0):
        """ Give propagation lengths for defined k \n
        propagation length is computed as lambda = vg*tau
        Arguments: \n
        n -- Quantization number \n
        nc(optional) -- Second quantization number. Used for hybridization \n
        nT(optional) -- Waveguide (transversal) quantization number """
        if nc == -1:
            nc = n
        propLen = self.GetLifetime(n = n, nc = nc, nT = nT)[0:-1]*self.GetGroupVelocity(n = n, nc = nc, nT = nT)
        return propLen
    def GetSecondPerturbation(self, n, nc):
        """ Give degenerate dispersion relation based on the secular equation 54 \n
        Arguments: \n
        n -- Quantization number \n
        nc -- Quantization number of crossing mode \n """
        if self.boundaryCond == 4:
            kappa = self.GetPartiallyPinnedKappa(n)
            kappac = self.GetPartiallyPinnedKappa(nc)
        else:
            kappa = n*np.pi/self.d
            kappac = nc*np.pi/self.d
        Om = self.w0 + self.wM*self.A*(kappa**2 + self.kxi**2);
        Omc = self.w0 + self.wM*self.A*(kappac**2 + self.kxi**2);
        Pnnc = self.GetPropagationVector(n = n, nc = nc)
        Pnn = self.GetPropagationVector(n = n, nc = n) 
        Pncnc = self.GetPropagationVector(n = nc, nc = nc)
        Qnnc = self.GetPropagationQVector(n = n, nc = nc)
        wnn = self.GetDispersion(n = n, nc = n)
        wncnc = self.GetDispersion(n = nc, nc = nc)
        if self.theta == 0:
            wdn = np.sqrt(wnn**2+wncnc**2-np.sqrt(wnn**4-2*wnn**2.*wncnc**2+wncnc**4+4*Om*Omc*Pnnc**2*self.wM**2))/np.sqrt(2);
            wdnc = np.sqrt(wnn**2+wncnc**2+np.sqrt(wnn**4-2*wnn**2.*wncnc**2+wncnc**4+4*Om*Omc*Pnnc**2*self.wM**2))/np.sqrt(2);
        elif self.theta == np.pi/2:          
            wdn = (1/np.sqrt(2))*(np.sqrt(wnn**2 + wncnc**2 - 2*Pnnc**2*self.wM**2 - 8*Qnnc**2*self.wM**2 - np.sqrt(wnn**4 + wncnc**4 - 4*(Pnnc**2 + 4*Qnnc**2)*wncnc**2*self.wM**2 - 
                                 2*wnn**2*(wncnc**2 + 2*(Pnnc**2 + 4*Qnnc**2)*self.wM**2) + 4*self.wM**2*(Om*(Pnnc**2 + 4*Qnnc**2)*(2*Omc + self.wM) + self.wM*(Omc*(Pnnc**2 + 
                                           4*Qnnc**2) +  4*(Pncnc + Pnn - 2*Pncnc*Pnn)*Qnnc**2*self.wM + Pnnc**2*(1 - Pnn + Pncnc*(-1 + 2*Pnn) + 16*Qnnc**2)*self.wM))))) 
            wdnc = (1/np.sqrt(2))*(np.sqrt(wnn**2 + wncnc**2 - 2*Pnnc**2*self.wM**2 - 8*Qnnc**2*self.wM**2 + np.sqrt(wnn**4 + wncnc**4 - 4*(Pnnc**2 + 4*Qnnc**2)*wncnc**2*self.wM**2 - 
                                 2*wnn**2*(wncnc**2 + 2*(Pnnc**2 + 4*Qnnc**2)*self.wM**2) + 4*self.wM**2*(Om*(Pnnc**2 + 4*Qnnc**2)*(2*Omc + self.wM) + self.wM*(Omc*(Pnnc**2 + 
                                           4*Qnnc**2) +  4*(Pncnc + Pnn - 2*Pncnc*Pnn)*Qnnc**2*self.wM + Pnnc**2*(1 - Pnn + Pncnc*(-1 + 2*Pnn) + 16*Qnnc**2)*self.wM)))))      
        else:
            raise Exception("Sorry, for degenerate perturbation you have to choose theta = pi/2 or 0.") 
        return(wdn, wdnc)
    def GetDensityOfStates(self, n=0, nc=-1, nT=0):
        """ Give density of states for \n
        Density of states is computed as DoS = 1/vg \n
        Arguments: \n
        n -- Quantization number \n
        nc(optional) -- Second quantization number. Used for hybridization \n
        nT(optional) -- Waveguide (transversal) quantization number """
        if nc == -1:
            nc = n
        DoS = 1/self.GetGroupVelocity(n = n, nc = nc, nT = nT)
        return DoS
    def GetExchangeLen(self):
        exLen = np.sqrt(self.A)
        return exLen
    def GetAk(self):
        gk = 1 - (1 - np.exp(-self.kxi*self.d))
        return(self.w0 + self.wM*self.A*self.kxi**2 + self.wM/2*(gk*np.sin(self.phi)**2+(1-gk)))
    def GetBk(self):
        gk = 1 - (1 - np.exp(-self.kxi*self.d))
        return(self.wM/2*(gk*np.sin(self.phi)**2 - (1 - gk)))
    def GetEllipticity(self):
        return(2*abs(self.GetBk())/(self.GetAk() + abs(self.GetBk())))
    def GetCouplingParam(self):
        return(self.gamma*self.GetBk()/(2*self.GetDispersion(n=0, nc=0, nT=0)))
    def GetThresholdField(self):
        return(2*np.pi/(self.GetLifetime(n=0, nc=0, nT=0)*abs(self.GetCouplingParam())))
def wavenumberToWavelength(wavenumber):
    """ Convert wavelength to wavenumber
    lambda = 2*pi/k     
    Arguments: \n
    wavenumber -- wavenumber of the wave (rad/m)
    Return: \n
    wavelength (m)"""
    return 2*np.pi/np.array(wavenumber)
def wavelengthTowavenumber(wavelength):
    """ Convert wavenumber to wavelength
    k = 2*pi/lambda 
    Arguments: \n
    wavelength -- wavelength of the wave (m)
    Return: \n
    wavenumber (rad/m)"""
    return 2*np.pi/np.array(wavelength)
def wrapAngle(angle):
    # return np.mod(angle + np.pi, 2 * np.pi) - np.pi
    return np.mod(angle, 2 * np.pi)
      
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
    def __init__(self, Ms, Aex, alpha, mu0dH0 = 0, gamma = 28.1*2*np.pi*1e9, Ku = 0):
        self.Ms = Ms
        self.Aex = Aex
        self.alpha = alpha
        self.gamma = gamma
        self.mu0dH0 = mu0dH0
        self.Ku = Ku
#Predefined materials
NiFe = Material(Ms = 800e3, Aex = 16e-12, alpha = 70e-4, gamma = 28.8*2*np.pi*1e9, Ku = 0)
CoFeB = Material(Ms = 1250e3, Aex = 15e-12, alpha = 40e-4, gamma=30*2*np.pi*1e9)
FeNi = Material(Ms = 1410e3, Aex = 11e-12, alpha = 80e-4)
YIG = Material(Ms = 140e3, Aex = 3.6e-12, alpha = 1.5e-4, gamma = 28*2*np.pi*1e9, mu0dH0 = 0.18e-3)