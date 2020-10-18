# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 16:01:13 2020

@author: ondre
"""

import SpinWaveToolkit as SWT
import numpy as np

#Here is an example of code    
kxi = np.linspace(1e-12, 20e6, 150)

NiFeChar = SWT.DispersionCharacteristic(kxi = kxi, theta = np.pi/2, phi = np.pi/2,n =  0, d = 100e-9, weff = 2e-6, nT = 0, boundaryCond = 2, Bext = 20e-3, material = SWT.Material(Ms = 1.570/SWT.mu0, Aex = 15e-12, alpha = 40e-4, gamma=30*2*np.pi*1e9))
DispPy = NiFeChar.GetDispersion()*1e-9/(2*np.pi) #GHz
vgPy = NiFeChar.GetGroupVelocity()*1e-3 # km/s
lifetimePy = NiFeChar.GetLifetime()*1e9 #ns
propLen = NiFeChar.GetPropLen()*1e6 #um