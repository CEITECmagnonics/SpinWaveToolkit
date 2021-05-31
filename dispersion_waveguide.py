# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 17:03:34 2020

@author: igort
"""

import SpinWaveToolkit as SWT
import numpy as np
import matplotlib.pyplot as plt

#Here is an example of code    
kxi = np.linspace(1, 10e6, 150)

# for w in weffe:
FlesFeNi = SWT.Material(Aex = 3.5e-12, Ms = 140e3, gamma = 1.76*1e11, alpha = 5e-3)
FeNiChar = SWT.DispersionCharacteristic(kxi = kxi, theta = np.pi/2, phi = 0, d = 100e-9, boundaryCond = 4, weff = 1e-6, Bext = 100e-3, material = FlesFeNi)
FeNiCharBV = SWT.DispersionCharacteristic(kxi = kxi, theta = np.pi/2, phi = 0, d = 100e-9, boundaryCond = 1, weff = 1e-6, Bext = 100e-3, material = FlesFeNi)
f00 = FeNiChar.GetDispersion(n=1, nc=1, nT=1)*1e-9/(2*np.pi) #GHz
f00BV = FeNiCharBV.GetDispersion(n=1, nc=1, nT=0)*1e-9/(2*np.pi) #GHz

plt.plot(kxi*1e-6, f00, kxi*1e-6, f00BV);
plt.xlabel('kxi (rad/um)');
plt.show()

#ffilm = FeNiChar.GetDispersion(n=0, nc=0, nT=0)*1e-9/(2*np.pi) #GHz
#
#plt.plot(kxi*1e-6, f00, kxi*1e-6, f00Fl,);
#plt.xlabel('kxi (rad/um)');
#plt.ylabel('Frequency (GHz)');
#plt.legend(['nT = 1, Bext = 0 mT, weff = 1.5 um','nT = 0'])
#plt.title('Dispersion relation of FeNi, totally pinned')
#plt.show()


