# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 16:01:13 2020

@author: Ondrej Wojewoda
"""

import SpinWaveToolkit as SWT
import numpy as np
import matplotlib.pyplot as plt

#Here is an example of code    
kxi = np.linspace(1, 5e6, 150)

# Degeneration of the 00 and 22 mode in CoFeB
MagnetronCoFeB = SWT.Material(Aex = 13.5e-12, Ms = 1.24e6, gamma = 30.8*2*np.pi*1e9, alpha = 5e-3)
CoFeBchar = SWT.DispersionCharacteristic(kxi = kxi, theta = np.pi/2, phi = np.pi/2, d = 100e-9, boundaryCond = 4, dp = 2.2e7, Bext = 0.08, material = MagnetronCoFeB)
w00 = CoFeBchar.GetDispersion(n=0)*1e-9/(2*np.pi)
w22 = CoFeBchar.GetDispersion(n=2)*1e-9/(2*np.pi)
[wd00, wd22] =  [w*1e-9/(2*np.pi) for w in CoFeBchar.GetSecondPerturbation(n=0, nc=2)]

plt.plot(kxi*1e-6, w00, kxi*1e-6, w22, kxi*1e-6, wd00, kxi*1e-6, wd22);
plt.xlabel('kxi (rad/um)');
plt.ylabel('Frequency (GHz)');
plt.legend(['00', '22', '02', '20'])
plt.title('Dispersion relation of degenerate modes in CoFeB')
plt.show()

NiFeChar = SWT.DispersionCharacteristic(kxi = kxi, theta = np.pi/2, phi = np.pi/2, d = 30e-9, boundaryCond = 1, Bext = 0.2, material = SWT.NiFe)

f00 = NiFeChar.GetDispersion(n=0)*1e-9/(2*np.pi) 
f11= NiFeChar.GetDispersion(n=1)*1e-9/(2*np.pi) 
vg00 = NiFeChar.GetGroupVelocity(n=0)*1e-3 # um/ns
tau00 = NiFeChar.GetLifetime(n=0)*1e9 #ns
propLen00 = NiFeChar.GetPropLen(n=0)*1e6 #um

vg11 = NiFeChar.GetGroupVelocity(n=1)*1e-3 # um/ns
tau11 = NiFeChar.GetLifetime(n=1)*1e9 #ns
propLen11 = NiFeChar.GetPropLen(n=1)*1e6 #um

plt.plot(kxi*1e-6, f00, kxi*1e-6, f11);
plt.xlabel('kxi (rad/um)');
plt.ylabel('Frequency (GHz)');
plt.legend(['00', '11'])
plt.title('Dispersion relation of NiFe n=0,1')
plt.show()

plt.plot(kxi[0:-1]*1e-6, vg00, kxi[0:-1]*1e-6, vg11);
plt.xlabel('kxi (rad/um)');
plt.ylabel('Group velocity (um/ns)');
plt.legend(['00', '11'])
plt.title('Group velocities of NiFe n=0,1')
plt.show()

plt.plot(kxi[0:-1]*1e-6, propLen00, kxi[0:-1]*1e-6, propLen11);
plt.xlabel('kxi (rad/um)');
plt.ylabel('Propagation lenght (um)');
plt.legend(['00', '11'])
plt.title('Propagation lengths of NiFe n=0,1')
plt.show()