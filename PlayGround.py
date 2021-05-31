# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 16:01:13 2020

@author: Ondrej Wojewoda
"""

import SpinWaveToolkit as SWT
import numpy as np
import matplotlib.pyplot as plt

#Here is an example of code    
kxi = np.linspace(0, 100e6, 150)

# Degeneration of the 00 and 22 mode in CoFeB
n = 0
nc = 1
MagnetronCoFeB = SWT.Material(Aex = 13.5e-12, Ms = 1.24e6, gamma = 30.8*2*np.pi*1e9, alpha = 5e-3)
test = SWT.Material(Aex = 13e-12, Ms = 801e3, gamma = 29*2*np.pi*1e9, alpha = 5e-3)
CoFeBchar = SWT.DispersionCharacteristic(kxi = kxi, theta = np.pi/2, phi = np.deg2rad(90), d = 50e-9, boundaryCond = 1, dp = 8e6, Bext = 10e-3, material = SWT.CoFeB)

#T = CoFeBchar.GetTVector(1,1,50,50)
w = CoFeBchar.GetDispersion(n=0)*1e-9/(2*np.pi)
vg = CoFeBchar.GetGroupVelocity(n=0)*1e-3
propLen = CoFeBchar.GetPropLen(n=0)*1e6

plt.plot( kxi*1e-6, w);
plt.xlabel('kxi (rad/um)');
plt.ylabel('Frequency (GHz)');
plt.title('Dispersion relation CoFeB')
plt.show()

plt.plot(kxi[0:-1]*1e-6, vg);
plt.xlabel('kxi (rad/um)');
plt.ylabel('Group velocity (um/ns)');
plt.show()

plt.plot(kxi[0:-1]*1e-6, propLen);
plt.xlabel('kxi (rad/um)');
plt.ylabel('Propagation lenght (um)');
plt.show()

#[wd02, wd20] =  [w*1e-9/(2*np.pi) for w in CoFeBchar.GetSecondPerturbation(n=n, nc=nc)]
#
#plt.plot(kxi*1e-6, w00, kxi*1e-6, w22, kxi*1e-6, wd02, kxi*1e-6, wd20);
#plt.plot(kxi*1e-6, w00, kxi*1e-6, w22);
#plt.xlabel('kxi (rad/um)');
#plt.ylabel('Frequency (GHz)');
#plt.legend([str(n) + str(n), str(nc) + str(nc), str(n) + str(nc), str(nc) + str(n)])
#plt.title('Dispersion relation of degenerate modes in CoFeB, dp = {:.2e}'.format(CoFeBchar.dp))
#plt.show()
#
#kxi = np.linspace(1, 60e6, 150)
#NiFeChar = SWT.DispersionCharacteristic(kxi = kxi, theta = np.pi/2, phi = np.pi/2, d = 30e-9, boundaryCond = 1, Bext = 0.100, material = MagnetronCoFeB)
#NiFeCharBV = SWT.DispersionCharacteristic(kxi = kxi, theta = np.pi/2, phi = 0, d = 30e-9, boundaryCond = 1, Bext = 0.100, material = MagnetronCoFeB)
#
#f00 = NiFeChar.GetDispersion(n=0)*1e-9/(2*np.pi) 
#f11BV = NiFeCharBV.GetDispersion(n=0)*1e-9/(2*np.pi) 
#f11 = NiFeChar.GetDispersion(n=2)*1e-9/(2*np.pi) 
#
#vg00 = NiFeChar.GetGroupVelocity(n=0)*1e-3 # um/ns
#vg00BV = NiFeChar.GetGroupVelocity(n=1)*1e-3 # um/ns
#tau00 = NiFeChar.GetLifetime(n=0)*1e9 #ns
#tau00BV = NiFeCharBV.GetLifetime(n=0)*1e9 #ns
#propLen00 = NiFeChar.GetPropLen(n=0)*1e6 #um
#propLen00BV = NiFeCharBV.GetPropLen(n=0)*1e6 #um
#DoS00 = NiFeChar.GetDensityOfStates(n=0)*1e6 #um
#
#vg11 = NiFeChar.GetGroupVelocity(n=1)*1e-3 # um/ns
#tau11 = NiFeChar.GetLifetime(n=1)*1e9 #ns
#propLen11 = NiFeChar.GetPropLen(n=1)*1e6 #um
#DoS11 = NiFeChar.GetDensityOfStates(n=1)*1e6 #um
#
#plt.plot(kxi*1e-6, f00, kxi*1e-6, f11BV);
#plt.xlabel('kxi (rad/um)');
#plt.ylabel('Frequency (GHz)');
#plt.legend(['DE', 'BV'])
#plt.title('Dispersion relation of CoFeB')
#plt.show()
#
#plt.plot(kxi[0:-1]*1e-6, vg00, kxi[0:-1]*1e-6, vg00BV);
#plt.xlabel('kxi (rad/um)');
#plt.ylabel('Group velocity (um/ns)');
#plt.legend(['DE', 'BV'])
#plt.title('Group velocities of NiFe')
#plt.show()
#
#plt.plot(kxi*1e-6, tau00, kxi*1e-6, tau00BV);
#plt.xlabel('kxi (rad/um)');
#plt.ylabel('Lifetime (ns)');
#plt.legend(['DE', 'BV'])
#plt.title('Lifetime of NiFe')
#plt.show()
#
#plt.plot(kxi[0:-1]*1e-6, propLen00, kxi[0:-1]*1e-6, propLen00BV);
#plt.xlabel('kxi (rad/um)');
#plt.ylabel('Propagation lenght (um)');
#plt.legend(['DE', 'BV'])
#plt.title('Propagation lengths of NiFe')
#plt.show()
#
##Sum of the density of states of the first two thickness modes
#fDoS = np.unique(np.concatenate((f00,f11)))
#DoSsum = np.interp(fDoS, f00[0:-1], DoS00, left=0, right=0) + np.interp(fDoS, f11[0:-1], DoS11, left=0, right=0)
#
#plt.semilogy(f00[0:-1], DoS00, f11[0:-1], DoS11, fDoS, DoSsum);
#plt.xlabel('Frequency (GHz)');
#plt.ylabel('Density of states()');
#plt.legend(['00', '11', 'Sum'])
#plt.title('Density of states NiFe n=0,1')
#plt.xlim([np.min([f00, f11]), np.max([f00, f11])])
#plt.show()

#phiInter = np.linspace(0, np.pi/2, 180)
#
#f00Inter = [0]
#DoSsum = [0]
#kxi = np.linspace(1, 100e6, 150)
#for phi in phiInter:
#    NiFeChar = SWT.DispersionCharacteristic(kxi = kxi, theta = np.pi/2, phi = phi, d = 30e-9, boundaryCond = 1, Bext = 0.55, material = SWT.NiFe)
#    f00 = NiFeChar.GetDispersion(n=0)*1e-9/(2*np.pi)
#    f00Inter = np.unique(np.concatenate((f00Inter,f00)))
#    DoS00 = NiFeChar.GetDensityOfStates(n=0)*1e6 #um
#    DoSsum = np.interp(f00Inter, f00Inter[0:-1], DoSsum, left=0, right=0) + np.interp(f00Inter, f11[0:-1], DoS00, left=0, right=0)
#    


#kxi = np.linspace(1, 30e6, 150)
#FeNiCharDE = SWT.DispersionCharacteristic(kxi = kxi, theta = np.pi/2, phi = np.pi/2, d = 12e-9, boundaryCond = 1, weff = 0.1e-6, Bext = 25e-3, material = SWT.FeNi)
#FeNiCharBV = SWT.DispersionCharacteristic(kxi = kxi, theta = np.pi/2, phi = 0, d = 12e-9, boundaryCond = 1, weff = 0.1e-6, Bext = 155e-3, material = SWT.FeNi)
#f1DE = FeNiCharDE.GetDispersion(nT=1)*1e-9/(2*np.pi) #GHz
#f1BV = FeNiCharBV.GetDispersion(nT=1)*1e-9/(2*np.pi) #GHz
#f2DE = FeNiCharDE.GetDispersion(nT=2)*1e-9/(2*np.pi) #GHz
#f2BV = FeNiCharBV.GetDispersion(nT=2)*1e-9/(2*np.pi) #GHz
#
#plt.plot(kxi*1e-6, f1DE, kxi*1e-6, f2DE, kxi*1e-6, f1BV, kxi*1e-6, f2BV);
#plt.xlabel('kxi (rad/um)');
#plt.ylabel('Frequency (GHz)');
#plt.legend(['DE - 1', 'DE - 2', 'BV - 1', 'BV - 2'])
#plt.title('Dispersion of FeNi waveguides, Bext = 155 mT, weff = 0.1 um')
#plt.show()
