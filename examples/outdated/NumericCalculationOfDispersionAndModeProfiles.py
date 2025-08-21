# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 09:58:03 2024
This example shows how to use numeric model from Tacchi
In this example the hybridized dispersion relation of 80nm-thick CoFeB slab 
is calculated
Afterwards the eigenvectors are used to calculate the mode profiles of spin waves

@author: Ondrej Wojewoda
"""
import sys
sys.path.insert(1, '../')

import SpinWaveToolkit as SWT
import numpy as np
import matplotlib.pyplot as plt

kxi = np.linspace(1, 80e6, 150)
d = 80e-9
NiFeCharDE = SWT.DispersionCharacteristic(kxi = kxi, theta = np.pi/2, phi = np.pi/2, d = d, boundary_cond = 1, Bext = 550e-3, material = SWT.CoFeB)

fDE, VDE = NiFeCharDE.GetDispersionTacchi()

NiFeCharBV = SWT.DispersionCharacteristic(kxi = kxi, theta = np.pi/2, phi = 0*np.pi/2, d = d, boundary_cond = 1, Bext = 550e-3, material = SWT.CoFeB)

fBV, VBV = NiFeCharDE.GetDispersionTacchi()

mkSWprofiles00 = VDE[:,3,:] #Here you can select of which mode you want to plot spin wave profiles

z = np.linspace(0, d, 50) #Set the grid for mode profile calculation
SWprofile00OOP = np.zeros((np.size(z,0), np.size(kxi,0)))
SWprofile00IP = np.zeros((np.size(z,0), np.size(kxi,0)))
for idx, k in enumerate(kxi):
    SWprofile00OOP[:,idx] =  mkSWprofiles00[0,idx]*1/np.sqrt(2)*1 + mkSWprofiles00[2,idx]*np.cos(1*np.pi*z/d) + mkSWprofiles00[4,idx]*np.cos(2*np.pi*z/d)
    SWprofile00IP[:,idx] =  mkSWprofiles00[1,idx]*1/np.sqrt(2)*1 + mkSWprofiles00[3,idx]*np.cos(1*np.pi*z/d) + mkSWprofiles00[5,idx]*np.cos(2*np.pi*z/d)
    
# Plot of the mode profiles for different k-vectors of fundamental mode
plt.figure(0)
kToPlot = np.round(np.linspace(0,149, 5))
for idx in kToPlot:
    idx = int(idx)
    plt.plot(z*1e9, SWprofile00OOP[:,idx]);
    plt.plot(z*1e9, SWprofile00IP[:,idx], linestyle='--');
plt.xlabel('z (nm)');
plt.ylabel('Spin wave amplitude ()');
plt.ylim((-1,1))
plt.title('Mode profile')
plt.show()

plt.figure(1)
plt.plot(kxi*1e-6, fDE[3]*1e-9/(2*np.pi), kxi*1e-6, fBV[4]*1e-9/(2*np.pi), kxi*1e-6, fBV[5]*1e-9/(2*np.pi)  );
plt.xlabel('kxi (rad/um)');
plt.ylabel('Frequency (GHz)');
plt.title('Dispersion relation')
plt.show()