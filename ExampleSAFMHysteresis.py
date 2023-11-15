import SpinWaveToolkit as SWT
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

Bexts = np.linspace(0.3, -0.3, 101)
kxi = np.linspace(0.1, 6e6, 201)
M1 = np.zeros(np.size(Bexts))
M2 = np.zeros(np.size(Bexts))

phi=np.deg2rad(0)
# J = -1.1e-3
Jbl = -0.706e-3
Jbq = -0.236e-3*0
# Jbq = (J-Jbl)/2
Ku = 1.5e3
phi1x0 = SWT.wrapAngle(phi)
phi2x0 = SWT.wrapAngle(phi)

fac = 4

CoFeB = SWT.Material(Ms = 1329e3, Aex = 15e-12, alpha = 40e-4, gamma=30*2*np.pi*1e9)

for i, Bext in enumerate(Bexts):
    
    SAF = SWT.DispersionCharacteristic(kxi = kxi, theta = np.deg2rad(90), phi = phi, d = 15e-9, boundaryCond = 1,  Bext = Bext, material = CoFeB, Ku = Ku, Jbl=Jbl, Jbq=Jbq, s=0.6e-9, d2=15e-9)
    phi1x0 = phi1x0
    phi2x0 = phi2x0
    result = minimize(SAF.GetFreeEnergySAFM, x0=[phi1x0, phi2x0], tol=1e-20, method='Powell', bounds=((0, 2*np.pi), (0, 2*np.pi)))
    phi1x0, phi2x0 = SWT.wrapAngle(result.x)
    M1[i] = (np.cos(phi1x0-phi) + fac*np.cos(phi2x0-phi))
    M2[i] = (np.sin(phi1x0-phi) + fac*np.sin(phi2x0-phi))

plt.figure(11)
Bexport = np.append(-Bexts, Bexts)*1e3
M2export = np.append(-M2,M2)/np.max(abs(M2))
M1export = np.append(-M1,M1)/np.max(abs(M1))

plt.plot(np.append(-Bexts, Bexts)*1e3, np.append(-M1,M1)/np.max(abs(M1)), np.append(-Bexts, Bexts)*1e3, np.append(-M2,M2)/np.max(abs(M2)))

plt.xlabel('External field (mT)')
plt.ylabel('M (kA/m)');