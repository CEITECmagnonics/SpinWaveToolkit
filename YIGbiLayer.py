import SpinWaveToolkit as SWT
import numpy as np
import matplotlib.pyplot as plt

Jbl = 0
Jbq = 0

Ku = 10e3
Ku2 = 10e3
phi = np.deg2rad(0)

JblDyn = Jbl 
JbqDyn = Jbq

d1=15e-9
d2=d1
s=0.5e-9

Bext=5e-3

fac = 0

YIG = SWT.YIG

kxi = np.linspace(-20.001e6, 20e6, 201)  
YIGbiLayer=SWT.DispersionCharacteristic(kxi = kxi, theta = np.deg2rad(90), phi = phi, phiAnis1=np.deg2rad(-45), phiAnis2=np.deg2rad(-135), d = d1, d2=d2, boundaryCond = 1,  Bext = Bext, material = YIG, Ku = Ku, Ku2 = Ku2, Jbl=Jbl, Jbq=Jbq, s=s) 

fGal = YIGbiLayer.GetDispersionSAFMNumeric()/(1e9*2*np.pi) 
phi1, phi2 = np.rad2deg(YIGbiLayer.GetPhisSAFM())

plt.figure(3)
plt.plot(kxi*1e-6, fGal[2], kxi*1e-6, fGal[3])
plt.ylabel('Frequency (GHz)')
plt.xlabel('k(rad/um)')




# kxi = [-1, 0.01, 1]
# Bexts = np.linspace(-0.03, 0.03, 101)
# M1 = np.zeros(np.size(Bexts))
# M2 = np.zeros(np.size(Bexts))
# SAFMREA = np.zeros(np.size(Bexts))
# SAFMREA2 = np.zeros(np.size(Bexts))
# SAFMRHA = np.zeros(np.size(Bexts))
# for i, Bext in enumerate(Bexts):
#     SAFMEA = SWT.DispersionCharacteristic(kxi = kxi, theta = np.deg2rad(90), phi = np.deg2rad(0) , phiAnis=np.deg2rad(90), d = d1, boundaryCond = 1,  Bext = Bext, material = CoFeB, material2 = CoFeB2, Ku = Ku, Ku2 = Ku2, Jbl=Jbl, Jbq=Jbq, s=s, d2=d2, JblDyn=JblDyn, JbqDyn=JbqDyn)
#     SAFMHA = SWT.DispersionCharacteristic(kxi = kxi, theta = np.deg2rad(90), phi = np.deg2rad(90), phiAnis=np.deg2rad(90), d = d1, boundaryCond = 1,  Bext = Bext, material = CoFeB, material2 = CoFeB2, Ku = Ku, Ku2 = Ku2, Jbl=Jbl, Jbq=Jbq, s=s, d2=d2, JblDyn=JblDyn, JbqDyn=JbqDyn)
#     phi1, phi2 = SAFMHA.GetPhisSAFM()
#     M1[i] = (np.cos(phi1) + fac*np.cos(phi2))
#     M2[i] = (np.sin(phi1) + fac*np.sin(phi2))
  
    
#     fEA = SAFMEA.GetDispersionSAFMNumeric()/(1e9*2*np.pi)
#     fHA = SAFMHA.GetDispersionSAFMNumeric()/(1e9*2*np.pi)
#     SAFMREA[i] = fEA[2,1]
#     SAFMREA2[i] = fEA[3,1]
#     SAFMRHA[i] = fHA[2,1]

# # plt.figure(11)
# # plt.plot(np.append(Bexts, -Bexts)*1e3, np.append(M1,-M1)/np.max(np.abs(M1)), np.append(Bexts, -Bexts)*1e3, np.append(M2,-M2)/np.max(np.abs(M2)))
# # # plt.plot(Bexts,M1/1e3,Bexts, M2/1e3)
# # plt.xlabel('External field (mT)')
# # plt.ylabel('M (kA/m)');

# # plt.figure(12)
# # plt.plot(Bexts*1e3, SAFMREA, Bexts*1e3, SAFMREA2, VNAField, VNASAFMR, '.', VNAFieldHigh*1e3, VNASAFMRHigh, '.')
# # plt.ylabel('Frequency (GHz)')
# # plt.xlabel('External field (mT)');

# plt.figure(12)
# plt.plot(Bexts*1e3, SAFMREA, Bexts*1e3, SAFMREA2, VNAField, VNASAFMR, '.')
# plt.ylabel('Frequency (GHz)')
# plt.xlabel('External field (mT)');



# fFi = SAFiM.GetDispersionSAFMNumeric()/(1e9*2*np.pi)
# plt.figure(1)
# plt.plot(kxi*1e-6, f0, kxi*1e-6, f1, kxi*1e-6, f[0], kxi*1e-6, f[1])
# plt.ylabel('Frequency (GHz)')
# plt.xlabel('k(rad/um)');



# plt.figure(7)
# plt.plot(kxi*1e-6, wKS)
# plt.ylabel('Frequency (GHz)')
# plt.xlabel('k(rad/um)');

# plt.figure(6)
# plt.plot(kxi*1e-6, wTa[3], kxi*1e-6, wTa[4], kxi*1e-6, wTa[5])
# plt.ylabel('Frequency (GHz)')
# plt.xlabel('k(rad/um)');
# plt.ylim([0, 1.75])

# plt.figure(15)
# plt.plot(wKS[0:-1],GroupVel)
# plt.ylabel('Group velocity (um/ns)')
# plt.xlabel('Frequency (GHz)');

# plt.figure(8)
# plt.plot(kxi[0:-1]*1e-6, DecLen)
# # plt.plot(wKS[0:-1], DecLen)
# plt.xlabel('kxi (rad/um)');
# # plt.xlabel('Frequency (GHz)')
# plt.ylabel('Decay length (um)');

# plt.figure(5)
# plt.plot(kxi*1e-6, wKLifetime)
# # plt.plot(wKS[0:-1], DecLen)
# plt.xlabel('kxi (rad/um)');
# # plt.xlabel('Frequency (GHz)')
# plt.ylabel('Lifetime (ns)');

# bexts = np.linspace(start=100e-3, stop=0e-3, num=201)
# WaveB = np.array(bexts)

# for idx, bext in enumerate(bexts):
#     NiFeBextSweep = SWT.DispersionCharacteristic(kxi = kxi, theta = np.deg2rad(90), phi = np.deg2rad(90), d = 34e-9, boundaryCond = 1, dp = 1e7, Bext = bext, material = NiFe)
#     wBextSweep = NiFeBextSweep.GetDispersion(n=0, nc=0, nT=0)/(1e9*2*np.pi)
#     # ind = wBextSweep[np.abs(wBextSweep-14).argmin()]
#     ind = min(range(len(wBextSweep)), key=lambda i: abs(wBextSweep[i]-14))
#     WaveB[idx] = 2*np.pi/kxi[ind]

# plt.figure(8)
# plt.plot(bexts*1e3, WaveB*1e9)
# plt.xlabel('External field (mT)')
# plt.ylabel('Wavelength (nm)')


# ell = NiFeTest.GetEllipticity()

# plt.plot(kxi*1e-6, ell)
# plt.xlabel('kxi (rad/um)')
# plt.ylabel('Ellipticity ()')
# plt.show()

# ThresField = NiFeUP.GetThresholdField()*1e3

# plt.figure(3)
# # plt.plot(kxi*1e-6, ThresField)
# plt.plot(wKS, ThresField)
# plt.xlabel('kxi (rad/um)')
# plt.xlabel('Frequency (GHz)')
# plt.ylabel('Threshold field (mT)')
# plt.ylim(0, 100)
# plt.show()

# wKS1 = NiFeTest.GetDispersion(n=1)/(1e9*2*np.pi)
# wKSUP1 = NiFeUP.GetDispersion(n=1)/(1e9*2*np.pi)

# plt.figure(5)
# # plt.plot(kxi*1e-6, w[3,:], kxi*1e-6, w[4,:], kxi*1e-6, w[5,:]);
# plt.plot(kxi*1e-6, wKS, kxi*1e-6, wKS1, kxi*1e-6, wKS2);
# # # # # plt.plot(kxi*1e-6, wKS, kxi*1e-6, wKS1, kxi*1e-6, wKS2);
# plt.xlabel('kxi (rad/um)');
# plt.ylabel('Frequency (GHz)');
# # plt.legend(['n = 1', 'n = 3', 'n = 0', 'n=3'])
# # plt.title('Unpinned BC - Numeric vs Perturbation theory')
# plt.show()

# plt.figure(6)
# # plt.plot(kxi*1e-6, w[3,:], kxi*1e-6, w[4,:], kxi*1e-6, w[5,:]);
# plt.plot(kxi[0:-1]*1e-6, DecLen, kxi[0:-1]*1e-6, DecLen1, kxi[0:-1]*1e-6, DecLenTF,);
# # # # # plt.plot(kxi*1e-6, wKS, kxi*1e-6, wKS1, kxi*1e-6, wKS2);
# plt.xlabel('kxi (rad/um)');
# plt.ylabel('Decay length (um)');
# plt.legend(['n = 1', 'n = 3', 'n = 0', 'n=3'])
# # plt.title('Unpinned BC - Numeric vs Perturbation theory')
# plt.show()


# wKSDoS = NiFeTest.GetDensityOfStates(n=0)
# plt.plot(wKS[0:-1], wKSDoS);
# plt.xlabel('Frequency (GHz)');
# plt.ylabel('Density of states');
# plt.show()
