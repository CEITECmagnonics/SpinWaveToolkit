import SpinWaveToolkit as SWT
import numpy as np
import matplotlib.pyplot as plt

kxi = np.linspace(1e-6, 150e6, 1500)

model = SWT.SingleLayer(Bext=20e-3, kxi=kxi, theta=np.pi/2,
                         phi=np.pi/2, d=30e-9, material=SWT.NiFe)

Nf = 1000

w, bf = model.GetBlochFunction(n=0, Nf=Nf)

plt.figure()
plt.contourf(kxi/1e6, w[:,0]/2/np.pi/1e9, bf, cmap='viridis')
plt.xlabel('kx (rad/µm)')
plt.ylabel('f (GHz)')
plt.colorbar()
plt.show()