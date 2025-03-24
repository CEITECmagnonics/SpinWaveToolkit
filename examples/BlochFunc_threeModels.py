import SpinWaveToolkit as SWT
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

kxi = np.linspace(1e-6, 150e6, 1500)

model = SWT.SingleLayer(Bext=20e-3, kxi=kxi, theta=np.pi/2,
                         phi=np.pi/2, d=200e-9, material=SWT.NiFe)

Nf = 1000

w0, bf0 = model.GetBlochFunction(n=0, Nf=Nf)
w1, bf1 = model.GetBlochFunction(n=1, Nf=Nf)
w2, bf2 = model.GetBlochFunction(n=2, Nf=Nf)

# Define a common frequency grid
w_common = np.linspace(np.min([w0.min(), w1.min(), w2.min()]), np.max([w0.max(), w1.max(), w2.max()]), Nf)

# Interpolate bf0, bf1, bf2 along the first dimension (frequency axis)
bf0_interp = interp1d(w0, bf0, axis=0, kind='linear', bounds_error=False, fill_value=0)(w_common)
bf1_interp = interp1d(w1, bf1, axis=0, kind='linear', bounds_error=False, fill_value=0)(w_common)
bf2_interp = interp1d(w2, bf2, axis=0, kind='linear', bounds_error=False, fill_value=0)(w_common)

# Sum all interpolated Bloch functions
bf_sum = bf0_interp + bf1_interp + bf2_interp

plt.figure()
plt.contourf(kxi/1e6, w_common/2/np.pi/1e9, bf_sum, 100, cmap='viridis')
plt.xlabel('kx (rad/µm)')
plt.ylabel('f (GHz)')
plt.colorbar()
plt.title('Bloch function - analytic')
plt.show()

modelNumeric = SWT.SingleLayerNumeric(Bext=20e-3, kxi=kxi, theta=np.pi/2,
                         phi=np.pi/2, d=200e-9, material=SWT.NiFe)

Nf = 1000

# Interpolate all Bloch functions to the same frequency grid
w0, bf0 = modelNumeric.GetBlochFunction(n=0, Nf=Nf)
w1, bf1 = modelNumeric.GetBlochFunction(n=1, Nf=Nf)
w2, bf2 = modelNumeric.GetBlochFunction(n=2, Nf=Nf)

# Define a common frequency grid
w_common = np.linspace(np.min([w0.min(), w1.min(), w2.min()]), np.max([w0.max(), w1.max(), w2.max()]), Nf)

# Interpolate bf0, bf1, bf2 along the first dimension (frequency axis)
bf0_interp = interp1d(w0, bf0, axis=0, kind='linear', bounds_error=False, fill_value=0)(w_common)
bf1_interp = interp1d(w1, bf1, axis=0, kind='linear', bounds_error=False, fill_value=0)(w_common)
bf2_interp = interp1d(w2, bf2, axis=0, kind='linear', bounds_error=False, fill_value=0)(w_common)

# Sum all interpolated Bloch functions
bf_sum = bf0_interp + bf1_interp + bf2_interp

plt.figure()
plt.contourf(kxi/1e6, w_common/2/np.pi/1e9, bf_sum, 100, cmap='viridis')
plt.xlabel('kx (rad/µm)')
plt.ylabel('f (GHz)')
plt.title('Bloch function - numeric')
plt.colorbar()
plt.show()


kxi = np.linspace(-20e6+1e-6, 20e6, 1500)
modelDouble = SWT.DoubleLayerNumeric(Bext=50e-3, kxi=kxi, theta=np.pi/2,
                         phi=np.pi/2, d=20e-9, material=SWT.CoFeB, s=0.6e-9, Jbl=1e-3)

Nf = 1000

# Interpolate all Bloch functions to the same frequency grid
w0, bf0 = modelDouble.GetBlochFunction(n=0, Nf=Nf, lifeTime=3e-9)
w1, bf1 = modelDouble.GetBlochFunction(n=1, Nf=Nf, lifeTime=3e-9)

# Define a common frequency grid
w_common = np.linspace(np.min([w0.min(), w1.min()]), np.max([w0.max(), w1.max()]), Nf)

# Interpolate bf0, bf1, bf2 along the first dimension (frequency axis)
bf0_interp = interp1d(w0, bf0, axis=0, kind='linear', bounds_error=False, fill_value=0)(w_common)
bf1_interp = interp1d(w1, bf1, axis=0, kind='linear', bounds_error=False, fill_value=0)(w_common)

# Sum all interpolated Bloch functions
bf_sum = bf0_interp + bf1_interp

plt.figure()
plt.contourf(kxi/1e6, w_common/2/np.pi/1e9, bf_sum, 100, cmap='viridis')
plt.xlabel('kx (rad/µm)')
plt.ylabel('f (GHz)')
plt.title('Bloch function - numeric double layer')
plt.colorbar()
plt.show()