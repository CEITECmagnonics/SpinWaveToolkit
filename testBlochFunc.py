import SpinWaveToolkit as SWT
import numpy as np
import matplotlib.pyplot as plt

kxi = np.linspace(1e-6, 150e6, 150)

model = SWT.SingleLayer(Bext=20e-3, kxi=kxi, theta=np.pi/2,
                         phi=np.pi/2, d=30e-9, material=SWT.NiFe)

Nf = 100

blochFunc = SWT.BlochFunction.GetBlochFunction(kxi, Nf, model)

