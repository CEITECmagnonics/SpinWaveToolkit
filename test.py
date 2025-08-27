# import modules
import numpy as np  # for vectorization
import matplotlib.pyplot as plt  # for plotting
import SpinWaveToolkit as SWT

# Define material properties using custom parameters
mat = SWT.Material(
    Ms=800e3,   # (A/m) saturation magnetization
    Aex=16e-12,  # (J/m) exchange stiffness
    alpha=0.007,  # () damping
    gamma=28.8*1e9*2*np.pi  # (rad*Hz/T) gyromagnetic ratio
)

# or use built-in materials (NiFe, YIG, CoFeB, FeNi)
mat = SWT.NiFe

# Define the propagation geometry
d = 10e-9  # (m) layer thickness
Bext = 1.1  # (T) external magnetic field
theta = np.pi/2*0.6  # (rad) for in-plane magnetization
phi = np.pi/2  # (rad) for Damon-Eshbach geometry
k = np.linspace(1e4, 25e6, 200)  # (rad/m) SW wavenumber range 
# Note the +1 above used to avoid badly conditioned calculations at k=0
bc = 1 # boundary condition (1 - totally unpinned, 2 - totally pinned spins)

# Instantiate the SingleLayer class
sl = SWT.SingleLayer(Bext-mat.Ms*SWT.MU0, mat, d, k, theta, phi, boundary_cond=bc)
# Instantiate the SingleLayerNumeric class with the same parameters as above
sln = SWT.SingleLayerNumeric(Bext-mat.Ms*SWT.MU0, mat, d, k, theta, phi)

f_de_num = sln.GetDispersion()[0]/(2e9*np.pi)  # (GHz) dispersion for 3 lowest modes

nmodes = 3  # number of PSSW modes to calculate
fig = plt.figure(figsize=(4, 2.5))
for i in range(nmodes):
    f_de_pssw = sl.GetDispersion(i)/(2e9*np.pi)  # (GHz) DE SW dispersion
    plt.plot(k*1e-6, f_de_pssw, '--', label=f"KS $n=${i}")
    plt.plot(k*1e-6, f_de_num[i], label=f"Num $n=${i}")
plt.xlabel(r"wavenumber $k$ (rad/µm)")
plt.ylabel(r"frequency $f$ (GHz)")
plt.legend(loc="lower right")
