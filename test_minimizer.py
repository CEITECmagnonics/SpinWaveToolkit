import numpy as np
import matplotlib.pyplot as plt
import SpinWaveToolkit as swt


def main():
    """Testing for the MacrospinEquilibrium."""
    # min_kwargs = {"method": "COBYLA", "options": {"disp": 1, "tol": 1e-12}}
    # min_kwargs = {"method": "SLSQP", "options": {"disp": 1, "tol": 1e-12}}
    min_kwargs = {}
    maceq = swt.MacrospinEquilibrium(
        Ms=800e3, Bext=150e-3, theta_H=np.deg2rad(10), 
        phi_H=np.deg2rad(60), theta=0, phi=12
    )
    maceq.add_uniaxial_anisotropy("uni0", Ku=-150e3, theta=0, phi=np.deg2rad(10))
    maceq.add_uniaxial_anisotropy("uni1", Ku=-100e3, theta=np.deg2rad(80), phi=np.pi)
    maceq.minimize(scipy_kwargs=min_kwargs)
    print(maceq.M)

    # print(maceq.anis["uni0"]["Na"], maceq.anis["uni1"]["Na"], sep="\n")

    nth, nph = 200, 100
    th = np.linspace(0, np.pi, nth)
    ph = np.linspace(0, 2*np.pi, nph)
    th_mesh, ph_mesh = np.meshgrid(th, ph, indexing="ij")
    m = swt.sphr2cart(th_mesh, ph_mesh).transpose((1, 2, 0))
    eden = np.zeros((nth, nph))
    for th_i in range(nth):
        for ph_i in range(nph):
            eden[th_i, ph_i] = maceq.eval_energy(m[th_i, ph_i])
    
    fig = plt.figure(figsize=(4, 3), constrained_layout=True, dpi=200)
    pc = plt.pcolor(ph, th, eden/1e3, cmap="viridis")
    plt.colorbar(pc, label=r"energy density (kJ/m$^3$)")
    plt.plot([maceq.M["phi"]], [maceq.M["theta"]], "x", c="r", 
             label="minimizer result")
    npmin = np.argmin(eden)
    minx, miny = npmin // nph, npmin % nph
    plt.plot([ph[miny]], [th[minx]], "o", c="c", mfc="none", 
             label="grid minimum")
    plt.ylabel(r"polar angle $\theta$ (rad)")
    plt.xlabel(r"azimuthal angle $\phi$ (rad)")
    plt.legend(loc="upper right")
    plt.show()


if __name__ == "__main__":
    main()
