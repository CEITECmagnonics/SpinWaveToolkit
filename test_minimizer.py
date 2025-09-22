import numpy as np
import matplotlib.pyplot as plt
import SpinWaveToolkit as swt
from cmcrameri import cm


def main():
    """Testing for the MacrospinEquilibrium."""
    # min_kwargs = {"method": "COBYLA", "options": {"disp": 1, "tol": 1e-12}}
    # min_kwargs = {"method": "SLSQP", "options": {"disp": 1, "tol": 1e-12}}
    min_kwargs = {"method": "Nelder-Mead",}
    maceq = swt.MacrospinEquilibrium(
        Ms=800e3, Bext=150e-3, theta_H=np.deg2rad(90.1), 
        phi_H=np.deg2rad(60), theta=np.deg2rad(30), phi=np.deg2rad(30)
        # demag=np.diag([1., 1., 1.])/3
    )
    maceq.add_uniaxial_anisotropy("uni0", Ku=150e3, theta=90, phi=0)
    maceq.add_uniaxial_anisotropy("uni1", Ku=10e3, 
        theta=np.deg2rad(70), phi=np.pi/2)
    m0 = dict(maceq.M)
    maceq.minimize(scipy_kwargs=min_kwargs)
    # print(maceq.M, maceq.Na_tot, maceq.demag)
    print(maceq.M, maceq.res.x)
    # return
    # print(maceq.anis["uni0"]["Na"], maceq.anis["uni1"]["Na"], sep="\n")

    nth, nph = 200, 100
    th = np.linspace(0, np.pi, nth)
    ph = np.linspace(0, 2*np.pi, nph)
    eden = np.zeros((nth, nph))
    for th_i in range(nth):
        for ph_i in range(nph):
            eden[th_i, ph_i] = maceq.eval_energy([th[th_i], ph[ph_i]])
    
    fig = plt.figure(figsize=(4, 3), constrained_layout=True, dpi=200)
    pc = plt.pcolor(ph, th, eden/1e3, cmap="viridis")
    plt.colorbar(pc, label=r"energy density (kJ/m$^3$)")
    plt.plot([maceq.M["phi"]], [maceq.M["theta"]], "x", c="r", 
             label="minimizer result")
    npmin = np.argmin(eden)
    minx, miny = npmin // nph, npmin % nph
    plt.plot([ph[miny]], [th[minx]], "o", c="c", mfc="none", 
             label="grid minimum")
    plt.plot([m0["phi"]], [m0["theta"]], "^", c="y", mfc="none", 
             label="initial direction")
    plt.ylabel(r"polar angle $\theta$ (rad)")
    plt.xlabel(r"azimuthal angle $\phi$ (rad)")
    plt.legend(loc="upper right")
    plt.show()

def test_hysteresis():
    maceq = swt.MacrospinEquilibrium(
        Ms=800e3, Bext=150e-3, theta_H=np.deg2rad(0), 
        phi_H=np.deg2rad(0), theta=0, phi=0,
    )
    maceq.add_uniaxial_anisotropy("uni0", Ku=50e3, theta=np.pi/2, phi=np.deg2rad(0))
    maceq.add_uniaxial_anisotropy("uni1", Ku=10e3, theta=np.deg2rad(50), phi=np.pi)
    nb, nt = 150, 7
    bexts = np.linspace(-1.5, 1.5, nb)
    bexts = np.concatenate((bexts[nb//2+1:], -bexts, bexts))
    nb = len(bexts)
    bphis = 0+1e-3
    bthetas_deg = np.linspace(0, 90, nt)+1e-3
    bthetas = np.deg2rad(bthetas_deg)
    thetas, phis = np.zeros((nt, nb)), np.zeros((nt, nb))
    for i in range(nt):
        thetas[i], phis[i] = maceq.hysteresis(bexts, bthetas[i], bphis)
    ms = swt.sphr2cart(thetas, phis)

    cmap = cm.lapaz
    fig, ax = plt.subplots(1, 3, figsize=(6, 3), constrained_layout=True, dpi=200)
    for i in range(nt):
        ax[0].plot(bexts, ms[0, i], "-", c=cmap(i/nt), label=f"{bthetas_deg[i]:.0f} °")
        ax[1].plot(bexts, ms[1, i], "-", c=cmap(i/nt))
        ax[2].plot(bexts, ms[2, i], "-", c=cmap(i/nt))
    ax[0].set_xlabel("mag. field (T)")
    ax[1].set_xlabel("mag. field (T)")
    ax[2].set_xlabel("mag. field (T)")
    ax[0].set_ylabel(r"$m_x$ ()")
    ax[1].set_ylabel(r"$m_y$ ()")
    ax[2].set_ylabel(r"$m_z$ ()")
    fig.legend(loc="outside right upper", title=r"$\theta_B$ (OOP angle)")
    plt.show()


def test_hysteresis_box():
    maceq = swt.MacrospinEquilibrium(
        Ms=800e3, Bext=150e-3, theta_H=0, phi_H=0, # demag=np.diag([1, 1, 1])/3
    )
    maceq.add_uniaxial_anisotropy("ua0", 0, 90+1e-3, 0, 15e-3)
    nb, na = 50, 5  # number of fields and angles
    bexts = np.linspace(-0.05, 0.05, nb)
    bexts = np.concatenate((bexts[nb//2+1:], -bexts, bexts))
    nb = len(bexts)

    # bphis = (0+1e-3)*np.ones(na)
    # bthetas_deg = np.linspace(0, 90, na)+1e-3
    # bthetas = np.deg2rad(bthetas_deg)

    bthetas = np.deg2rad(90)*np.ones(na)+1e-3
    bphis_deg = np.linspace(0, 90, na)+1e-3
    bphis = np.deg2rad(bphis_deg)
    
    thetas, phis = np.zeros((na, nb)), np.zeros((na, nb))
    for i in range(na):
        thetas[i], phis[i] = maceq.hysteresis(bexts, bthetas[i], bphis[i],
                                              scipy_kwargs={"method": "Nelder-Mead"})
    ms = swt.sphr2cart(thetas, phis)

    cmap = cm.lapaz
    fig, ax = plt.subplots(1, 3, figsize=(6, 3), constrained_layout=True, dpi=200)
    for i in range(na):
        # ax[0].plot(bexts, ms[0, i], "-", c=cmap(i/na), label=f"{bthetas_deg[i]:.0f} °")
        ax[0].plot(bexts, ms[0, i], "-", c=cmap(i/na), label=f"{bphis_deg[i]:.0f} °")
        ax[1].plot(bexts, ms[1, i], "-", c=cmap(i/na))
        ax[2].plot(bexts, ms[2, i], "-", c=cmap(i/na))
    ax[0].set_xlabel("mag. field (T)")
    ax[1].set_xlabel("mag. field (T)")
    ax[2].set_xlabel("mag. field (T)")
    ax[0].set_ylabel(r"$m_x$ ()")
    ax[1].set_ylabel(r"$m_y$ ()")
    ax[2].set_ylabel(r"$m_z$ ()")
    # fig.legend(loc="outside right upper", title=r"$\theta_B$ (OOP angle)")
    fig.legend(loc="outside right upper", title=r"$\phi_B$ (IP angle)")
    plt.show()


if __name__ == "__main__":
    # main()
    # test_hysteresis()
    test_hysteresis_box()
