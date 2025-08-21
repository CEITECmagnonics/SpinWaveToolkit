import numpy as np
import SpinWaveToolkit as SWT


class TestClass:
    def test_single_layer(self):
        """Simple dispersion calculation test."""
        show = False  # plot calculated params? (useful for debugging)
        # film and measurement parameters
        mat = SWT.NiFe
        d = 10e-9  # must be thin so that hybridization is weak
        bext = 50e-3
        kxi = np.linspace(0, 25e6, 200)
        kxi[np.argmin(kxi)] += 1e-3
        theta = np.pi/2  # IP magnetization
        phi = np.pi/2  # DE
        bc = 1  # totally unpinned

        sl = SWT.SingleLayer(bext, mat, d, kxi, theta, phi,
                             boundary_cond=bc)
        f_sl = sl.GetDispersion()*1e-9/np.pi/2
        sln = SWT.SingleLayerNumeric(bext, mat, d, kxi, theta, phi,
                                     boundary_cond=bc)
        f_sln = sln.GetDispersion()[0]*1e-9/np.pi/2
        slsc = SWT.SingleLayerSCcoupled(bext, mat, d, kxi, np.inf)
        f_slsc = slsc.GetDispersion()*1e-9/np.pi/2
        zeroth = 0  # select the lowest mode
        if show:
            import matplotlib.pyplot as plt
            plt.plot(kxi, f_sl, label="SL")
            plt.plot(kxi, f_sln[zeroth], label="SLN")
            plt.plot(kxi, f_slsc, label="SLSC")
            plt.title("f(k) (GHz)")
            plt.legend(loc="upper left")
            plt.show()

        assert np.all(np.isclose(f_sl, f_sln[zeroth], f_slsc, atol=4e-3))

        vg_sl = sl.GetGroupVelocity()*1e-3
        vg_sln = sln.GetGroupVelocity(zeroth)*1e-3
        vg_slsc = slsc.GetGroupVelocity()*1e-3
        if show:
            plt.plot(kxi, vg_sl, label="SL")
            plt.plot(kxi, vg_sln, label="SLN")
            plt.plot(kxi, vg_slsc, label="SLSC")
            plt.title("v_gt(k) (um/ns)")
            plt.legend(loc="lower left")
            plt.show()

        assert np.all(np.isclose(vg_sl, vg_sln, vg_slsc, atol=4e-3))

        dec_sl = sl.GetDecLen()*1e6
        dec_sln = sln.GetDecLen(zeroth)*1e6
        dec_slsc = slsc.GetDecLen()*1e6
        if show:
            plt.plot(kxi, dec_sl, label="SL")
            plt.plot(kxi, dec_sln, label="SLN")
            plt.plot(kxi, dec_slsc, label="SLSC")
            plt.title("Lam(k) (um)")
            plt.legend(loc="lower left")
            plt.show()

        assert np.all(np.isclose(dec_sl, dec_sln, dec_slsc, atol=4e-3))

    def test_double_layer(self):
        """Dispersion calculation test comparing the model to
        the model from this paper:
        Gallardo et al. Phys. Rev. Applied 12, 034012 (2019)
        https://doi.org/10.1103/PhysRevApplied.12.034012
        """
        show = False  # plot calculated params? (useful for debugging)

        refdata = np.genfromtxt("tests/testdata/testdata_SAF_Gallardo2019.txt",
                                delimiter=";", dtype=float).T
        nd = 3

        nife = SWT.Material(Ms=658e3,
                            Aex=5.47e-9**2*SWT.MU0*658e3**2/2,  # l_ex to Aex
                            alpha=1e-3,  # some random value
                            gamma=1.7587e11)
        co = SWT.Material(Ms=1150e3,
                          Aex=5.88e-9**2*SWT.MU0*1150e3**2/2,  # l_ex to Aex
                          alpha=1e-3,  # some random value
                          gamma=1.7587e11)
        s = 1e-9
        d = np.array((2, 5, 20))*1e-9
        Ku = 69.6e-3*1150e3/2  # H_u to K_u for Co layer
        J = -1.5e-3  # interlayer coupling (bilinear)
        f_dln = np.zeros((nd, refdata.shape[1]))
        notnan = ~np.isnan(refdata[[0, 2, 4]])
        for i in range(nd):
            dln = SWT.DoubleLayerNumeric(0, co, d[i], refdata[2*i, notnan[i]]*1e6,
                                         theta=np.pi/2, Ku=Ku, Jbl=J, s=s, material2=nife)
            # only frequencies of the first (acoustic) mode
            f_dln[i, notnan[i]] = dln.GetDispersion()[0][0]/2e9/np.pi

        if show:
            import matplotlib.pyplot as plt
            for i in range(nd):
                plt.plot(refdata[2*i], refdata[i*2+1], "--", lw=2,
                         label=f"Gallardo, d={d[i]*1e9}nm")
                plt.plot(refdata[2*i, notnan[i]], f_dln[i, notnan[i]],
                         label=f"SWT, d={d[i]*1e9}nm")
            plt.xlabel("k (rad/µm)")
            plt.ylabel("f(k) (GHz)")
            plt.legend(loc="lower left")
            plt.show()

        for i in range(nd):
            assert np.all(np.isclose(refdata[2*i+1, notnan[i]], f_dln[i, notnan[i]], atol=15e-3))
