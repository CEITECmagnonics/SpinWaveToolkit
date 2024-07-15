import numpy as np
import SpinWaveToolkit as SWT


class TestClass:
    def test_single_layer(self):
        """Simple dispersion calculation test."""
        show = False  # plot calculated params? (useful for debugging)
        # film and measurement parameters
        mat = SWT.NiFe
        d = 10e-9
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
        zeroth = 3
        if show:
            import matplotlib.pyplot as plt
            plt.plot(kxi, f_sl, label="SL")
            plt.plot(kxi, f_sln[zeroth], label="SLN")
            plt.title("f(k) (GHz)")
            plt.legend(loc="upper left")
            plt.show()

        assert np.all(np.isclose(f_sl, f_sln[zeroth], atol=4e-3))

        vg_sl = sl.GetGroupVelocity()*1e-3
        vg_sln = sln.GetGroupVelocity(zeroth)*1e-3
        if show:
            plt.plot(kxi, vg_sl, label="SL")
            plt.plot(kxi, vg_sln, label="SLN")
            plt.title("v_gt(k) (um/ns)")
            plt.legend(loc="lower left")
            plt.show()

        assert np.all(np.isclose(vg_sl, vg_sln, atol=4e-3))

        dec_sl = sl.GetDecLen()*1e6
        dec_sln = sln.GetDecLen(zeroth)*1e6
        if show:
            plt.plot(kxi, dec_sl, label="SL")
            plt.plot(kxi, dec_sln, label="SLN")
            plt.title("Lam(k) (um)")
            plt.legend(loc="lower left")
            plt.show()

        assert np.all(np.isclose(dec_sl, dec_sln, atol=4e-3))
