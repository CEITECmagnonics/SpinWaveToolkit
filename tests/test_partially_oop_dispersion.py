import numpy as np
import SpinWaveToolkit as SWT


class TestClass:
    def test_single_layer_and_maceq(self):
        """SingleLayer static and dispersion calculation test."""
        show = False  # plot calculated params? (useful for debugging)
        # film and measurement parameters
        mat = SWT.NiFe
        d = 20e-9  # must be thin so that hybridization is weak
        bext = 1.5
        kxi = np.linspace(0, 100e6, 601)
        kxi = kxi[1:]  # avoid k=0 (not saved in tetrax testdata)
        phi = np.pi/2  # DE

        nth_s = 91  # number of points in theta static problem
        nth_d = 7  # number of points in theta dispersions
        thi, thf = 0., np.pi/2  # (rad) initial and final value of theta
        thetas = np.linspace(thi, thf, nth_s)  # theta vector
        thetad = np.linspace(thi, thf, nth_d)  # theta vector
        thetad_deg = np.rad2deg(thetad)
        ntx, nswt = 5, 5  # number of modes to calculate with tx and swt

        # STATIC PART
        m_swt = np.zeros((nth_s, 3))  # theta (rad), phi (rad), M/Ms
        heff_swt = np.zeros((nth_s, 3))  # theta (rad), phi (rad), mu0Heff (T)
        for i in range(nth_s):
            maceq = SWT.MacrospinEquilibrium(mat.Ms, bext, thetas[i], phi, verbose=False)
            maceq.minimize(scipy_kwargs={"method": "Powell"})
            m_swt[i] = maceq.M["theta"], maceq.M["phi"], 1
            heff_swt[i] = maceq.getHeff()

        tx_s = np.genfromtxt("tests/testdata/testdata_partial_oop_tetrax_static.txt",
                             delimiter=" ", skip_header=1).T

        if show:
            import matplotlib.pyplot as plt
            plt.subplot(121)
            plt.plot(thetas, m_swt[:, 0], label="theta_M (SWT)")
            plt.plot(thetas, m_swt[:, 1], label="phi_M (SWT)")
            plt.plot(thetas, heff_swt[:, 0], label="theta_Heff (SWT)")
            plt.plot(thetas, heff_swt[:, 1], label="phi_Heff (SWT)")
            plt.plot(tx_s[0], tx_s[1], "--", label="theta_M (Tetrax)")
            plt.plot(tx_s[0], tx_s[2], "--", label="phi_M (Tetrax)")
            plt.plot(tx_s[0], tx_s[4], "--", label="theta_Heff (Tetrax)")
            plt.plot(tx_s[0], tx_s[5], "--", label="phi_Heff (Tetrax)")
            plt.xlabel("theta_Bext (rad)")
            plt.legend(loc="upper left")
            plt.subplot(122)
            plt.plot(thetas, heff_swt[:, 2], label="Heff mag (SWT)")
            plt.plot(tx_s[0], tx_s[6], "--", label="Heff mag (Tetrax)")
            plt.xlabel("theta_Bext (rad)")
            plt.legend(loc="upper left")
            plt.show()

        # check all calculated quantities (omitting theta=0, where phi is arbitrary)
        atol = 1e-5  # prescribed absolute tolerance (~ 10 uT or 10 urad)
        assert np.all(np.isclose(m_swt[1:, 0], tx_s[1, 1:], atol=atol))  # theta_M
        assert np.all(np.isclose(m_swt[1:, 1], tx_s[2, 1:], atol=atol))  # phi_M
        assert np.all(np.isclose(heff_swt[1:, 0], tx_s[4, 1:], atol=atol))  # theta_Heff
        assert np.all(np.isclose(heff_swt[1:, 1], tx_s[5, 1:], atol=atol))  # phi_Heff
        assert np.all(np.isclose(heff_swt[1:, 2], tx_s[6, 1:], atol=atol))  # Heff mag

        # DISPERSION PART
        swt_f = np.zeros((nth_d, nswt, kxi.shape[0]))  # mode frequencies [theta, nmode, k]
        for i in range(nth_d):
            maceq = SWT.MacrospinEquilibrium(mat.Ms, bext, thetad[i], phi, verbose=False)
            maceq.minimize(scipy_kwargs={"method": "Powell"})
            sl = SWT.SingleLayer(material=mat, d=d, kxi=kxi, **maceq.M, **maceq.Bext)
            swt_f[i] = np.array([sl.GetDispersion(j)/(np.pi*2e9) for j in range(nswt)])

        tx_d = np.genfromtxt("tests/testdata/testdata_partial_oop_tetrax_dispersion.txt",
                             delimiter=" ", skip_header=1).T

        if show:
            for i in range(nth_d):
                for j in range(nswt):
                    plt.plot(kxi*1e-6, swt_f[i, j],
                             label=f"SWT, theta={thetad_deg[i]:.1f} deg, mode {j}")
                    plt.plot(tx_d[0]*1e-6, tx_d[1 + j + i*ntx], "--",
                             label=f"Tetrax, theta={thetad_deg[i]:.1f} deg, mode {j}")
            plt.xlabel("k (rad/µm)")
            plt.ylabel("f(k) (GHz)")
            plt.show()

        # prescribed absolute mode tolerances
        atol = [0.3, 0.5, 0.5, 0.6, 0.7]  # in GHz, increasing for higher modes
        # check all calculated frequencies
        for i in range(nth_d):
            for j in range(nswt):
                # print(f"theta={thetad_deg[i]:.1f} deg, mode {j},
                #       swt-tx={np.max(np.abs(swt_f[i, j]-tx_d[1 + j + i*ntx])):.3f} GHz")
                assert np.all(np.isclose(swt_f[i, j], tx_d[1 + j + i*ntx], atol=atol[j])), f"Failed at theta={thetad_deg[i]:.1f} deg, mode {j}"
