# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 15:18:49 2020

@author: ondre
"""

from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt

def transEq(kappa, d, dp):
    e = (kappa**2 - dp**2)*np.tan(kappa*d) - kappa*dp*2 
    return e

dp = 5e6
d = 100e-9
#
#
R = fsolve(transEq, x0=1e7, args = (d, dp))

kappa = np.linspace(-5e6, 7e7, 10000)
y = transEq(kappa, d, dp)

plt.semilogy(kappa*1e-6, abs(y), '.', [1*np.pi/100e-9*1e-6, 1*np.pi/100e-9*1e-6], [-1e99, 1e99], [2*np.pi/100e-9*1e-6, 2*np.pi/100e-9*1e-6], [-1e99, 1e99], [min(kappa)*1e-6, max(kappa)*1e-6], [0, 0], [R*1e-6, R*1e-6], [-1e99, 1e99]);
plt.xlabel('Kappa (rad/um()');
plt.ylim([1e11, 1e16])
plt.legend(['Trans eq', 'n = 1', 'n = 2'])