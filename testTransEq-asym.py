# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 17:56:30 2021

@author: ondre
"""
import numpy as np
from scipy.optimize import fsolve

dp1 = 1.5e7
dp2 = 1.5e7
d = 100e-9

def transEq(kappa, d, dp1, dp2):
    e = (kappa**2 - dp1*dp2)*np.tan(kappa*d) - kappa*(dp1 + dp2) 
    return e
#The classical thickness mode is given as starting point
#kappa = fsolve(transEq, x0=(3*np.pi/d), args = (d, dp1, dp2), maxfev=10000, epsfcn=1e-10, factor=0.1)
    
kappa = np.linspace(0, 20e7, 500)

eq = transEq(kappa, d, dp1, dp2)