# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 09:54:59 2024

@author: ondre
"""




from scipy.optimize import fsolve
def f(w,b=2,x=1,n=0.015,S_0=0.002,Q=21):
    return (w0 - 1/2*(2*epsilon*w**2/c**2 - kx**2)/(k**2 - epsilon*w**2/c**2)*wM)**2 - w**2 - (1/2*kx**2/(k**2 - epsilon*w**2/c**2)*wM)**2
a=fsolve(f,1)
print(a)
print(f(a))

(w0 - 1/2*(2*epsilon*w**2/c**2 - kx**2)/(k**2 - epsilon*w**2/c**2)*wM)**2 - w**2 - (1/2*kx**2/(k**2 - epsilon*w**2/c**2)*wM)**2

