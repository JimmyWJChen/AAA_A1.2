import scipy
import numpy as np
import scipy.integrate
import AAA
import AAA.Qmatrix
import matplotlib.pyplot as plt

def solve(nonlinear_aeroelastic_section, x_0, tmax):
    Q = nonlinear_aeroelastic_section.Q
    q_n = nonlinear_aeroelastic_section.q_n
    Kh7 = nonlinear_aeroelastic_section.Kh7

    y_0 = np.zeros_like(q_n)
    y_0[0:3] = x_0
    
    def y_dot(t, y):
        y = Q @ y + q_n * Kh7 * y[0]**7
        return y
    
    result = scipy.integrate.solve_ivp(y_dot, [0, tmax], y_0, method="Radau", dense_output=True)
    return result
    

    
