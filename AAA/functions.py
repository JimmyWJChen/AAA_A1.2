import AAA
import AAA.Qmatrix
import scipy
import numpy as np
from time import time


def get_flutter_speed(structural_section: AAA.Qmatrix.StructuralSection, ρ, v_0):
    """
    Finds and returns the flutter speed of a given structural section.

    If Newtons method fail it will try bisection search between 0 and 200 m/s, if that fails it will throw out an error.
    """
    def get_residual(v):
        aeroelastic_section = AAA.Qmatrix.AeroelasticSection(structural_section, ρ, v)
        Q = AAA.Qmatrix.get_Q_matrix(aeroelastic_section, Jones = False)
        λ, U = scipy.linalg.eig(Q)
    	
        # Filter out all parts with 0 imaginary part
        n = np.count_nonzero(np.imag(λ))
        assert n % 2 == 0, "uneven number of eigenvalues with imaginary component"

        indices = np.argsort(np.imag(λ))
        λ = λ[indices][:n // 2]
        return max(np.real(λ))

    starttime = time()
    v_f = scipy.optimize.newton(get_residual, v_0)
    if v_f < 0:
        print("[WARNING] Newtons method found negative solution, please try another initial guess. Falling back to bisection search...")
        v_f = scipy.optimize.bisect(get_residual, 0, 200)
        
    return v_f