import AAA
import AAA.Qmatrix
import scipy
import numpy as np


def get_flutter_speed(structural_section: AAA.Qmatrix.StructuralSection, ρ, v_0):
    """
    Finds and returns the flutter speed of a given structural section.

    If Newtons method fail it will try bisection search between 0 and 200 m/s, if that fails it will throw out an error.
    """
    def get_residual(v, return_full_eigenvalue = False):
        aeroelastic_section = AAA.Qmatrix.AeroelasticSection(structural_section, ρ, v)
        Q = AAA.Qmatrix.get_Q_matrix(aeroelastic_section, Jones = False)
        λ, U = scipy.linalg.eig(Q)
    	
        # Filter out all parts with 0 imaginary part
        n = np.count_nonzero(np.imag(λ))
        assert n % 2 == 0, "uneven number of eigenvalues with imaginary component"

        indices = np.argsort(np.imag(λ))
        λ = λ[indices][:n // 2]

        i_λ_flutter = np.argmax(np.real(λ))
        λ_flutter = λ[i_λ_flutter]

        if return_full_eigenvalue:
            U_flutter = U[:, indices][:, :n//2][:3, i_λ_flutter]
            return λ_flutter, U_flutter
        else:
            return np.real(λ_flutter)
    
    v_f = scipy.optimize.newton(get_residual, v_0)
    if v_f < 0:
        print("[WARNING] Newtons method found negative solution, please try another initial guess. Falling back to bisection search...")
        v_f = scipy.optimize.bisect(get_residual, 0, 200)

    λ_f, U_f = get_residual(v_f, return_full_eigenvalue=True)
    ω_f = abs(np.imag(λ_f))
    return v_f, ω_f, U_f
