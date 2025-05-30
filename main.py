import AAA
import matplotlib.pyplot as plt
import scipy
import numpy as np
from time import time


"""
Code primarily to generate equivalent linearisation result plots in section 2
"""
if __name__ == "__main__":
    # Setup structural parameters
    from setup_structural_section import *
    
    # Flight conditions
    ρ = 1.225
    v_0 = 60  # Initial guess for linear flutter speed
    v_max = 120  # For linear plots

    starttime = time()
    v_f, ω_f, U_f = AAA.functions.get_flutter_speed(structural_section, ρ, v_0)

    print(f"Found flutter speed of {v_f:#.04g} [m/s] with frequency {ω_f:#.04g} [rad/s] in {time() - starttime:#.04g} [s]")
    

    A_h = 0.1
    SectionNL = AAA.Qmatrix.StructuralSection(
        a, b, c, m, S, S_β, I_α, I_αβ, I_β, C_h, C_α, C_β, K_h, K_α, K_β, K_h7, A_h, True)

    AESectionNL = AAA.Qmatrix.AeroelasticSection(SectionNL, ρ, v_0)
    Qeqlin = AAA.Qmatrix.get_Q_matrix(AESectionNL, False)

    NLInput = AAA.Qmatrix.StructuralSectionInput(
        a, b, c, m, S, S_β, I_α, I_αβ, I_β, C_h, C_α, C_β, K_h, K_α, K_β, K_h7)

    Ah = np.linspace(0, 2, 1001)
    v_fs, ω_fs = AAA.functions.get_LCO_flutter_speeds(
        NLInput, ρ, v_0, Ah)
    
    λs, vs_scatter, λ_plunge, λ_torsion, λ_flap = AAA.functions.get_eigenvalues(structural_section, ρ, v_fs)

    dA = 1e-10
    path_A_v = f'output/nonlinear/amplitude_vs_velocity.pdf'
    path_ω_v = f'output/nonlinear/frequency_vs_velocity.pdf'
    # AAA.plotting.bifurcation_plots_eq_lin(
    #     Ah, v_fs, ω_fs, NLInput, ρ, dA, path_A_v, path_ω_v, ω_flap = np.imag(λ_flap))
    AAA.plotting.bifurcation_plots_eq_lin(
        Ah, v_fs, ω_fs, NLInput, ρ, dA, path_A_v, path_ω_v, ω_flap = [])