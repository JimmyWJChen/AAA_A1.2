import AAA
import numpy as np
import matplotlib.pyplot as plt

"""
Code primarily for reproducing plots in section 2
"""
if __name__ == "__main__":
    from setup_structural_section import *

    # Flight conditions
    ρ = 1.225
    v_0 = 60  # Initial guess for linear flutter speed
    v_max = 120  # For linear plots

    # Eigenmodes, and flutter diagram
    AAA.plotting.plot_uncoupled_structural_eigenmodes(
        structural_section, "output/linear/uncoupled_undamped_eigenmodes.pdf", multiplier=0.5, ylim=[-0.7, 0.6])
    AAA.plotting.plot_coupled_structural_eigenmodes(
        structural_section, "output/linear/coupled_undamped_eigenmodes.pdf", multiplier=0.5, ylim=[-0.7, 0.6])
    AAA.plotting.linear_flutter_diagrams(
        structural_section, v_max, ρ, "output/linear/velocity_against_eigenvalues.pdf", "output/linear/real_against_imaginary.pdf")

    v_f, ω_f, U_f = AAA.functions.get_flutter_speed(structural_section, ρ, v_0)

    # Showing flutter mode and saving it as a static plot at several timepoints
    T_flutter = 2*np.pi / ω_f
    timepoints = np.linspace(0, T_flutter, 9)
    Us_flutter = []
    ts_flutter = []
    plt.figure(figsize=(12, 4))
    for i, t in enumerate(timepoints):
        # Method from Moti Karpel guest lecture from fundamentals of aeroelasticity
        U_t = np.real(U_f * np.exp(1j * ω_f * t))
        plt.subplot(int(f"19{i + 1}"))
        if i == 0:
            noyticklabels = False
        else:
            noyticklabels = True
        AAA.plotting.plot_eigenmodes_subplot(
            structural_section, U_t, fr"t = {t / T_flutter:#.02g}$T_f$", norm = False, 
            noyticklabels=noyticklabels, ylim=[-1.5, 1], loc="upper right", multiplier = 25, legend=False)
    plt.subplots_adjust(wspace=0)
    plt.savefig("output/linear/fluttermode.pdf", bbox_inches = "tight")

    # Showing flutter mode and saving it as a video
    T_flutter = 2*np.pi / ω_f
    timepoints = np.linspace(0, 4 * T_flutter, 240)
    Us_flutter = []
    ts_flutter = []
    for t in timepoints:
        # Method from Moti Karpel guest lecture from fundamentals of aeroelasticity
        U_t = np.real(U_f * np.exp(1j * ω_f * t))
        Us_flutter.append(U_t)
        ts_flutter.append(t)
    AAA.plotting.animate_DOFs(Us_flutter, t, structural_section,
                              "output/linear/fluttermode.mp4", multiplier=25)