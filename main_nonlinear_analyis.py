import numpy as np
import scipy
import matplotlib.pyplot as plt

import AAA
import AAA.Qmatrix
import AAA.functions
import AAA.nonlinearode
import AAA.plotting


if __name__ == "__main__":
    # Interesting points near flutter velocity
    near_flutter_analysis = False

    # Interesting velocities after flutter velocity
    post_flutter_analysis = False

    # Verify equivalent linearisation result
    equivalent_linearisation_verification = True

    # Geometry terms
    # [-]       Distance center of airfoil to elastic axis
    a = -0.2
    b = 0.5                 # [m]       Semichord length
    # [-]       Distance center of airfoil to hinge point
    c = 0.5

    # Inertia terms
    m = 50                  # [kg/m]    Mass per unit length
    # [kgm/m]   Static mass moment of the wing around x_f
    S = 5
    # [kgm/m]   Static mass moment of the control surface around x_f
    S_β = 1.56
    I_α = 4.67              # [kgm^2/m] Mass moment of the wing around x_f
    # [kgm^2/m] Product of inertia of the control surface
    I_αβ = 0.81
    I_β = 0.26              # [kgm^2/m] Mass moment of the control surface

    # Structural stiffness components
    K_h = 25e3              # [N/m]     Linear heave stiffness
    K_α = 9e3               # [Nm/rad]  Linear pitch stiffness
    K_β = 1e3               # [Nm/rad]  Linear control stiffness
    K_h7 = 100 * K_h        # [N/m^7]   Non-linear heave stiffness

    # Structural damping components
    C_h = K_h / 1000        # [Ns/m]    Structural heave damping
    C_α = K_α / 1000        # [Nms/rad] Structural elastic axis damping
    C_β = K_β / 1000        # [Nms/rad] Structural hinge damping

    # Nonlinear heave stiffness
    Kh7 = 100 * K_h

    # Flight conditions
    ρ = 1.225
    v_0 = 60
    tmax = 4
    n_points = 10000

    structural_section = AAA.Qmatrix.StructuralSection(
        a, b, c, m, S, S_β, I_α, I_αβ, I_β, C_h, C_α, C_β, K_h, K_α, K_β, Kh7)
    
    # Get exact flutter speed
    v_f, ω_f, U_f = AAA.functions.get_flutter_speed(structural_section, ρ, v_0)
    print(f"Found flutter speed of {v_f:#.04g} [m/s] with frequency {ω_f:#.04g} [rad/s]")
    δh = 0.15
    
    if near_flutter_analysis:
        # Solve system right below flutter, 
        aeroelastic_section = AAA.Qmatrix.AeroelasticSection(structural_section, ρ, v_f - 1)
        aeroelastic_section.set_up_nonlinear_part()
        solution = AAA.nonlinearode.solve(aeroelastic_section, [δh, 0, 0], tmax)
        t = np.linspace(0, tmax, n_points)
        AAA.plotting.plot_nonlinear_solution(t, solution, "output/nonlinear/pertubation_analysis/below_flutter_speed.pdf" )

        # Solve system at flutter speed, remaining relatively constant
        aeroelastic_section = AAA.Qmatrix.AeroelasticSection(structural_section, ρ, v_f)
        aeroelastic_section.set_up_nonlinear_part()
        solution = AAA.nonlinearode.solve(aeroelastic_section, [δh, 0, 0], tmax)
        t = np.linspace(0, tmax, n_points)
        AAA.plotting.plot_nonlinear_solution(t, solution, "output/nonlinear/pertubation_analysis/at_flutter_speed.pdf")

        # Solve system above flutter speed, transitioning into LCO
        tmax = 10
        tmax_first_plot = 5
        aeroelastic_section = AAA.Qmatrix.AeroelasticSection(structural_section, ρ, v_f + 1)
        aeroelastic_section.set_up_nonlinear_part()
        solution = AAA.nonlinearode.solve(aeroelastic_section, [δh, 0, 0], tmax)
        t = np.linspace(0, tmax_first_plot, n_points)
        AAA.plotting.plot_nonlinear_solution(t, solution, "output/nonlinear/pertubation_analysis/above_flutter_speed.pdf")

        t = np.linspace(tmax - 1, tmax, n_points)
        AAA.plotting.plot_nonlinear_solution(t, solution, "output/nonlinear/pertubation_analysis/LCO_slightly_above_flutter_speed.pdf")

        # Solve system at flutter speed, remaining relatively constant
        tmax = 4
        aeroelastic_section = AAA.Qmatrix.AeroelasticSection(structural_section, ρ, v_f)
        aeroelastic_section.set_up_nonlinear_part()
        solution = AAA.nonlinearode.solve(aeroelastic_section, [0.6, 0, 0], tmax)
        t = np.linspace(0, tmax, n_points)
        AAA.plotting.plot_nonlinear_solution(t, solution, "output/nonlinear/pertubation_analysis/at_flutter_speed_highh0.pdf")

    if post_flutter_analysis:
        # v = 70
        tmax = 10
        aeroelastic_section = AAA.Qmatrix.AeroelasticSection(structural_section, ρ, 70)
        aeroelastic_section.set_up_nonlinear_part()
        solution = AAA.nonlinearode.solve(aeroelastic_section, [0.6, 0, 0], tmax)
        t = np.linspace(9, tmax, n_points)
        AAA.plotting.plot_nonlinear_solution(t, solution, "output/nonlinear/pertubation_analysis/70.pdf", αβmultiplier = 1)

        # v = 80
        tmax = 10
        aeroelastic_section = AAA.Qmatrix.AeroelasticSection(structural_section, ρ, 80)
        aeroelastic_section.set_up_nonlinear_part()
        solution = AAA.nonlinearode.solve(aeroelastic_section, [0.6, 0, 0], tmax)
        t = np.linspace(9, tmax, n_points)
        AAA.plotting.plot_nonlinear_solution(t, solution, "output/nonlinear/pertubation_analysis/80.pdf", αβmultiplier = 0.05)

        # v = 90
        tmax = 10
        aeroelastic_section = AAA.Qmatrix.AeroelasticSection(structural_section, ρ, 90)
        aeroelastic_section.set_up_nonlinear_part()
        solution = AAA.nonlinearode.solve(aeroelastic_section, [0.6, 0, 0], tmax)
        t = np.linspace(9, tmax, n_points)
        AAA.plotting.plot_nonlinear_solution(t, solution, "output/nonlinear/pertubation_analysis/90.pdf", αβmultiplier = 0.05)

        # v = 100
        tmax = 10
        aeroelastic_section = AAA.Qmatrix.AeroelasticSection(structural_section, ρ, 100)
        aeroelastic_section.set_up_nonlinear_part()
        solution = AAA.nonlinearode.solve(aeroelastic_section, [0.6, 0, 0], tmax)
        t = np.linspace(9, tmax, n_points)
        AAA.plotting.plot_nonlinear_solution(t, solution, "output/nonlinear/pertubation_analysis/100.pdf", αβmultiplier = 0.05)

        # v = 110
        tmax = 10
        aeroelastic_section = AAA.Qmatrix.AeroelasticSection(structural_section, ρ, 110)
        aeroelastic_section.set_up_nonlinear_part()
        solution = AAA.nonlinearode.solve(aeroelastic_section, [0.6, 0, 0], tmax)
        t = np.linspace(9, tmax, n_points)
        AAA.plotting.plot_nonlinear_solution(t, solution, "output/nonlinear/pertubation_analysis/110.pdf", αβmultiplier = 0.05)

        # v = 115
        tmax = 10
        aeroelastic_section = AAA.Qmatrix.AeroelasticSection(structural_section, ρ, 115)
        aeroelastic_section.set_up_nonlinear_part()
        solution = AAA.nonlinearode.solve(aeroelastic_section, [0.6, 0, 0], tmax)
        t = np.linspace(9, tmax, n_points)
        AAA.plotting.plot_nonlinear_solution(t, solution, "output/nonlinear/pertubation_analysis/115.pdf", αβmultiplier = 0.05)

        # v = 120
        tmax = 10
        aeroelastic_section = AAA.Qmatrix.AeroelasticSection(structural_section, ρ, 120)
        aeroelastic_section.set_up_nonlinear_part()
        solution = AAA.nonlinearode.solve(aeroelastic_section, [0.6, 0, 0], tmax)
        t = np.linspace(9, tmax, n_points)
        AAA.plotting.plot_nonlinear_solution(t, solution, "output/nonlinear/pertubation_analysis/120.pdf", αβmultiplier = 0.05)

        # v = 124
        tmax = 10
        aeroelastic_section = AAA.Qmatrix.AeroelasticSection(structural_section, ρ, 124)
        aeroelastic_section.set_up_nonlinear_part()
        solution = AAA.nonlinearode.solve(aeroelastic_section, [0.6, 0, 0], tmax)
        t = np.linspace(9, tmax, n_points)
        AAA.plotting.plot_nonlinear_solution(t, solution, "output/nonlinear/pertubation_analysis/124.pdf", αβmultiplier = 0.05)

        # v = 125
        tmax = 10
        aeroelastic_section = AAA.Qmatrix.AeroelasticSection(structural_section, ρ, 125)
        aeroelastic_section.set_up_nonlinear_part()
        solution = AAA.nonlinearode.solve(aeroelastic_section, [0.6, 0, 0], tmax)
        t = np.linspace(9, tmax, n_points)
        AAA.plotting.plot_nonlinear_solution(t, solution, "output/nonlinear/pertubation_analysis/125.pdf", αβmultiplier = 0.05)

    
    if equivalent_linearisation_verification:
        # Get nonlinear solution
        vs = np.linspace(49, 124.8, 100)
        As, ωs = AAA.nonlinearode.velocity_sweep(structural_section, ρ, 0.8, vs)
        AAA.plotting.plot_nonlinear_velocity_sweep(vs, As, ωs, "output/nonlinear/exact_time_propagation.pdf")

        # Get equivalent linearisation solution
        AESectionNL = AAA.Qmatrix.AeroelasticSection(structural_section, ρ, v_0)
        Qeqlin = AAA.Qmatrix.get_Q_matrix(AESectionNL, False)

        NLInput = AAA.Qmatrix.StructuralSectionInput(
            a, b, c, m, S, S_β, I_α, I_αβ, I_β, C_h, C_α, C_β, K_h, K_α, K_β, K_h7)

        Ah = np.linspace(0, 2, 1001)
        v_fs, ω_fs = AAA.functions.get_LCO_flutter_speeds(NLInput, ρ, v_0, Ah)

        A_list, v_list, ω_list = AAA.functions.get_LCO_branches(Ah, v_fs, ω_fs)

        dA = 1e-10
        path_A_v = f'output/nonlinear/verification_amplitude_vs_velocity.pdf'
        path_ω_v = f'output/nonlinear/verification_frequency_vs_velocity.pdf'
        AAA.plotting.bifurcation_plots_eq_lin(
        Ah, v_fs, ω_fs, NLInput, ρ, dA, path_A_v, path_ω_v, vs_exact = vs, As_exact = As, ωs_exact = ωs)