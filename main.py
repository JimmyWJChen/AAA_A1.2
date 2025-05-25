import AAA
import AAA.Qmatrix
import AAA.functions
import AAA.nonlinearode
import AAA.plotting
import matplotlib.pyplot as plt
import scipy
import numpy as np
from time import time

if __name__ == "__main__":
    # Geometry terms
    a = -0.2                # [-]       Distance center of airfoil to elastic axis
    b = 0.5                 # [m]       Semichord length
    c = 0.5                 # [-]       Distance center of airfoil to hinge point
    
    # Inertia terms
    m = 50                  # [kg/m]    Mass per unit length
    S = 5                   # [kgm/m]   Static mass moment of the wing around x_f
    S_β = 1.56              # [kgm/m]   Static mass moment of the control surface around x_f
    I_α = 4.67              # [kgm^2/m] Mass moment of the wing around x_f
    I_αβ = 0.81             # [kgm^2/m] Product of inertia of the control surface
    I_β = 0.26              # [kgm^2/m] Mass moment of the control surface

    # Structural stiffness components
    K_h = 25e3              # [N/m]     Linear heave stiffness
    K_α = 9e3               # [Nm/rad]  Linear pitch stiffness
    K_β = 1e3               # [Nm/rad]  Linear control stiffness

    # Structural damping components
    C_h = K_h / 1000        # [Ns/m]    Structural heave damping
    C_α = K_α / 1000        # [Nms/rad] Structural elastic axis damping
    C_β = K_β / 1000        # [Nms/rad] Structural hinge damping

    # Nonlinear heave stiffness
    Kh7 = 100 * K_h

    # Flight conditions
    ρ = 1.225
    v_0 = 60 # Initial guess for linear flutter speed
    v_max = 70 # For linear plots

    structural_section = AAA.Qmatrix.StructuralSection(a, b, c, m, S, S_β, I_α, I_αβ, I_β, C_h, C_α, C_β, K_h, K_α, K_β)

    AAA.plotting.plot_uncoupled_structural_eigenmodes(structural_section, "output/linear/uncoupled_undamped_eigenmodes.pdf", heavemultiplier=0.5, ylim = [-0.7, 0.6])
    AAA.plotting.plot_coupled_structural_eigenmodes(structural_section, "output/linear/coupled_undamped_eigenmodes.pdf", heavemultiplier=0.5, ylim = [-0.7, 0.6])
    AAA.plotting.linear_flutter_diagrams(structural_section, v_max, ρ, "output/linear/velocity_against_eigenvalues.pdf", "output/linear/real_against_imaginary.pdf")

    starttime = time()
    v_f, ω_f, U_f = AAA.functions.get_flutter_speed(structural_section, ρ, v_0)
    print(f"Found flutter speed of {v_f:#.04g} [m/s] with frequency {ω_f:#.04g} [rad/s] in {time() - starttime:#.04g} [s]")

    vs = np.linspace(60, 125, 100)
    AAA.nonlinearode.velocity_sweep(structural_section, ρ, Kh7, 0.1, vs)
    
    # v = 124 m / s compared to 123 m / s is very interesting
    # nonlinear_aeroelastic_section = AAA.Qmatrix.NonlinearAeroelasticSection(structural_section, ρ, 59.7, Kh7)
    # result = AAA.nonlinearode.solve(nonlinear_aeroelastic_section, [0.1, 0, 0], 20)
    # t = np.linspace(10, 20, 10000)
    # y = result.sol(t)

    # h = y[0, :]
    # h_dot = y[3, :]

    # plt.figure(figsize=(8, 8))
    # plt.plot(h, h_dot)

    # plt.show()
    # Us = np.array([y[0, :], y[1, :], y[2, :]]).T
    # print(Us.shape)
    
    # plt.plot(t, y[0, :])
    # plt.xlabel("t [s]")
    # plt.ylabel("h [m]")
    # plt.grid()
    # plt.show()
    
    # Showing flutter mode and saving it as a video
    # T_flutter = 2*np.pi / ω_f
    # timepoints = np.linspace(0, 4*T_flutter, 240)
    # Us_flutter = []
    # ts_flutter = []
    # for t in timepoints:
    #     # Method from Moti Karpel guest lecture from fundamentals of aeroelasticity
    #     U_t = np.real(U_f * np.exp(1j * ω_f * t))
    #     Us_flutter.append(U_t)
    #     ts_flutter.append(t)
    
    # AAA.plotting.animate_DOFs(Us, t, structural_section, "output/nonlinear/fluttermode.mp4", multiplier=1)
