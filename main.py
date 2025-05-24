import AAA
import AAA.Qmatrix
import AAA.plotting
import matplotlib.pyplot as plt

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

    # Flight conditions
    ρ = 1.225
    v = 67 
    v_max = 80 # For linear plots

    structural_section = AAA.Qmatrix.StructuralSection(a, b, c, m, S, S_β, I_α, I_αβ, I_β, C_h, C_α, C_β, K_h, K_α, K_β)
    aeroelastic_section = AAA.Qmatrix.AeroelasticSection(structural_section, ρ, v)
    Q = AAA.Qmatrix.get_Q_matrix(aeroelastic_section, Jones=False)

    print(structural_section.M_s)
    print(structural_section.K_s)
    print(structural_section.C_s)

    AAA.plotting.plot_uncoupled_structural_eigenmodes(structural_section, "output/linear/uncoupled_undamped_eigenmodes.pdf", heavemultiplier=0.5, ylim = [-0.7, 0.6])
    AAA.plotting.plot_coupled_structural_eigenmodes(structural_section, "output/linear/coupled_undamped_eigenmodes.pdf", heavemultiplier=0.5, ylim = [-0.7, 0.6])
    AAA.plotting.linear_flutter_diagrams(structural_section, v_max, ρ, "output/linear/velocity_against_eigenvalues.pdf", "output/linear/real_against_imaginary.pdf")