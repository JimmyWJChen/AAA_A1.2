import AAA

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
K_h7 = 100 * K_h        # [N/m^7]   Non-linear heave stiffness

# Structural damping components
C_h = K_h / 1000        # [Ns/m]    Structural heave damping
C_α = K_α / 1000        # [Nms/rad] Structural elastic axis damping
C_β = K_β / 1000        # [Nms/rad] Structural hinge damping

# Structural section
structural_section = AAA.Qmatrix.StructuralSection(
        a, b, c, m, S, S_β, I_α, I_αβ, I_β, C_h, C_α, C_β, K_h, K_α, K_β, K_h7)