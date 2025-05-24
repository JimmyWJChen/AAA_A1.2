import numpy as np


class Section:
    def __init__(self, a: float, b: float, c: float, m: float, S: float, S_β: float, I_α: float, I_αβ: float, I_β: float, C_h: float, C_α: float, C_β: float, K_h: float, K_α: float, K_β: float) -> None:
        """
        Structural section, only features structural matrices
        """
        # Geometry terms
        self.a = a          # [-]       Distance center of airfoil to elastic axis
        self.b = b          # [m]       Semichord length
        self.c = c          # [-]       Distance center of airfoil to hinge point
        
        # Inertia terms
        self.m = m          # [kg/m]    Mass per unit length
        self.S = S          # [kgm/m]   Static mass moment of the wing around x_f
        self.S_β = S_β      # [kgm/m]   Static mass moment of the control surface around x_f
        self.I_α = I_α      # [kgm^2/m] Mass moment of the wing around x_f
        self.I_αβ = I_αβ    # [kgm^2/m] Product of inertia of the control surface
        self.I_β = I_β      # [kgm^2/m] Mass moment of the control surface

        # Structural damping components
        self.C_h = C_h      # [Ns/m]    Structural heave damping
        self.C_α = C_α      # [Nms/rad] Structural elastic axis damping
        self.C_β = C_β      # [Nms/rad] Structural hinge damping

        # Structural stiffness components
        self.K_h = K_h      # [N/m]     Linear heave stiffness
        self.K_α = K_α      # [Nm/rad]  Linear pitch stiffness
        self.K_β = K_β      # [Nm/rad]  Linear control stiffness

        self.M_s = self.mass_matrix()
        self.C_s = self.damping_matrix()
        self.K_s = self.stiffness_matrix()

        # Extra components, perhaps for plotting eigenmodes if needed later
        self.x_α = S / (m * b)
        self.x_β = S_β / (m * b)


    def mass_matrix(self) -> np.ndarray:
        """
        Obtains the structural mass matrix
        """
        M = np.array([[self.m, self.S, self.S_β],
                      [self.S, self.I_α, self.I_αβ],
                      [self.S_β, self.I_αβ, self.I_β]])
        return M
    

    def damping_matrix(self) -> np.ndarray:
        """
        Obtains the structural damping matrix
        """
        C = np.array([[self.C_h, 0, 0],
                      [0, self.C_α, 0],
                      [0, 0, self.C_β]])
        return C


    def stiffness_matrix(self) -> np.ndarray:
        """
        Obtains the structural stiffness matrix
        """
        K = np.array([[self.K_h, 0, 0],
                      [0, self.K_α, 0],
                      [0, 0, self.K_β]])
        return K
