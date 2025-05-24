import numpy as np
from numpy import arccos as acos, sqrt # Shortens theodorsen constants significantly
import scipy


class StructuralSection:
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
    

    def compute_coupled_undamped_eigenmodes(self):
        """
        Computes coupled eigenfrequencies and eigenmodes of the structural system, without damping
        """
        self.ωs, self.U = scipy.linalg.eigh(a = self.K_s, b = self.M_s) # symmetric matrices, eigh is faster and sorted than eig.
        self.ωs = np.sqrt(self.ωs) # Solving for ω^2, units [rad/s]
        return self.ωs, self.U
    

    def compute_uncoupled_undamped_eigenmodes(self):
        """
        Computes uncoupled eigenfrequencies and eigenmodes of the structural system, without damping
        """
        # Take only the diagonal of the mass matrix (damping and stiffness is already diagonal)
        M_s_uncoupled = np.diag(np.diag(self.M_s))
        self.ωs, self.U = scipy.linalg.eigh(a = self.K_s, b = M_s_uncoupled) # symmetric matrices, eigh is faster and sorted than eig.
        self.ωs = np.sqrt(self.ωs) # Solving for ω^2, units [rad/s]
        return self.ωs, self.U


class AeroelasticSection():
    def __init__(self, structural_section: StructuralSection, ρ: float, v: float) -> None:
        """ 
        structural_section: Structural Section class, this way you only need to define this once
        ρ: air density [kg/m^3]
        v: velocity [m/s]
        
        Generates both stuctural matrices as well as the aerodynamic matrices
        """
        self.structural_section = structural_section
        self.ρ = ρ
        self.v = v
        
        # Copying some things over for ease of use
        self.M_s = structural_section.M_s
        self.C_s = structural_section.C_s
        self.K_s = structural_section.K_s
        self.a = structural_section.a
        self.b = structural_section.b
        self.c = structural_section.c
        c = self.c
        
        self.q = 0.5 * ρ * v**2
        
        # Theodorsen constants
        self.T_1 = c*acos(c) - sqrt(1 - c**2)*(c**2 + 2)/3
        self.T_2 = c*(1 - c**2) + c*acos(c)**2 - sqrt(1 - c**2)*(c**2 + 1)*acos(c)
        self.T_3 = c*sqrt(1 - c**2)*(2*c**2 + 7)*acos(c)/4 - (1 / 8 - c**2 / 8)*(5*c**2 + 4) + (-c**2 + -1 / 8)*acos(c)**2
        self.T_4 = - acos(c) + c*sqrt(1 - c**2) 
        self.T_5 = -(1 - c**2) - (acos(c))**2 + 2 * c * sqrt(1-c**2) * acos(c)
        self.T_6 = c*(1 - c**2) + c*acos(c)**2 - sqrt(1 - c**2)*(c**2 + 1)*acos(c)
        self.T_7 = c*sqrt(1 - c**2)*(2*c**2 + 7)/8 + (-c**2 + -1 / 8)*acos(c)
        self.T_8 = -1/3 * sqrt(1-c**2) * (2*c**2 + 1) + c * acos(c)
        self.T_9 = 0.5 * (1/3 * sqrt(1-c**2) + self.a * self.T_4)
        self.T_10 = sqrt(1 - c**2) + acos(c)
        self.T_11 = acos(c) * (1 - 2*c) + sqrt(1-c**2) * (2-c)
        self.T_12 = sqrt(1 - c**2)*(c + 2) - (2*c + 1)*acos(c)
        self.T_13 = -c*sqrt(1 - c**2)*(2*c**2 + 7)/16 - (-self.a + c)*(c*acos(c) - sqrt(1 - c**2)*(c**2 + 2)/3)/2 - (-c**2 + -1 / 8)*acos(c)/2
        self.T_14 = self.a*c/2 + 1 / 16

        # Circulatory matrices at C(k) = 1
        self.C_a_c, self.K_a_c  = self.circulatory_matrices(1)

        # Noncirculatory matrices
        self.M_a_nc, self.C_a_nc, self.K_a_nc = self.non_circulatory_matrices()

    
    def circulatory_matrices(self, Ck: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates the noncirculatory aerodynamic matrices according to appendix A
        """
        b = self.b
        a = self.a

        K_a = 2*self.q*self.b * Ck * np.array([
            [0, -2*np.pi, -2*self.T_10],
            [0, 2*np.pi*b*(a + (1 / 2)), 2*self.T_10*b*(a + (1 / 2))],
            [0, -self.T_12*b, -self.T_10*self.T_12*b/np.pi]])

        C_a = self.ρ * self.v * b * Ck * np.array([
            [-2*np.pi, -2*np.pi*b*((1 / 2) - a), -self.T_11*b],
            [2*np.pi*b*(a + (1 / 2)), 2*np.pi*b**2*((1 / 2) - a)*(a + (1 / 2)), self.T_11*b**2*(a + (1 / 2))], 
            [-self.T_12*b, -self.T_12*b**2*((1 / 2) - a), -self.T_11*self.T_12*b**2/(2*np.pi)]])

        return C_a, K_a 
    

    def non_circulatory_matrices(self):
        """
        Generates the noncirculatory aerodynamic matrices according to appendix A
        """
        M_a_nc = - self.ρ * self.b**2 * np.array([
            [np.pi, -np.pi*self.a*self.b, -self.T_1*self.b], 
            [-np.pi*self.a*self.b, np.pi*self.b**2*(self.a**2 + (1 / 8)), -self.b**2*(self.T_1*(-self.a + self.c) + self.T_7)], 
            [-self.T_1*self.b, 2*self.T_13*self.b**2, -self.T_3*self.b**2/np.pi]])
        
        C_a_nc = -self.ρ * self.v * self.b**2 * np.array([
            [0, np.pi, -self.T_4],
            [0, np.pi*self.b*((1 / 2) - self.a), self.b*(self.T_1 + self.T_11/2 - self.T_4*(-self.a + self.c) - self.T_8)], 
            [0, self.b*(-self.T_1 + self.T_4*(self.a + (-1 / 2)) - 2*self.T_9), -self.T_11*self.T_4*self.b/(2*np.pi)]])

        K_a_nc = -self.q*self.b**2 * np.array([
            [0, 0, 0], 
            [0, 0, 2*self.T_10 + 2*self.T_4],
            [0, 0, 2*(-self.T_10*self.T_4 + self.T_5)/np.pi]])
        
        return M_a_nc, C_a_nc, K_a_nc


    def set_up_statespace_nterm(self, a_s: list, p_s: list) -> None:
        """
        a_s, p_s from approximation of Ck = 1 + a_1 k / (k + p_1 i) + a_2 k / (k + p_2 i)+ ... + a_n k / (k + p_n i)
        Generates state space according to the methods described in appendix A
        """
        # Set up A_0, A_1, A_2
        p_s = np.array(p_s)
        b_s = p_s * -1 * self.v/self.b
        A_0 = (self.K_a_nc + self.K_a_c)
        A_1 = (self.C_a_nc + self.C_a_c * (1 + sum(a_s)))
        A_2 = (self.M_a_nc) 

        # Set up extra lag term aerodynamic matrices
        A_extras = []
        for i in range(len(a_s)):
            A_i = a_s[i] * self.K_a_c - a_s[i] * b_s[i] * self.C_a_c
            A_extras.append(A_i)
        
        # Define the state space
        size = 6 + 3*len(a_s)
        submatrices = size // 3
        aeromatrices = len(a_s)
        Q = np.zeros((size, size))

         # Set up identities along vertical
        for i in range(submatrices):
            Q[3*i:3*(i+1), 3:6] = np.identity(3)

        # B terms
        for i in range(aeromatrices):
            Q[3*(i+2):3*(i+3), 3*(i+2):3*(i+3)] = - b_s[i] * np.identity(3)
        
        # Finally set up S matrices
        inv = np.linalg.inv(self.M_s - A_2)
        S_0 = inv @ (A_0 - self.K_s)
        S_1 = inv @ (A_1 - self.C_s)

        S_array = [S_0, S_1]
        for i in range(aeromatrices):
            S_array.append(inv @ A_extras[i])

        for i, S in enumerate(S_array):
            Q[3:6, 3*i:3*(i+1)] = S

        return Q


def get_Q_matrix(aeroelastic_section: AeroelasticSection, Jones=False):
    # more accurate 3 term approximation
    if not Jones:
        return aeroelastic_section.set_up_statespace_nterm([-0.26202386, -0.05434653, -0.18300204], [-0.12080652, -0.01731469, -0.46477241])
    # Jones approximation (equivalent to Wagner)
    else:
        return aeroelastic_section.set_up_statespace_nterm([-0.165, -0.335], [-0.0455, -0.3])