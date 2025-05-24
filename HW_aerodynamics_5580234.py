import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from HW_structural_analysis_5580234 import Section
from numpy import pi
import scipy.linalg


class Aerodynamics:
    def __init__(self, Section: Section, CLa: float, q: float) -> None:
        # section properties
        self.S = Section.S
        self.b = Section.b
        self.c = Section.c
        self.a = Section.a
        # input aerodynamic properties
        self.CLa = CLa
        self.q = q
        # calculate the T coefficients from Theodorsen
        self.T4 = self.calculate_T4()
        self.T5 = self.calculate_T5()
        self.T10 = self.calculate_T10()
        self.T12 = self.calculate_T12()
        # calculate CLb and Cmacb
        self.CLb = self.calculate_CLb()
        self.Cmacb = self.calculate_Cmacb()

    def calculate_T4(self) -> float:
        return -np.arccos(self.c) + self.c*np.sqrt(1 - self.c**2)

    def calculate_T5(self) -> float:
        return -(1 - self.c**2) - (np.arccos(self.c))**2 + 2*self.c*np.sqrt(1 - self.c**2) * np.arccos(self.c)

    def calculate_T10(self) -> float:
        return np.sqrt(1 - self.c**2) + np.arccos(self.c)

    def calculate_T12(self) -> float:
        return np.sqrt(1 - self.c**2) * (2 + self.c) - np.arccos(self.c)*(2*self.c + 1)

    def calculate_CLb(self) -> float:
        return 2*self.T10

    def calculate_Cmacb(self) -> float:
        return -1/2 * (self.T4 + self.T10)

    def calculate_Ka_steady(self) -> np.ndarray:
        Ka = self.q * self.S * np.array([[0, -self.CLa, -self.CLb],
                                         [0, (1/2 + self.a)*self.b*self.CLa, 2*self.b *
                                          self.Cmacb + (1/2 + self.a)*self.b*self.CLb],
                                         [0, -self.b * self.T12, -self.b/np.pi * (self.T5 - self.T4*self.T10) - self.b * self.T12/np.pi * self.T10]])
        return Ka


class UnsteadyAerodynamics(Aerodynamics):
    def __init__(self, Section: Section, CLa: float, q: float, rho: float, psi1: float, psi2: float, epsilon1: float, epsilon2: float) -> None:
        super().__init__(Section, CLa, q)
        self.rho = rho
        self.V = np.sqrt(2*self.q/self.rho)
        # Wagner function coefficients
        self.psi1 = psi1
        self.psi2 = psi2
        # Wagner function exponents
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        # we already have from steady aerodynamics: T4, T5, T10, T12
        # for Ma, we still need: T1, T3, T7 and T13
        self.T1 = self.calculate_T1()
        self.T3 = self.calculate_T3()
        self.T7 = self.calculate_T7()
        self.T13 = self.calculate_T13()
        # for Ca, we still need: T8, T9, T11
        self.T8 = self.calculate_T8()
        self.T9 = self.calculate_T9()
        self.T11 = self.calculate_T11()

    def calculate_T1(self) -> float:
        """
        Calculates T1
        """
        term1 = -1/3*np.sqrt(1 - self.c**2)*(2+self.c**2)
        term2 = self.c * np.arccos(self.c)
        T1 = term1 + term2
        return T1

    def calculate_T3(self) -> float:
        """
        Calculates T3
        """
        term1 = -(1/8 + self.c**2)*(np.arccos(self.c))**2
        term2 = 1/4*self.c*np.sqrt(1 - self.c**2) * \
            np.arccos(self.c)*(7 + 2*self.c**2)
        term3 = -1/8*(1 - self.c**2)*(5*self.c**2 + 4)
        T3 = term1 + term2 + term3
        return T3

    def calculate_T7(self) -> float:
        """
        Calculates T7
        """
        term1 = -(1/8 + self.c**2)*np.arccos(self.c)
        term2 = 1/8*self.c*np.sqrt(1 - self.c**2)*(7 + 2*self.c**2)
        T7 = term1 + term2
        return T7

    def calculate_T8(self) -> float:
        """
        Calculates T8
        """
        term1 = -1/3*np.sqrt(1 - self.c**2)*(2*self.c**2 + 1)
        term2 = self.c * np.arccos(self.c)
        T8 = term1 + term2
        return T8

    def calculate_T9(self) -> float:
        """
        Calculates T9
        """
        p = -1/3*(np.sqrt(1 - self.c**2))**1
        term1 = self.a * self.T4
        T9 = 1/2*(-p + term1)
        return T9

    def calculate_T11(self) -> float:
        """
        Calculates T11 
        """
        term1 = np.arccos(self.c)*(1 - 2*self.c)
        term2 = np.sqrt(1 - self.c**2)*(2 - self.c)
        T11 = term1 + term2
        return T11

    def calculate_T13(self) -> float:
        """
        Calculates T13
        """
        T13 = 1/2*(-self.T7 - (self.c - self.a)*self.T1)
        return T13

    def calculate_Ma(self) -> np.ndarray:
        """
        Calculates the aerodynamic mass matrix
        """
        S = self.S
        b = self.b
        q = self.q
        V = self.V
        a = self.a
        T_1 = self.T1
        c = self.c
        T_7 = self.T7
        T_13 = self.T13
        T_3 = self.T3
        Ma = np.array([[-pi*S*b*q/V**2, pi*S*a*b**2*q/V**2, S*T_1*b**2*q/V**2],
                       [pi*S*a*b**2*q/V**2, -pi*S*b**3*q *
                        (a**2 + 1/8)/V ** 2, -S*b**3*q*(-T_1*(-a + c) - T_7)/V**2],
                       [S*T_1*b**2*q/V**2, -2*S*T_13*b**3*q/V**2, S*T_3*b**3*q/(pi*V**2)]])
        return Ma

    def calculate_Ca(self) -> np.ndarray:
        """
        Calculates the aerodynamic damping matrix
        """
        S = self.S
        q = self.q
        psi_1 = self.psi1
        psi_2 = self.psi2
        V = self.V
        b = self.b
        a = self.a
        T_11 = self.T11
        T_4 = self.T4
        T_1 = self.T1
        c = self.c
        T_8 = self.T8
        T_12 = self.T12
        T_9 = self.T9
        Ca = np.array([[-2*pi*S*q*(-psi_1 - psi_2 + 1)/V, -2*pi*S*b*q*(0.5 - a)*(-psi_1 - psi_2 + 1)/V - pi*S*b*q/V, -S*T_11*b*q*(-psi_1 - psi_2 + 1)/V + S*T_4*b*q/V],
                       [2*pi*S*b*q*(a + 0.5)*(-psi_1 - psi_2 + 1)/V, 2*pi*S*b**2*q*(0.5 - a)*(a + 0.5)*(-psi_1 - psi_2 + 1)/V - pi*S*b**2*q*(0.5 - a)/V,
                        S*T_11*b**2*q*(a + 0.5)*(-psi_1 - psi_2 + 1)/V - S*b**2*q*(T_1 + 0.5*T_11 - T_4*(-a + c) - T_8)/V],
                       [-S*T_12*b*q*(-psi_1 - psi_2 + 1)/V, -S*T_12*b**2*q*(0.5 - a)*(-psi_1 - psi_2 + 1)/V - S*b**2*q*(-T_1 + T_4*(a + -0.5) - 2*T_9)/V, -S*T_11*T_12*b**2*q*(-psi_1 - psi_2 + 1)/(2*pi*V) + S*T_11*T_4*b**2*q/(2*pi*V)]])
        return Ca

    def calculate_Ka(self) -> np.ndarray:
        """
        Calculates the unsteady aerodynamic stiffness matrix
        """
        S = self.S
        q = self.q
        epsilon_1 = self.epsilon1
        psi_1 = self.psi1
        b = self.b
        epsilon_2 = self.epsilon2
        psi_2 = self.psi2
        a = self.a
        T_10 = self.T10
        T_11 = self.T11
        T_4 = self.T4
        T_12 = self.T12
        T_5 = self.T5
        Ka = np.array([[-2*pi*S*q*(epsilon_1*psi_1/b + epsilon_2*psi_2/b), -2*pi*S*q*(epsilon_1*psi_1*(0.5 - a) + epsilon_2*psi_2*(0.5 - a) - psi_1 - psi_2 + 1), -2*pi*S*q*(T_10*(-psi_1 - psi_2 + 1)/pi + T_11*epsilon_1*psi_1/(2*pi) + T_11*epsilon_2*psi_2/(2*pi))],
                       [2*pi*S*b*q*(a + 0.5)*(epsilon_1*psi_1/b + epsilon_2*psi_2/b), 2*pi*S*b*q*(a + 0.5)*(epsilon_1*psi_1*(0.5 - a) + epsilon_2*psi_2*(0.5 - a) - psi_1 -
                                                                                                            psi_2 + 1), -S*b*q*(T_10 + T_4) + 2*pi*S*b*q*(a + 0.5)*(T_10*(-psi_1 - psi_2 + 1)/pi + T_11*epsilon_1*psi_1/(2*pi) + T_11*epsilon_2*psi_2/(2*pi))],
                       [-S*T_12*b*q*(epsilon_1*psi_1/b + epsilon_2*psi_2/b), -S*T_12*b*q*(epsilon_1*psi_1*(0.5 - a) + epsilon_2*psi_2*(0.5 - a) - psi_1 - psi_2 + 1), -S*T_12*b*q*(T_10*(-psi_1 - psi_2 + 1)/pi + T_11*epsilon_1*psi_1/(2*pi) + T_11*epsilon_2*psi_2/(2*pi)) - S*b*q*(-T_10*T_4 + T_5)/pi]])
        return Ka

    def calculate_W(self) -> np.ndarray:
        """
        Calculates the aerodynamic lag matrix 
        """
        S = self.S
        V = self.V
        epsilon_1 = self.epsilon1
        psi_1 = self.psi1
        q = self.q
        b = self.b
        epsilon_2 = self.epsilon2
        psi_2 = self.psi2
        a = self.a
        T_10 = self.T10
        T_11 = self.T11
        T_12 = self.T12
        W = np.array([[2*pi*S*V*epsilon_1**2*psi_1*q/b**2, 2*pi*S*V*epsilon_2**2*psi_2*q/b**2, -2*pi*S*q*(-V*epsilon_1**2*psi_1*(0.5 - a)/b + V*epsilon_1*psi_1/b), -2*pi*S*q*(-V*epsilon_2**2*psi_2*(0.5 - a)/b + V*epsilon_2*psi_2/b), -2*pi*S*q*(T_10*V*epsilon_1*psi_1/(pi*b) - T_11*V*epsilon_1**2*psi_1/(2*pi*b)), -2*pi*S*q*(T_10*V*epsilon_2*psi_2/(pi*b) - T_11*V*epsilon_2**2*psi_2/(2*pi*b))],
                      [-2*pi*S*V*epsilon_1**2*psi_1*q*(a + 0.5)/b, -2*pi*S*V*epsilon_2**2*psi_2*q*(a + 0.5)/b, 2*pi*S*b*q*(a + 0.5)*(-V*epsilon_1**2*psi_1*(0.5 - a)/b + V*epsilon_1*psi_1/b), 2*pi*S*b*q*(a + 0.5)
                       * (-V*epsilon_2**2*psi_2*(0.5 - a)/b + V*epsilon_2*psi_2/b), 2*pi*S*b*q*(a + 0.5)*(T_10*V*epsilon_1*psi_1/(pi*b) - T_11*V*epsilon_1**2*psi_1/(2*pi*b)), 2*pi*S*b*q*(a + 0.5)*(T_10*V*epsilon_2*psi_2/(pi*b) - T_11*V*epsilon_2**2*psi_2/(2*pi*b))],
                      [S*T_12*V*epsilon_1**2*psi_1*q/b, S*T_12*V*epsilon_2**2*psi_2*q/b, -S*T_12*b*q*(-V*epsilon_1**2*psi_1*(0.5 - a)/b + V*epsilon_1*psi_1/b), -S*T_12*b*q*(-V*epsilon_2**2*psi_2*(0.5 - a)/b + V*epsilon_2*psi_2/b), -S*T_12*b*q*(T_10*V*epsilon_1*psi_1/(pi*b) - T_11*V*epsilon_1**2*psi_1/(2*pi*b)), -S*T_12*b*q*(T_10*V*epsilon_2*psi_2/(pi*b) - T_11*V*epsilon_2**2*psi_2/(2*pi*b))]])
        return W

    def calculate_W1(self) -> np.ndarray:
        """
        Calculates the W1 matrix used to couple DOFs to derivatives of lag states
        """
        W1 = np.array([[1, 0, 0],
                       [1, 0, 0],
                       [0, 1, 0],
                       [0, 1, 0],
                       [0, 0, 1],
                       [0, 0, 1]])
        return W1

    def calculate_W2(self) -> np.ndarray:
        """
        Calculates the W2 matrix used to couple lag states to their derivatives 
        """
        V = self.V
        epsilon_1 = self.epsilon1
        b = self.b
        epsilon_2 = self.epsilon2
        W2 = np.array([[-V*epsilon_1/b, 0, 0, 0, 0, 0],
                      [0, -V*epsilon_2/b, 0, 0, 0, 0],
                      [0, 0, -V*epsilon_1/b, 0, 0, 0],
                      [0, 0, 0, -V*epsilon_2/b, 0, 0],
                      [0, 0, 0, 0, -V*epsilon_1/b, 0],
                      [0, 0, 0, 0, 0, -V*epsilon_2/b]])
        return W2


if __name__ == "__main__":
    mu = 40
    omegah = 50
    omegaa = 100
    omegab = 300
    a = -0.4
    b = 1
    c = 0.6
    x_theta = 0.2
    x_beta = -0.025
    r_a2 = 0.25
    r_b2 = 0.00625
    rho = 1.225
    m = mu * np.pi * rho * b**2
    section_input = [m, x_theta, x_beta, a, b,
                     c, r_a2, r_b2, omegah, omegaa, omegab]
    section = Section(*section_input)
    CLa = 2*np.pi
    V = 300
    q = 1/2*rho*V**2
    psi1 = 0.165
    psi2 = 0.355
    epsilon1 = 0.0455
    epsilon2 = 0.3
    aerodynamics = UnsteadyAerodynamics(
        section, CLa, q, rho, psi1, psi2, epsilon1, epsilon2)
    Ms = section.mass_matrix()
    Ks = section.stiffness_matrix()
    Ma = aerodynamics.calculate_Ma()
    Ca = aerodynamics.calculate_Ca()
    Ka = aerodynamics.calculate_Ka()
    W = aerodynamics.calculate_W()
    W1 = aerodynamics.calculate_W1()
    W2 = aerodynamics.calculate_W2()
    A = np.zeros((12, 12))
    Mae = Ms - Ma
    Mae_inv = scipy.linalg.inv(Mae)
    A[0:3, 0:3] = Mae_inv @ Ca
    A[0:3, 3:6] = Mae_inv @ (Ka - Ks)
    A[0:3, 6:] = Mae_inv @ W
    A[3:6, 0:3] = np.eye(3)
    A[6:, 3:6] = W1
    A[6:, 6:] = W2
    print(A)
    eig, v = scipy.linalg.eig(A)
    print(eig)
    assert (eig[0] == np.conjugate(eig[1]))
