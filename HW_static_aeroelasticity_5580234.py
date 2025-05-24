import numpy as np
import scipy as sp
import scipy.linalg
from HW_structural_analysis_5580234 import Section
from HW_aerodynamics_5580234 import Aerodynamics


def calculate_q_divergence(Section: Section, Aerodynamics: Aerodynamics) -> float:
    Ktheta = Section.Ktheta
    S = Section.S
    a = Section.a
    b = Section.b
    CLa = Aerodynamics.CLa
    return Ktheta/(S*CLa*(1/2 + a)*b)


def calculate_q_rev(Section: Section, Aerodynamics: Aerodynamics) -> float:
    CLb = Aerodynamics.CLb
    Ktheta = Section.Ktheta
    CLa = Aerodynamics.CLa
    Cmacb = Aerodynamics.Cmacb
    b = Section.b
    S = Section.S
    return - CLb*Ktheta/(CLa*Cmacb*2*b*S)


def static_aeroelasticity(Section: Section, CLa: float, rho: float) -> tuple[float, float]:
    """
    Returns the divergence speed and the reversal speed
    """
    aerodynamic_model = Aerodynamics(Section, CLa, 1)
    q_div = calculate_q_divergence(Section, aerodynamic_model)
    q_rev = calculate_q_rev(Section, aerodynamic_model)
    V_div = np.sqrt(2*q_div/rho)
    V_rev = np.sqrt(2*q_rev/rho)
    return V_div, V_rev


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
    aerodynamic_model = Aerodynamics(section, CLa, 1)
    print(aerodynamic_model.Cmacb)
    print(section.K)
    print(aerodynamic_model.calculate_Ka_steady())
    Ka = aerodynamic_model.calculate_Ka_steady()
    eigs, eigvectors = scipy.linalg.eig(section.K, Ka)
    print(eigs)
    qd = calculate_q_divergence(section, aerodynamic_model)
    qrev = calculate_q_rev(section, aerodynamic_model)
    Vd = np.sqrt(2*qd/rho)
    Vrev = np.sqrt(2*qrev/rho)
    print(f'divergence speed q_div = {qd}')
    print(f'divergence speed V_div = {Vd}')
    print(f'reversal speed qrev = {qrev}')
    print(f'reversal speed Vrev = {Vrev}')
    # divergence speed is 707.1067811865477 m/s
    # reversal speed is 367.3485251180083 m/s
    V_div, V_rev = static_aeroelasticity(section, CLa, rho)
    print(V_div)
    print(V_rev)
