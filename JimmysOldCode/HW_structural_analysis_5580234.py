import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


class Section:
    def __init__(self, m: float, x_theta: float, x_beta: float, a: float, b: float, c: float, r2_a: float, r2_b: float, omegah: float, omegaa: float, omegab: float) -> None:
        self.m = m
        self.x_theta = x_theta
        self.x_beta = x_beta
        self.a = a
        self.b = b
        self.c = c
        self.S = 2*self.b*1
        self.r2a = r2_a
        self.r2b = r2_b
        self.omegah = omegah
        self.omegaa = omegaa
        self.omegab = omegab
        self.M = self.mass_matrix()
        self.K = self.stiffness_matrix()
        self.Kh, self.Ktheta, self.Kbeta = self.K.diagonal()
        # define mesh
        # self.x0 = np.linspace(-self.b, self.b, 20)
        # self.z0 = np.zeros_like(self.x0)
        # self.x = np.zeros_like(self.x0)
        # self.z = np.zeros_like(self.x0)
        self.x_ea = self.a*self.b
        self.x_hinge = self.c*self.b
        # airofoil consists of 2 elements:
        # element 1: from LE to  hinge, element 2: from hinge to TE
        self.x0 = np.array([-self.b, self.x_hinge, self.b])
        self.z0 = np.zeros_like(self.x0)
        self.x = np.copy(self.x0)
        self.z = np.zeros_like(self.x0)
        # local coordinates of element 1: origin located at x_hinge
        self.offset = -self.x_ea
        self.offset_coords = np.array(
            [-self.b + self.offset, 0, self.x_hinge + self.offset, 0, self.b + self.offset, 0])
        self.local_coords_1 = np.array(
            [-self.b + self.offset, 0, self.x_hinge + self.offset, 0])
        self.offset_2 = - self.x_hinge
        self.local_coords_2 = np.array(
            [self.x_hinge + self.offset_2, 0, self.b + self.offset_2, 0])

    def mass_matrix(self) -> np.ndarray:
        M = self.m * self.b**2 * np.array([[1/self.b**2, self.x_theta/self.b, self.x_beta/self.b],
                                           [self.x_theta/self.b, self.r2a,
                                               self.r2b + self.x_beta*(self.c - self.a)],
                                           [self.x_beta/self.b, self.r2b + self.x_beta*(self.c - self.a), self.r2b]])
        return M

    def uncoupled_mass_matrix(self) -> np.ndarray:
        M = self.m * self.b**2 * np.array([[1/self.b**2, 0, 0],
                                           [0, self.r2a, 0],
                                           [0, 0, self.r2b]])
        return M

    def stiffness_matrix(self) -> np.ndarray:
        K = self.m * self.b**2 * np.array([[self.omegah**2/self.b**2, 0, 0],
                                           [0, self.r2a * self.omegaa**2, 0],
                                           [0, 0, self.r2b * self.omegab**2]])
        return K

    def update_mesh(self, h: float, theta: float, beta: float) -> None:
        # update z
        # perform translation
        self.z = self.z0 + h
        ct = np.cos(theta)
        st = np.sin(theta)
        Rt = np.array([[ct, -st, 0, 0, 0, 0],
                       [st, ct, 0, 0, 0, 0],
                       [0, 0, ct, -st, 0, 0],
                       [0, 0, st, ct, 0, 0],
                       [0, 0, 0, 0, ct, -st],
                       [0, 0, 0, 0, st, ct]])
        cb = np.cos(beta)
        sb = np.sin(beta)
        Rb = np.array([[cb, -sb, 0, 0],
                       [sb, cb, 0, 0],
                       [0, 0, cb, -sb],
                       [0, 0, sb, cb]])
        # perform rotations
        rotated_coords_theta = Rt @ self.offset_coords
        dr_theta = rotated_coords_theta - self.offset_coords
        rotated_coords_beta = Rb @ self.local_coords_2
        dr_beta = rotated_coords_beta - self.local_coords_2
        dx = dr_theta[0::2]
        dx[1:3] += dr_beta[0::2]
        dz = dr_theta[1::2]
        dz[1:3] += dr_beta[1::2]
        # apply changes to x and z
        self.x = self.x0 + dx
        self.z = self.z + dz
        # self.z[self.x0 <= self.x_ea] = self.z0[self.x0 <= self.x_ea] + \
        #     (self.x0[self.x0 <= self.x_ea] - self.x_ea) * -np.sin(theta)
        # self.z[self.x0 > self.x_ea] = self.z0[self.x0 > self.x_ea] + \
        #     (self.x0[self.x0 > self.x_ea] - self.x_ea) * np.sin(theta)
        # self.z[self.x0 >= self.x_hinge] = self.z0[self.x0 >= self.x_hinge] + \
        #     (self.x0[self.x0 >= self.x_hinge] - self.x_hinge) * np.sin(beta)
        # # update x
        # dx1 = (self.x0[self.x0 <= self.x_ea] - self.x_ea) * \
        #     np.cos(theta) - self.x0[self.x0 <= self.x_ea]
        # dx2 = (self.x0[(
        #     self.x0 > self.x_ea) & (self.x0 < self.x_hinge)] - self.x_ea) * -np.cos(theta) - self.x0[(
        #         self.x0 > self.x_ea) & (self.x0 < self.x_hinge)]

        # self.x[self.x0 <= self.x_ea] = (
        #     self.x0[self.x0 <= self.x_ea] - self.x_ea) * np.cos(theta)
        # self.x[(self.x0 > self.x_ea) & (self.x0 < self.x_hinge)] = (self.x0[(
        #     self.x0 > self.x_ea) & (self.x0 < self.x_hinge)] - self.x_ea) * -np.cos(theta)
        # self.x[self.x0 >= self.x_hinge] = (self.x0[self.x0 >= self.x_hinge] -
        #                                    (self.x_hinge + (self.x_hinge - self.x_ea) * -np.cos(theta))) * -np.cos(beta+theta)

    def plot_eigenmodes(self, v: np.ndarray, coupled: str) -> None:
        v_h = v[:, 0]
        v_theta = v[:, 1]
        v_beta = v[:, 2]
        # normalise
        v_h = v_h/np.linalg.norm(v_h)
        v_theta = v_theta/np.linalg.norm(v_theta)
        v_beta = v_beta/np.linalg.norm(v_beta)
        fig, (ax1, ax2, ax3) = plt.subplots(
            1, 3, dpi=200, figsize=(18, 5), sharey=True)
        self.update_mesh(*v_h)
        ax1.plot(self.x, self.z, label=f'deformed', marker='.')
        ax1.plot(self.x0, self.z0, 'k--', label=f'undeformed', marker='.')
        ax1.grid()
        ax1.legend(bbox_to_anchor=(1., 1), loc='upper right')
        ax1.set_title(f'first eigenmode')
        ax1.set_xlabel(f'x position [m]')
        ax1.set_ylabel(f'z position [m]')
        ax1.invert_yaxis()
        # ax1.set_ylim(self.x0.min(), self.x0.max())
        # ax1.set_aspect('equal')
        self.update_mesh(*v_theta)
        ax2.plot(self.x, self.z, label=f'deformed', marker='.')
        ax2.plot(self.x0, self.z0, 'k--', label=f'undeformed', marker='.')
        ax2.grid()
        ax2.legend(bbox_to_anchor=(1., 1), loc='upper right')
        ax2.set_title(f'second eigenmode')
        ax2.set_xlabel(f'x position [m]')
        # ax2.set_ylabel(f'z position [m]')
        ax2.invert_yaxis()
        # ax2.set_aspect('equal')
        # print(
        #     f'length is {np.sqrt((self.x[-1] - self.x[0])**2 + (self.z[-1] - self.z[0])**2)}')
        self.update_mesh(*v_beta)
        ax3.plot(self.x, self.z, label=f'deformed', marker='.')
        ax3.plot(self.x0, self.z0, 'k--', label=f'undeformed', marker='.')
        ax3.grid()
        ax3.legend(bbox_to_anchor=(1., 1), loc='upper right')
        ax3.set_title(f'third eigenmode')
        ax3.set_xlabel(f'x position [m]')
        # ax3.set_ylabel(f'z position [m]')
        ax3.invert_yaxis()
        # ax3.set_aspect('equal')
        plt.tight_layout()
        plt.savefig(f'{coupled}_eigenmodes.pdf', bbox_inches='tight')
        plt.show()


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
    section = Section(m, x_theta, x_beta, a, b, c, r_a2,
                      r_b2, omegah, omegaa, omegab)
    print(f'{section.Kh}, {section.Ktheta}, {section.Kbeta}')
    M = section.mass_matrix()
    m_tot = M[0, 0]
    S_theta = M[0, 1]
    S_beta = M[0, 2]
    I_theta = M[1, 1]
    I_beta = M[2, 2]
    m_f = S_beta/(x_beta * b)
    m_a = m_tot - m_f
    I_beta_star = I_beta - m_f * x_beta**2 * b**2
    I_theta_star = I_theta - m_a * x_theta**2 * b**2 - \
        m_f * (c - a + x_beta)**2 * b**2 - I_beta_star
    print(f'm = {M[0, 0]}')
    print(f'S_θ = {M[0, 1]}')
    print(f'S_β = {M[0, 2]}')
    print(f'I_θ = {M[1, 1]}')
    print(f'I_β = {M[2, 2]}')
    print(f'm_f = {m_f}')
    print(f'm_a = {m_a}')
    print(f'I_β* = {I_beta_star}')
    print(f'I_θ* = {I_theta_star}')
    K = section.stiffness_matrix()
    # solve generalised eigen problem
    w, v = sp.linalg.eigh(K, M)
    omega = np.sqrt(w)
    print(f'coupled eigenfrequencies')
    print(rf'$ω_h$ ', omega[0])
    print(rf'$ω_θ$ = ', omega[1])
    print(rf'$ω_β$ = ', omega[2])
    print(f'v_h = \n{v[:, 0]}')
    print(f'v_θ = \n {v[:, 1]}')
    print(f'v_β = \n {v[:, 2]}')
    print(f'{v}')
    M_uncoupled = section.uncoupled_mass_matrix()
    w_uncoupled, v_uncoupled = sp.linalg.eigh(K, M_uncoupled)
    omega_uncoupled = np.sqrt(w_uncoupled)
    print(f'uncoupled eigenfrequencies: ')
    print(
        f'ω h = {omega_uncoupled[0]} \nω θ = {omega_uncoupled[1]} \nω β = {omega_uncoupled[2]}')
    print(f'v_h = \n{v_uncoupled[:, 0]}')
    print(f'v_θ = \n {v_uncoupled[:, 1]}')
    print(f'v_β = \n {v_uncoupled[:, 2]}')
    section.plot_eigenmodes(v_uncoupled, 'uncoupled')
    section.plot_eigenmodes(v, 'coupled')
