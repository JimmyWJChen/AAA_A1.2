import numpy as np
from HW_aerodynamics_5580234 import UnsteadyAerodynamics
from HW_structural_analysis_5580234 import Section
import scipy.linalg
import matplotlib.pyplot as plt


class Flutter:
    def __init__(self, Section: Section, rho: float, V_range: list[float, float], CLa: float, psi1: float, psi2: float, epsilon1: float, epsilon2: float, precision: float = 0.1) -> None:
        self.Section = Section
        # obtain the structural mass and stiffness matrices
        self.Ms = self.Section.mass_matrix()
        self.Ks = self.Section.stiffness_matrix()
        self.rho = rho
        self.CLa = CLa
        self.psi1 = psi1
        self.psi2 = psi2
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.V_range = V_range
        self.precision = precision
        self.V = np.linspace(
            self.V_range[0], self.V_range[1], int(self.V_range[1]/self.precision))
        self.eigenvalues = []
        self.eigenvectors = []
        self.V_flutter = 0
        self.u_flutter = np.zeros(3)

    def construct_A_matrix(self, Ma: np.ndarray, Ca: np.ndarray, Ka: np.ndarray, W: np.ndarray, W1: np.ndarray, W2: np.ndarray) -> np.ndarray:
        """
        Assembles the state matrix
        """
        A = np.zeros((12, 12))
        Mae = self.Ms - Ma
        Mae_inv = scipy.linalg.inv(Mae)
        A[0:3, 0:3] = Mae_inv @ Ca
        A[0:3, 3:6] = Mae_inv @ (Ka - self.Ks)
        A[0:3, 6:] = Mae_inv @ W
        A[3:6, 0:3] = np.eye(3)
        A[6:, 3:6] = W1
        A[6:, 6:] = W2
        return A

    def calculate_eigenvalues(self) -> None:
        """
        Calculates the eigenvalues of the state matrix
        """
        for i, Vi in enumerate(self.V):
            # initialise aerodynamic model
            q = 0.5*self.rho*Vi**2
            aerodynamics = UnsteadyAerodynamics(
                self.Section, self.CLa, q, self.rho, self.psi1, self.psi2, self.epsilon1, self.epsilon2)
            # obtain the aerodynamic matrices
            Ma = aerodynamics.calculate_Ma()
            Ca = aerodynamics.calculate_Ca()
            Ka = aerodynamics.calculate_Ka()
            W = aerodynamics.calculate_W()
            W1 = aerodynamics.calculate_W1()
            W2 = aerodynamics.calculate_W2()
            # assemble the state matrix
            A = self.construct_A_matrix(Ma, Ca, Ka, W, W1, W2)
            # find the eigenvalues and eigenvectors
            eigenvalues, eigenvectors = scipy.linalg.eig(A)
            # 12 eigenvalues: 6 are complex conjugate pairs and 6 are real
            # append to eigenvalues and eigenvectors
            self.eigenvalues.append(eigenvalues)
            self.eigenvectors.append(eigenvectors)

    def plot_eigenvalues(self, save: bool = True, plot: bool = False) -> None:
        """
        Plot the eigenvalues for the given velocities
        """
        plt.close('all')
        self.calculate_eigenvalues()
        # there are 12 eigenvalues, but 6 are complex conjugate pairs
        eigenvalues = np.array(self.eigenvalues)
        eigs1 = eigenvalues[:, 0]
        eigs2 = eigenvalues[:, 2]
        eigs3 = eigenvalues[:, 4]
        eigsreal1 = eigenvalues[:, 6]
        eigsreal2 = eigenvalues[:, 7]
        eigsreal3 = eigenvalues[:, 8]
        eigsreal4 = eigenvalues[:, 9]
        eigsreal5 = eigenvalues[:, 10]
        eigsreal6 = eigenvalues[:, 11]
        # to plot the eigenvalues, we have to plot real vs imaginary
        eigs1r = np.real(eigs1)
        eigs1i = np.imag(eigs1)
        eigs2r = np.real(eigs2)
        eigs2i = np.imag(eigs2)
        eigs3r = np.real(eigs3)
        eigs3i = np.imag(eigs3)
        # print(eigs1r)
        # print(np.abs(eigs1r).min())
        # print(self.V[np.abs(eigs1r) == np.abs(eigs1r).min()])
        # print(eigs2r)
        # print(np.abs(eigs2r).min())
        # print(self.V[np.abs(eigs2r) == np.abs(eigs2r).min()])
        # print(eigs3r)
        # print(np.abs(eigs3r).min())
        # print(self.V[np.abs(eigs3r) == np.abs(eigs3r).min()])
        # print(np.where(np.abs(eigs2r) == np.abs(eigs2r).min())[0])
        # print(self.V)
        # only accept V greater than 100
        idxV = int(100/self.precision)
        V_f1 = self.V[idxV:][np.abs(eigs1r[idxV:]) ==
                             np.abs(eigs1r[idxV:]).min()][0]
        V_f2 = self.V[idxV:][np.abs(eigs2r[idxV:]) ==
                             np.abs(eigs2r[idxV:]).min()][0]
        V_f3 = self.V[idxV:][np.abs(eigs3r[idxV:]) ==
                             np.abs(eigs3r[idxV:]).min()][0]
        V = np.array([[V_f1, V_f2, V_f3],
                      [np.abs(eigs1r[idxV:]).min(), np.abs(eigs2r[idxV:]).min(), np.abs(eigs3r[idxV:]).min()]])
        # only take velocities greater than 10
        # idxV = np.where(V[0, :] > 10)[0]
        print(V)
        # print(idxV)
        # if not len(idxV) < 1:
        #     V = V[:, idxV]
        # print(V)
        idx = np.where(V[1, :] == V[1, :].min())[0]
        V_f = V[0, idx][0]
        # update flutter speed
        self.V_flutter = V_f
        # print(V_f)
        # print(V)

        eigsreal1r = np.real(eigsreal1)
        # print(eigsreal1r.shape[0])
        # print(np.zeros(350))
        eigsreal1i = np.zeros_like(eigsreal1r)
        eigsreal2r = np.real(eigsreal2)
        eigsreal2i = np.zeros_like(eigsreal2r)
        eigsreal3r = np.real(eigsreal3)
        eigsreal3i = np.zeros_like(eigsreal3r)
        eigsreal4r = np.real(eigsreal4)
        eigsreal4i = np.zeros_like(eigsreal4r)
        eigsreal5r = np.real(eigsreal5)
        eigsreal5i = np.zeros_like(eigsreal5r)
        eigsreal6r = np.real(eigsreal6)
        eigsreal6i = np.zeros_like(eigsreal6r)

        # make the plots with just the complex eigenvalues
        fig, ax = plt.subplots(dpi=200, figsize=(5, 5))
        ax.plot(eigs1r, eigs1i, label='complex eigenvalue 1',
                marker='.', linestyle='None')
        ax.plot(eigs2r, eigs2i, label='complex eigenvalue 2',
                marker='.', linestyle='None', color='red')
        ax.plot(eigs3r, eigs3i, label='complex eigenvalue 3',
                marker='.', linestyle='None', color='red')
        ax.set_xlabel('Re(λ) [rad/s]')
        ax.set_ylabel('Im(λ) [rad/s]')
        ax.grid()
        ax.legend()
        plt.tight_layout()
        if save:
            plt.savefig(f'flutter_root_loci.pdf', bbox_inches='tight')
        if plot:
            plt.show()

        fig, ax = plt.subplots(dpi=200, figsize=(5, 5))
        ax.plot(self.V, eigs1r, label='complex eigenvalue 1',
                marker='.', linestyle='None')
        ax.plot(self.V, eigs2r, label='complex eigenvalue 2',
                linestyle='None', marker='.', color='red')
        ax.plot(self.V, eigs3r, label='complex eigenvalue 3',
                linestyle='None', marker='.', color='red')
        ax.set_xlabel('V [m/s]')
        ax.set_ylabel('Re(λ) [rad/s]')
        ax.grid()
        ax.legend()
        plt.tight_layout()
        if save:
            plt.savefig(f'flutter_real_vs_V.pdf', bbox_inches='tight')
        if plot:
            plt.show()

        # now do the same with all eigenvalues
        fig, ax = plt.subplots(dpi=200, figsize=(5, 5))
        ax.plot(eigs1r, eigs1i, label='complex eigenvalue 1',
                marker='.', linestyle='None')
        ax.plot(eigs2r, eigs2i, label='complex eigenvalue 2',
                marker='.', linestyle='None', color='red')
        ax.plot(eigs3r, eigs3i, label='complex eigenvalue 3',
                marker='.', linestyle='None', color='red')
        ax.plot(eigsreal1r, eigsreal1i, label='real eigenvalue 1',
                marker='.', linestyle='None')
        ax.plot(eigsreal2r, eigsreal2i, label='real eigenvalue 2',
                marker='.', linestyle='None')
        ax.plot(eigsreal3r, eigsreal3i, label='real eigenvalue 3',
                marker='.', linestyle='None')
        ax.plot(eigsreal4r, eigsreal4i, label='real eigenvalue 4',
                marker='.', linestyle='None')
        ax.plot(eigsreal5r, eigsreal5i, label='real eigenvalue 5',
                marker='.', linestyle='None')
        ax.plot(eigsreal6r, eigsreal6i, label='real eigenvalue 6',
                marker='.', linestyle='None')
        ax.set_xlabel('Re(λ) [rad/s]')
        ax.set_ylabel('Im(λ) [rad/s]')
        ax.grid()
        ax.legend()
        plt.tight_layout()
        if save:
            plt.savefig(f'full_root_loci.pdf', bbox_inches='tight')
        if plot:
            plt.show()

        fig, ax = plt.subplots(dpi=200, figsize=(5, 5))
        ax.plot(self.V, eigs1r, label='complex eigenvalue 1',
                marker='.', linestyle='None')
        ax.plot(self.V, eigs2r, label='complex eigenvalue 2',
                linestyle='None', marker='.', color='red')
        ax.plot(self.V, eigs3r, label='complex eigenvalue 3',
                linestyle='None', marker='.', color='red')
        ax.plot(self.V, eigsreal1r, label='real eigenvalue 1',
                marker='.', linestyle='None')
        ax.plot(self.V, eigsreal2r, label='real eigenvalue 2',
                marker='.', linestyle='None')
        ax.plot(self.V, eigsreal3r, label='real eigenvalue 3',
                marker='.', linestyle='None')
        ax.plot(self.V, eigsreal4r, label='real eigenvalue 4',
                marker='.', linestyle='None')
        ax.plot(self.V, eigsreal5r, label='real eigenvalue 5',
                marker='.', linestyle='None')
        ax.plot(self.V, eigsreal6r, label='real eigenvalue 6',
                marker='.', linestyle='None')
        ax.set_xlabel('V [m/s]')
        ax.set_ylabel('Re(λ) [rad/s]')
        ax.grid()
        ax.legend()
        plt.tight_layout()
        if save:
            plt.savefig(f'full_real_vs_V.pdf', bbox_inches='tight')
        if plot:
            plt.show()
        plt.close('all')

    def get_flutter_mode(self) -> None:
        """
        Get the flutter eigenvector 
        """
        V_f = self.V_flutter
        # construct A matrix
        q = 0.5*self.rho*V_f**2
        aerodynamics = UnsteadyAerodynamics(
            self.Section, self.CLa, q, self.rho, self.psi1, self.psi2, self.epsilon1, self.epsilon2)
        Ma = aerodynamics.calculate_Ma()
        Ca = aerodynamics.calculate_Ca()
        Ka = aerodynamics.calculate_Ka()
        W = aerodynamics.calculate_W()
        W1 = aerodynamics.calculate_W1()
        W2 = aerodynamics.calculate_W2()
        A = self.construct_A_matrix(Ma, Ca, Ka, W, W1, W2)
        eig, eigvector = scipy.linalg.eig(A)
        # identify the correct eigenvalue: Re must be nearly 0
        eig_real = np.real(eig)
        idx = np.where(np.abs(eig_real) == np.abs(eig_real).min())[0]
        # you will get the complex conjugate pair, so we only need one of the two pairs
        # take the pair with positive imaginary part
        # print(eig_real)
        # print(idx)
        flutter_eigenvalue = eig[idx]
        idx2 = np.where(np.imag(flutter_eigenvalue) >= 0)[0]
        # print(idx2)
        flutter_eigenvector = eigvector[:, idx[idx2]][:, 0]
        # print(flutter_eigenvalue)
        # print(flutter_eigenvector)
        # print(flutter_eigenvector.shape)
        # the eigenvector still includes all 12 states, but we only need the 3 DOFs.
        # in the state matrix, the 3 DOFs correspond to states 4 to 6, so slice it from 3 to 6
        flutter_eigenvector = flutter_eigenvector[3:6]
        print(flutter_eigenvector)
        print(np.abs(flutter_eigenvector))
        # print(np.real(flutter_eigenvector))
        # print(np.imag(flutter_eigenvector))
        # eigenvector is complex, so it means the DOFs have a relative phase shift w.r.t. each other
        # take the real part and normalise it to the unit length
        flutter_eigenvector_real = np.real(flutter_eigenvector)
        flutter_eigenvector_real = flutter_eigenvector_real / \
            np.linalg.norm(flutter_eigenvector_real)
        # print(flutter_eigenvector_real)
        # update the flutter eigenvector
        self.u_flutter = flutter_eigenvector_real

    def plot_flutter_mode_shape(self) -> None:
        """
        Plots the flutter mode shape using the typical section mesh
        """
        # get the flutter mode
        self.get_flutter_mode()
        # get the section
        section = self.Section
        u_flutter = self.u_flutter
        print(u_flutter)
        fig, ax = plt.subplots(dpi=200, figsize=(5, 5))
        # update the mesh with the flutter mode shape
        section.update_mesh(*u_flutter)
        ax.plot(section.x, section.z, label='deformed', marker='.')
        ax.plot(section.x0, section.z0, 'k--', label='undeformed',
                marker='.')
        ax.grid()
        ax.legend()
        ax.set_xlabel('x position [m]')
        ax.set_ylabel('z position [m]')
        # invert yaxis because heave positive down while z is positive up
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'flutter_eigenmode.pdf', bbox_inches='tight')
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
    section_input = [m, x_theta, x_beta, a, b,
                     c, r_a2, r_b2, omegah, omegaa, omegab]
    section = Section(*section_input)
    CLa = 2*np.pi
    V_range = [1, 350]
    psi1 = 0.165
    psi2 = 0.355
    epsilon1 = 0.0455
    epsilon2 = 0.3
    flutter = Flutter(section, rho, V_range, CLa,
                      psi1, psi2, epsilon1, epsilon2)
    flutter.plot_eigenvalues(plot=True)

    print(f'flutter speed [m/s] = {flutter.V_flutter}')
    # flutter speed is 300.02886539011143 m/s
    # this is lower than reversal and divergence speeds
    flutter.get_flutter_mode()
    flutter.plot_flutter_mode_shape()
