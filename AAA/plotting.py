import matplotlib.pyplot as plt
import numpy as np
import scipy
import AAA
import AAA.Qmatrix

def linear_flutter_diagrams(structural_section: AAA.Qmatrix.StructuralSection, max_v: float, ρ: float, name_velocity_eigenvalues: str, name_eigenvalue_eigenvalue: str) -> None:
    """
    Creates the following 2 plots using the linear state space A matrix:

    - Velocity against eigenvalue imaginary and real parts
    - Eigenvalue real vs eigenvalue imaginary part

    max_v is the maximum velocity used to compute the eigenvalues
    ρ is the flight condition
    the names are the output plot names
    """
    # Set up plot
    plt.figure(figsize=(12, 4))

    # Set up velocities
    vs = np.linspace(0, max_v, max_v)

    # Collecting data for plots
    # It is significantly faster to scatter all at the same time then adding a single scatter point in the loop.
    vs_scatter = []
    λs = []
    λ_plunge = []
    λ_torsion = []
    λ_flap = []

    for v in vs:
        # Set up A matrix
        section = AAA.Qmatrix.AeroelasticSection(structural_section, ρ, v)
        Q = section.set_up_statespace_nterm([-0.26202386, -0.05434653, -0.18300204], [-0.12080652, -0.01731469, -0.46477241]) # Q 8

        # Compute eigenvalues
        λ, _ = scipy.linalg.eig(Q) # Non symmetric matrices
        λ = np.array(λ)
        
        # Track branches to sort the flap, torsion and plunge modes
        if v == vs[0]:
            # Sort them based on imaginary component, or the structural natural frequency with some extra air damping
            indices = np.argsort(np.imag(λ))[-3:]
            index_flap = indices[-1]
            index_torsion = indices[-2]
            index_plunge = indices[-3]
        else:
            # Find closest to previous eigenvalue to follow the branch
            difference_flap = np.abs(λ - λ_flap[-1])
            index_flap = np.argmin(difference_flap)
            difference_torsion = np.abs(λ - λ_torsion[-1])
            index_torsion = np.argmin(difference_torsion)
            difference_plunge = np.abs(λ - λ_plunge[-1])
            index_plunge = np.argmin(difference_plunge)

        # Add them to the arrays
        λ_flap.append(λ[index_flap])
        λ_torsion.append(λ[index_torsion])
        λ_plunge.append(λ[index_plunge])

        # For scattering all datapoints
        λs.append(λ)
        vs_scatter.extend(len(λ) * [v])

    # Create velocity against real eigenvalue parts subplot
    plt.subplot(121)
    plt.scatter(vs_scatter, np.real(λs), alpha=0.2, color="grey", s = 2)
    plt.plot(vs, np.real(λ_plunge), color="red")
    plt.plot(vs, np.real(λ_torsion), color="blue")
    plt.plot(vs, np.real(λ_flap), color="green")
    plt.grid()
    plt.minorticks_on()
    plt.xlabel("V [m/s]")
    plt.ylabel("Re(λ) [rad/s]")
    plt.ylim([-50, 10])

    # Create velocity against imaginary eigenvalue parts subplot
    plt.subplot(122)
    plt.scatter(vs_scatter, np.imag(λs), alpha=0.2, color="grey", s = 2)
    plt.plot(vs, np.imag(λ_plunge), color="red", label= "plunge mode")
    plt.plot(vs, np.imag(λ_torsion), color="blue", label= "torsion mode")
    plt.plot(vs, np.imag(λ_flap), color="green", label= "flap mode")
    plt.grid()
    plt.minorticks_on()
    plt.xlabel("V [m/s]")
    plt.ylabel("Im(λ) [rad/s]")
    plt.legend()
    plt.tight_layout()

    # Save plot
    plt.savefig(name_velocity_eigenvalues, bbox_inches="tight")
    plt.close("all")

    # Create imaginary vs real eigenvalue parts plot
    plt.plot(np.real(λ_plunge)[vs <= 303], np.imag(λ_plunge)[vs <= 303], "o--", markersize=3, color="red", label= "plunge mode")
    plt.plot(np.real(λ_torsion)[vs <= 303], np.imag(λ_torsion)[vs <= 303], "o--", markersize=3, color="blue", label= "torsion mode")
    plt.plot(np.real(λ_flap)[vs <= 303], np.imag(λ_flap)[vs <= 303], "o--", markersize=3, color="green", label= "flap mode")

    plt.grid()
    plt.xlabel("Re(λ) [rad/s]")
    plt.ylabel("Im(λ) [rad/s]")
    # plt.ylim([-1, 370])
    # plt.xlim([-20, 15])
    plt.minorticks_on()
    plt.legend()
    plt.savefig(name_eigenvalue_eigenvalue, bbox_inches="tight")
    plt.close("all")
