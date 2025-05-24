import matplotlib.pyplot as plt
import numpy as np
import scipy
import AAA
import AAA.Qmatrix


def plot_eigenmodes_subplot(structural_section: AAA.Qmatrix.StructuralSection, U: np.ndarray, title: str, norm=True, noyticklabels = False, ylim = [-1.5, 1], loc="upper right", heavemultiplier = 1, legend=True) -> None:
    """

    Plots the eigenmode U

    structural_section  -> section that contains structural matrices
    U                   -> eigenvector, which should be [h, α, β].T
    title               -> title that should be above the plot, may be left empty
    norm                -> normalises the eigenvector
    noyticklabels       -> removes ytick labels (for multiple subplots side by side)
    ylim                -> vertical limit in the plot
    loc                 -> position of the legend
    heavemultiplier     -> adjusts heave magnitude for shorter / longer airfoils
    legend              -> shows legend if true
    """

    if norm:
        U /= np.linalg.norm(U)

    h_i = U[0] * heavemultiplier
    θ_i = U[1]
    β_i = U[2]

    X_LE_0 = -structural_section.b * (structural_section.a + 1)
    X_hinge_0 = structural_section.b * (structural_section.c - structural_section.a)
    X_TE_0 = X_hinge_0 + structural_section.b * (1 - structural_section.c)

    X_LE = -structural_section.b * (structural_section.a + 1) * np.cos(θ_i)
    Y_LE = -h_i + structural_section.b * (structural_section.a + 1) * np.sin(θ_i)

    X_hinge = structural_section.b * (structural_section.c - structural_section.a) * np.cos(θ_i)
    Y_hinge = -h_i - structural_section.b * (structural_section.c - structural_section.a) * np.sin(θ_i)

    X_TE = X_hinge + structural_section.b * (1 - structural_section.c) * np.cos(θ_i + β_i)
    Y_TE = Y_hinge - structural_section.b * (1 - structural_section.c) * np.sin(θ_i + β_i)

    if noyticklabels:
        plt.gca().set_yticklabels([])
    else:
        plt.ylabel("y [m]")

    plt.title(title)

    plt.plot([X_LE_0, X_TE_0], [0, 0], label = "Undeformed position", color = "black", alpha=0.8)
    plt.plot([X_LE, X_hinge], [Y_LE, Y_hinge], label = "Airfoil")
    plt.plot([X_hinge, X_TE], [Y_hinge, Y_TE], label = "Flap")
    plt.scatter(0, -h_i, s = 5, color="blue", label="Elastic Axis")
    if legend:
        plt.legend(loc = loc)
    plt.grid()
    plt.ylim(ylim)
    plt.gca().set_aspect("equal")
    plt.xlabel("x [m]")


def plot_coupled_structural_eigenmodes(structural_section: AAA.Qmatrix.StructuralSection, name, ylim = [-1, 1], heavemultiplier = 1):
    """
    Plots all coupled eigenmodes in a single plot

    heavemultiplier -> corrects heave motion if the airfoil is very short (rotation angles are fine)
    """
    ωs, Us = structural_section.compute_coupled_undamped_eigenmodes()

    plt.figure(figsize=(12, 4)) 
    plt.subplot(131)
    plot_eigenmodes_subplot(structural_section, Us[:, 0], f"ω = {ωs[0]:#.04g} [rad/s]", norm=True, ylim=ylim, heavemultiplier = heavemultiplier, legend=False)
    plt.subplot(132)
    plot_eigenmodes_subplot(structural_section, Us[:, 1], f"ω = {ωs[1]:#.04g} [rad/s]", norm=True, noyticklabels=True, ylim=ylim, heavemultiplier = heavemultiplier, legend=False)
    plt.subplot(133)
    plot_eigenmodes_subplot(structural_section, Us[:, 2], f"ω = {ωs[2]:#.04g} [rad/s]", norm=True, noyticklabels=True, ylim=ylim, heavemultiplier = heavemultiplier)
    plt.subplots_adjust(hspace=0)
    plt.savefig(name, bbox_inches="tight")
    plt.close("all")


def plot_uncoupled_structural_eigenmodes(structural_section: AAA.Qmatrix.StructuralSection, name, ylim = [-1, 1], heavemultiplier = 1):
    """
    Plots all coupled eigenmodes in a single plot
    
    heavemultiplier -> corrects heave motion if the airfoil is very short (rotation angles are fine)
    """
    ωs, Us = structural_section.compute_uncoupled_undamped_eigenmodes()

    plt.figure(figsize=(12, 4)) 
    plt.subplot(131)
    plot_eigenmodes_subplot(structural_section, Us[:, 0], f"ω = {ωs[0]:#.04g} [rad/s]", norm=True, ylim=ylim, heavemultiplier = heavemultiplier, legend=False)
    plt.subplot(132)
    plot_eigenmodes_subplot(structural_section, Us[:, 1], f"ω = {ωs[1]:#.04g} [rad/s]", norm=True, noyticklabels=True, ylim=ylim, heavemultiplier = heavemultiplier, legend=False)
    plt.subplot(133)
    plot_eigenmodes_subplot(structural_section, Us[:, 2], f"ω = {ωs[2]:#.04g} [rad/s]", norm=True, noyticklabels=True, ylim=ylim, heavemultiplier = heavemultiplier)
    plt.subplots_adjust(hspace=0)
    plt.savefig(name, bbox_inches="tight")
    plt.close("all")


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
        Q = AAA.Qmatrix.get_Q_matrix(section, Jones=False) # Q 8

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
    plt.plot(np.real(λ_plunge), np.imag(λ_plunge), "o--", markersize=3, color="red", label= "plunge mode")
    plt.plot(np.real(λ_torsion), np.imag(λ_torsion), "o--", markersize=3, color="blue", label= "torsion mode")
    plt.plot(np.real(λ_flap), np.imag(λ_flap), "o--", markersize=3, color="green", label= "flap mode")
    plt.grid()
    plt.xlabel("Re(λ) [rad/s]")
    plt.ylabel("Im(λ) [rad/s]")
    plt.minorticks_on()
    plt.legend()
    plt.savefig(name_eigenvalue_eigenvalue, bbox_inches="tight")
    plt.close("all")


def plot_flutter_mode(U, λ):
    pass