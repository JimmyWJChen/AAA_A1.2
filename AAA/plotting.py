import matplotlib.pyplot as plt
import numpy as np
import scipy
import AAA
import AAA.Qmatrix
from matplotlib.animation import FuncAnimation

import AAA.functions


def plot_eigenmodes_subplot(structural_section: AAA.Qmatrix.StructuralSection, U: np.ndarray, title: str, norm=True, noyticklabels=False, ylim=[-1.5, 1], loc="upper right", multiplier=1, legend=True) -> None:
    """

    Plots the eigenmode U

    structural_section  -> section that contains structural matrices
    U                   -> eigenvector, which should be [h, α, β].T
    title               -> title that should be above the plot, may be left empty
    norm                -> normalises the eigenvector
    noyticklabels       -> removes ytick labels (for multiple subplots side by side)
    ylim                -> vertical limit in the plot
    loc                 -> position of the legend
    multiplier     -> adjusts heave magnitude for shorter / longer airfoils
    legend              -> shows legend if true
    """

    if norm:
        U /= np.linalg.norm(U)

    h_i = U[0] * multiplier
    θ_i = U[1] * multiplier
    β_i = U[2] * multiplier
    a = structural_section.a
    b = structural_section.b
    c = structural_section.c

    plt.xlim([-(1.3 + a)*b, (1.3 - a)*b])

    X_EA = 0
    Y_EA = -h_i
    # Location of the leading edge
    X_LE = X_EA - (a + 1) * b * np.cos(θ_i)
    Y_LE = Y_EA + (a + 1) * b * np.sin(θ_i)

    # Location of the hinge
    X_hinge = X_EA + (c - a) * b * np.cos(θ_i)
    Y_hinge = Y_EA - 1 * (c - a) * b * np.sin(θ_i)

    # Location of the trailing edge
    L_flap = (c - a) * b
    X_TE = X_hinge + L_flap * np.cos(β_i + θ_i)
    Y_TE = Y_hinge - L_flap * np.sin(β_i + θ_i)

    if noyticklabels:
        plt.gca().set_yticklabels([])
    else:
        plt.ylabel("y [m]")

    plt.title(title)
    plt.plot([X_LE, X_hinge], [Y_LE, Y_hinge], label="Airfoil")
    plt.plot([X_hinge, X_TE], [Y_hinge, Y_TE], label="Flap")
    plt.scatter(X_EA, Y_EA, s=5, color="blue", label="Elastic Axis")
    if legend:
        plt.legend(loc=loc)
    plt.grid()
    plt.ylim(ylim)
    plt.gca().set_aspect("equal")
    plt.xlabel("x [m]")


def plot_coupled_structural_eigenmodes(structural_section: AAA.Qmatrix.StructuralSection, name, ylim=[-1, 1], multiplier=1):
    """
    Plots all coupled eigenmodes in a single plot

    multiplier -> corrects heave motion if the airfoil is very short (rotation angles are fine)
    """
    ωs, Us = structural_section.compute_coupled_undamped_eigenmodes()

    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plot_eigenmodes_subplot(
        structural_section, Us[:, 0], f"ω = {ωs[0]:#.04g} [rad/s]", norm=True, ylim=ylim, multiplier=multiplier, legend=False)
    plt.subplot(132)
    plot_eigenmodes_subplot(structural_section, Us[:, 1], f"ω = {ωs[1]:#.04g} [rad/s]",
                            norm=True, noyticklabels=True, ylim=ylim, multiplier=multiplier, legend=False)
    plt.subplot(133)
    plot_eigenmodes_subplot(structural_section, Us[:, 2], f"ω = {ωs[2]:#.04g} [rad/s]",
                            norm=True, noyticklabels=True, ylim=ylim, multiplier=multiplier)
    plt.subplots_adjust(hspace=0)
    plt.savefig(name, bbox_inches="tight")
    plt.close("all")


def plot_uncoupled_structural_eigenmodes(structural_section: AAA.Qmatrix.StructuralSection, name, ylim=[-1, 1], multiplier=1):
    """
    Plots all coupled eigenmodes in a single plot

    multiplier -> corrects heave motion if the airfoil is very short (rotation angles are fine)
    """
    ωs, Us = structural_section.compute_uncoupled_undamped_eigenmodes()

    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plot_eigenmodes_subplot(
        structural_section, Us[:, 0], f"ω = {ωs[0]:#.04g} [rad/s]", norm=True, ylim=ylim, multiplier=multiplier, legend=False)
    plt.subplot(132)
    plot_eigenmodes_subplot(structural_section, Us[:, 1], f"ω = {ωs[1]:#.04g} [rad/s]",
                            norm=True, noyticklabels=True, ylim=ylim, multiplier=multiplier, legend=False)
    plt.subplot(133)
    plot_eigenmodes_subplot(structural_section, Us[:, 2], f"ω = {ωs[2]:#.04g} [rad/s]",
                            norm=True, noyticklabels=True, ylim=ylim, multiplier=multiplier)
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

    vs = np.linspace(0, max_v, 100)
    λs, vs_scatter, λ_plunge, λ_torsion, λ_flap = AAA.functions.get_eigenvalues(structural_section, ρ, vs)

    plt.figure(figsize=(12, 4))

    # Create velocity against real eigenvalue parts subplot
    plt.subplot(121)
    plt.scatter(vs_scatter, np.real(λs), alpha=0.2, color="grey", s=2)
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
    plt.scatter(vs_scatter, np.imag(λs), alpha=0.2, color="grey", s=2)
    plt.plot(vs, np.imag(λ_plunge), color="red", label="plunge mode")
    plt.plot(vs, np.imag(λ_torsion), color="blue", label="torsion mode")
    plt.plot(vs, np.imag(λ_flap), color="green", label="flap mode")
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
    plt.plot(np.real(λ_plunge), np.imag(λ_plunge), "o--",
             markersize=3, color="red", label="plunge mode")
    plt.plot(np.real(λ_torsion), np.imag(λ_torsion), "o--",
             markersize=3, color="blue", label="torsion mode")
    plt.plot(np.real(λ_flap), np.imag(λ_flap), "o--",
             markersize=3, color="green", label="flap mode")
    plt.grid()
    plt.xlabel("Re(λ) [rad/s]")
    plt.ylabel("Im(λ) [rad/s]")
    plt.minorticks_on()
    plt.legend()
    plt.savefig(name_eigenvalue_eigenvalue, bbox_inches="tight")
    plt.close("all")


def animate_DOFs(Us: list, ts: list, structural_section: AAA.Qmatrix.StructuralSection, name, multiplier=1):
    """
    Creates an animation of the heave, pitch and flap DOFs, 60 fps

    Us -> list of degrees of freedom [h, θ, β]
    ts -> list of timepoints corresponding to U_i
    multiplier -> degree of multiplication of the degrees of freedom of flutter mode.
    """
    a = structural_section.a
    b = structural_section.b
    c = structural_section.c

    fig, axis = plt.subplots()
    fig.set_size_inches(20, 20)
    axis.set_aspect("equal")
    airfoil = axis.plot([], [])[0]
    flap = axis.plot([], [])[0]
    elastic_axis = axis.plot([], [], "o")[0]

    plt.xlim([-(1.2 + a)*b, (1.2 - a)*b])
    plt.ylim([-1.5 * b, 1.5 * b])

    def update(frame):
        U = Us[frame]
        h_i = U[0] * multiplier
        θ_i = U[1] * multiplier
        β_i = U[2] * multiplier

        X_EA = 0
        Y_EA = -h_i
        # Location of the leading edge
        X_LE = X_EA - (a + 1) * b * np.cos(θ_i)
        Y_LE = Y_EA + (a + 1) * b * np.sin(θ_i)

        # Location of the hinge
        X_hinge = X_EA + (c - a) * b * np.cos(θ_i)
        Y_hinge = Y_EA - 1 * (c - a) * b * np.sin(θ_i)

        # Location of the trailing edge
        L_flap = (c - a) * b
        X_TE = X_hinge + L_flap * np.cos(β_i + θ_i)
        Y_TE = Y_hinge - L_flap * np.sin(β_i + θ_i)

        airfoil.set_data([X_LE, X_hinge], [Y_LE, Y_hinge])
        flap.set_data([X_hinge, X_TE], [Y_hinge, Y_TE])
        elastic_axis.set_data([X_EA], [Y_EA])

        return airfoil, flap, elastic_axis

    animation = FuncAnimation(fig, update, len(
        Us), interval=1000 * (1 / 60), repeat=False)
    animation.save(name)


def bifurcation_plots_eq_lin(A: np.ndarray, v_f: np.ndarray, ω_f: np.ndarray, structural_input: AAA.Qmatrix.StructuralSectionInput, ρ: float, dA: float, path_A_v: str, path_ω_v: str, ω_flap = [], vs_exact = [], As_exact = [], ωs_exact = []) -> None:
    """
    Traces the LCO branches of the equivalent linearised system.

    parameters
    ----------
    A: np.ndarray
        The array of nonlinear DOF amplitudes.
    v_f: np.ndarray
        The array of flutter speeds for each amplitude Ai. 
    ω_f: np.ndarray
        The array of flutter frequencies for each amplitude Ai.
    structural_input: object
        An instance of the StructuralSectionInput class.
    ρ: float
        The air density.
    dA: float
        The amplitude perturbation.
    path_A_v: str
        The relative path for saving the amplitude-flutter speed plot.
    path_ω_v: str
        The relative path for saving the flutter frequency-flutter speed plot.
    """
    # get the LCO branches
    print(v_f)
    A_list, v_list, ω_list = AAA.functions.get_LCO_branches(A, v_f, ω_f)
    # find stability
    colours = []
    labels = []
    for i, Ai in enumerate(A_list):
        stable = AAA.functions.LCO_branch_stability(
            Ai, v_list[i], structural_input, ρ, dA)
        if stable:
            colour = 'blue'
            label = 'stable LCO'
        else:
            colour = 'red'
            label = 'unstable LCO'
        colours.append(colour)
        labels.append(label)
    # set up figure
    fig, ax = plt.subplots(dpi=200)
    ax.plot(v_f, A, label=f'full trajectory')
    for i, Ai in enumerate(A_list):
        ax.plot(v_list[i], Ai, "o--",
                markersize=3, color=colours[i], label=f'{labels[i]}')
    if ωs_exact != []:
        ax.plot(vs_exact, As_exact, "x--", color = "black", markersize = 5, label="Exact nonlinear solution")
    ax.grid()
    ax.legend()
    ax.set_xlabel(f'V [m/s]')
    ax.set_ylabel('$A_h$ [m]')
    ax.minorticks_on()
    plt.tight_layout()
    plt.savefig(f'{path_A_v}', bbox_inches='tight')
    plt.close('all')
    fig, ax = plt.subplots(dpi=200)
    ax.plot(v_f, ω_f, label=f'full trajectory')
    for i, v_i in enumerate(v_list):
        ax.plot(v_i, ω_list[i],  "o--",
                markersize=3, color=colours[i], label=f'{labels[i]}')
    
    if list(ωs_exact) != []:
        ax.plot(vs_exact, ωs_exact, "x--", color = "black", markersize = 5, label="Steady state nonlinear solution")

    if list(ω_flap) != []:
        ax.plot(v_f, ω_flap, label="Flap mode natural frequency")
        
    ax.grid()
    ax.legend()
    ax.set_xlabel(f'V [m/s]')
    ax.set_ylabel(f'ω [rad/s]')
    ax.minorticks_on()
    plt.tight_layout()
    plt.savefig(f'{path_ω_v}', bbox_inches='tight')
    plt.close('all')


def plot_nonlinear_velocity_sweep(vs, As, ωs, name):
    """
    vs -> list of velocities
    As -> LCO amplitudes
    ωs -> frequencies
    """
    plt.subplot(211)
    plt.plot(vs, As, "o--")
    plt.ylabel(r"$A_{h, LCO}$ [m]")
    plt.grid()
    plt.minorticks_on()
    plt.subplot(212)
    plt.plot(vs, ωs, "o--")
    plt.grid()
    plt.minorticks_on()
    plt.xlabel("v [m/s]")
    plt.ylabel(r"$ω_{h, LCO}$ [m]")
    
    plt.savefig(name, bbox_inches = "tight")
    plt.close("all")


def plot_nonlinear_solution(t, solution, name, αβmultiplier = 1):    
    """
    Plots h, α, β into a single plot as a function of time

    t is an input array with time points of interest. These must lie 0 < t_i <= tmax, where tmax is the simulation time
    """
    y = solution.sol(t)
    h = y[0]
    α = y[1]
    β = y[2]

    t_zero_crossings = solution.t_events[0]
    t_maxima = solution.t_events[1]

    t_zero_crossings = t_zero_crossings[(t_zero_crossings >= min(t)) & (t_zero_crossings <= max(t))]
    t_maxima = t_maxima[(t_maxima >= min(t)) & (t_maxima <= max(t))]

    y_maxima = solution.sol(t_maxima)
    h_maxima = y_maxima[0]


    plt.figure(figsize=(12, 4))
    
    if αβmultiplier == 1:
        plt.plot(t, β, label = "β [rad]")
        plt.plot(t, α, label = "α [rad]")
    else:
        plt.plot(t, αβmultiplier*β, label = fr"{αβmultiplier} $\times$ β [rad]")
        plt.plot(t, αβmultiplier*α, label = fr"{αβmultiplier} $\times$ α [rad]")
        
    plt.plot(t, h, label = "h [m]", color = "blue")
    plt.scatter(t_zero_crossings, np.zeros_like(t_zero_crossings), label="Zero Crossing Heave", color="blue", s = 15)
    plt.scatter(t_maxima, h_maxima, label="Maxima Heave", color = "red", s = 15)
    plt.xlabel("time [s]")
    plt.ylabel("amplitude [-]")
    plt.grid()
    plt.legend()
    plt.savefig(name, bbox_inches = "tight")
    plt.show()