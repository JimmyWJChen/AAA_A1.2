import AAA
import AAA.Qmatrix
import scipy
import numpy as np


def get_flutter_speed(structural_section: AAA.Qmatrix.StructuralSection, ρ, v_0):
    """
    Finds and returns the flutter speed of a given structural section.

    If Newtons method fail it will try bisection search between 0 and 200 m/s, if that fails it will throw out an error.
    """
    def get_residual(v, return_full_eigenvalue=False):
        aeroelastic_section = AAA.Qmatrix.AeroelasticSection(
            structural_section, ρ, v)
        Q = AAA.Qmatrix.get_Q_matrix(aeroelastic_section, Jones=False)
        λ, U = scipy.linalg.eig(Q)

        # Filter out all parts with 0 imaginary part
        n = np.count_nonzero(np.imag(λ))
        assert n % 2 == 0, "uneven number of eigenvalues with imaginary component"

        indices = np.argsort(np.imag(λ))
        λ = λ[indices][:n // 2]

        i_λ_flutter = np.argmax(np.real(λ))
        λ_flutter = λ[i_λ_flutter]

        if return_full_eigenvalue:
            U_flutter = U[:, indices][:, :n//2][:3, i_λ_flutter]
            return λ_flutter, U_flutter
        else:
            return np.real(λ_flutter)

    v_f = scipy.optimize.newton(get_residual, v_0)
    if v_f < 0:
        print("[WARNING] Newtons method found negative solution, please try another initial guess. Falling back to bisection search...")
        v_f = scipy.optimize.bisect(get_residual, 0, 200)

    λ_f, U_f = get_residual(v_f, return_full_eigenvalue=True)
    ω_f = abs(np.imag(λ_f))

    return v_f, ω_f, U_f


def get_LCO_flutter_speeds(structural_input: AAA.Qmatrix.StructuralSectionInput, ρ: float, v_0: float, A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the flutter speed and frequency for each Ai in A. 

    parameters
    ----------
    structural_input: object
        An instance of the StructuralSectionInput class. 
    ρ: float
        The air density.
    v_0: float
        An initial guess for the flutter speed.
    A: np.ndarray
        An array of amplitudes of the linearised nonlinear DOF.

    returns
    -------
    v_f: np.ndarray
        An array of flutter speeds for each amplitude Ai.
    ω_f: np.ndarray
        An array of flutter frequencies for each amplitude Ai.

    """
    # since the eq linearised nonlinear stiffness depends on Ai, we actually have to instantiate a section class for each Ai.
    # that's why I pass an input class which only contains the 'static' inputs for the section.

    # declare v_f and ω_f; they will have the same length as A
    v_f = np.zeros_like(A)
    ω_f = np.zeros_like(A)

    # find flutter speed and frequency for each amplitude Ai
    for i, Ai in enumerate(A):
        # instantiate a nonlinear structural section
        NLSection = AAA.Qmatrix.StructuralSection(
            *structural_input.input_vector_nonlinear, Ai, True)
        # call get_flutter_speed to obtain flutter speed and frequency
        v_f_i, ω_f_i, _ = get_flutter_speed(NLSection, ρ, v_0)
        v_f[i] = v_f_i
        ω_f[i] = ω_f_i
        v_0 = v_f_i

    return v_f, ω_f


def get_LCO_branches(A: np.ndarray, v_f: np.ndarray, ω_f: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """
    Extracts the LCO branches from the complete xy data. 

    parameters
    ----------
    A: np.ndarray
        The array of nonlinear DOF amplitudes.
    v_f: np.ndarray
        The array of flutter speeds.
    ω_f: np.ndarray
        The array of flutter frequencies.


    returns
    -------
    A_list: list
        The list of amplitudes for each LCO branch
    v_list: list
        The list of velocities for each LCO branch
    ω_list: list
        The list of frequencies for each LCO branch
    σ_list: list
        The list of decay factors for each LCO branch
    """
    # find the folds
    # use the fact that at the folds, the gradient of dU/dA is 0, so gradient changes sign
    dVdA = np.gradient(v_f, A)
    idx_folds = np.where(np.diff(np.sign(dVdA)))[0]
    # declare output lists
    A_list = []
    v_list = []
    ω_list = []

    for i, idx in enumerate(idx_folds):
        if not idx == idx_folds[-1]:
            next_idx = idx_folds[i+1]
            A_branch = A[idx:next_idx]
            v_branch = v_f[idx:next_idx]
            ω_branch = ω_f[idx:next_idx]

        else:
            A_branch = A[idx:]
            v_branch = v_f[idx:]
            ω_branch = ω_f[idx:]

        A_list.append(A_branch)
        v_list.append(v_branch)
        ω_list.append(ω_branch)

    return A_list, v_list, ω_list


def get_flutter_eigenvalue(structural_section: AAA.Qmatrix.StructuralSection, ρ: float, v: float) -> tuple[float, float]:
    """
    Calculates the real and imaginary part of the flutter eigenvalue.
    """
    aeroelastic_section = AAA.Qmatrix.AeroelasticSection(
        structural_section, ρ, v)
    Q = AAA.Qmatrix.get_Q_matrix(aeroelastic_section, Jones=False)
    λ, U = scipy.linalg.eig(Q)

    # Filter out all parts with 0 imaginary part
    n = np.count_nonzero(np.imag(λ))
    assert n % 2 == 0, "uneven number of eigenvalues with imaginary component"

    indices = np.argsort(np.imag(λ))
    λ = λ[indices][:n // 2]

    i_λ_flutter = np.argmax(np.real(λ))
    λ_flutter = λ[i_λ_flutter]
    σ_f = np.real(λ_flutter)
    ω_f = np.abs(np.imag(λ_flutter))
    return σ_f, ω_f


def LCO_branch_stability(A: np.ndarray, v_f: np.ndarray, structural_input: AAA.Qmatrix.StructuralSectionInput, ρ: float, dA: float) -> bool:
    """
    Assesses the stability of an LCO branch by taking an Ai and v_f_i and perturbing the system with dA and -dA, and calculating the Re of the eigenvalue.

    parameters
    ----------
    A: np.ndarray
        The array of amplitudes corresponding to this LCO branch.
    v_f: np.ndarray
        The array of flutter speeds corresponding to this LCO branch.
    structural_input: object
        An instance of the StructuralSectionInput class.
    ρ: float
        The air density.
    dA: float
        The amplitude perturbation.

    returns
    -------
    stable: bool
        True when LCO is table and False when it's not.
    """
    # declare σ_plus and σ_min
    σ_plus = np.zeros_like(A[1:])
    σ_min = np.zeros_like(A[1:])
    # loop over A
    for i, Ai in enumerate(A[1:]):
        vi = v_f[i+1]
        NLSection_plus = AAA.Qmatrix.StructuralSection(
            *structural_input.input_vector_nonlinear, Ai+dA, True)
        NLSection_min = AAA.Qmatrix.StructuralSection(
            *structural_input.input_vector_nonlinear, Ai-dA, True)
        σ_i_plus, _ = get_flutter_eigenvalue(NLSection_plus, ρ, vi)
        σ_i_min, _ = get_flutter_eigenvalue(NLSection_min, ρ, vi)
        σ_plus[i] = σ_i_plus
        σ_min[i] = σ_i_min
    stable_outside = True
    stable_inside = False
    # print(f'σ_plus = {σ_plus}')
    # print(f'σ_min = {σ_min}')
    if (σ_plus > 0).any():
        stable_outside = False
    if (σ_min > 0).any():
        stable_inside = True
    stable = stable_outside and stable_inside
    return stable
