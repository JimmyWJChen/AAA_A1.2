import scipy
import numpy as np
import scipy.integrate
import AAA
import AAA.Qmatrix
import matplotlib.pyplot as plt


def solve(nonlinear_aeroelastic_section, x_0, tmax):
    """
    Solves nonlinear aeroelastic section as function of time.
    Keeps track of zero crossings (h = 0) and local peaks (hdot = 0)
    
    x_0     -> initial heave amplitude
    tmax -> Propagation length
    """
    Q = nonlinear_aeroelastic_section.Q
    q_n = nonlinear_aeroelastic_section.q_n
    Kh7 = nonlinear_aeroelastic_section.K_h7

    # Zero crossing event
    def event_zero_crossing(t, y):
        return y[0]
    
    # Peak event
    def event_peak(t, y):
        return y[3]

    # The first 3 degrees of freedom can be given (initial pertubation) or the whole state vector
    if len(x_0) == 3:
        y_0 = np.zeros(Q.shape[0])
        y_0[:3] = x_0
    else:
        y_0 = x_0
    
    def y_dot(t, y):
        y = Q @ y + q_n * Kh7 * y[0]**7
        return y
    
    result = scipy.integrate.solve_ivp(y_dot, [0, tmax], y_0, method="Radau", dense_output=True, events=[event_zero_crossing, event_peak])
    return result
    

def velocity_sweep(structural_section, ρ, h_0, vs, debug_plots = False):
    """
    Takes in a structural section, and computes LCO amplitude and frequency given an array of velocity inputs, from the full nonlinear equations

    ρ           -> air density
    h_0         -> initial guess for first LCO at first velocity
    vs          -> array of velocities desired to be tested
    debug_plots -> shows solution at every velocity as function of time
    """
    # Initial solution at v_f + 1 [m/s]
    aeroelastic_section = AAA.Qmatrix.AeroelasticSection(structural_section, ρ, vs[0])
    aeroelastic_section.set_up_nonlinear_part()

    # Propagate first for 20 seconds of to obtain a good initial state vector
    sol = solve(aeroelastic_section, [h_0, 0, 0], 20)
    y = sol.y
    last_y = y[:, -1]

    As = []
    ωs = []

    # Interval between subsequent solutions
    tmax = 2
    for v in vs:
        aeroelastic_section = AAA.Qmatrix.AeroelasticSection(structural_section, ρ, v)
        aeroelastic_section.set_up_nonlinear_part()
        # Propagates new solution for 2 sesconds given the new velocity but last y as initial guess
        sol = solve(aeroelastic_section, last_y, tmax)

        # Take last 10 zero crossings to compute frequency
        t_zero_crossings = sol.t_events[0]
        dt = np.diff(t_zero_crossings)
        T_avg = 2 * np.mean(dt[-10:])
        ω_LCO = 2*np.pi / T_avg
        ωs.append(ω_LCO)

        # Take last 10 peaks to compute amplitude
        t_peaks = sol.t_events[1]
        peaks = sol.sol(t_peaks)[0, -10:]

        # A_LCO = np.mean(abs(peaks))
        A_LCO = max(abs(peaks))
        As.append(A_LCO)

        # Shows peaks and zero crossings over propagated solution
        if debug_plots:
            t = np.linspace(0, tmax, 10000)
            y = sol.sol(t)
            h = y[0, :]

            plt.plot(t, h, "Continuous solution")
            plt.scatter(t_peaks, sol.sol(t_peaks)[0, :], label = f"Found peaks, A_LCO = {A_LCO:#.04g} [m]")
            plt.scatter(t_zero_crossings, sol.sol(t_zero_crossings)[0, :], label = f"Found zero crossings, ω_LCO = {ω_LCO:#.04g} [rad/s]")
            plt.show()  

        last_y = sol.sol(tmax)

    return As, ωs
    