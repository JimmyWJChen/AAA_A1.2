import scipy
import numpy as np
import scipy.integrate
import AAA
import AAA.Qmatrix
import matplotlib.pyplot as plt

def solve(nonlinear_aeroelastic_section, x_0, tmax):
    Q = nonlinear_aeroelastic_section.Q
    q_n = nonlinear_aeroelastic_section.q_n
    Kh7 = nonlinear_aeroelastic_section.Kh7

    # The first 3 degrees of freedom can be given (initial pertubation) or the whole state vector
    if len(x_0) == 3:
        y_0 = np.zeros(Q.shape[0])
        y_0[:3] = x_0
    else:
        y_0 = x_0
    
    def y_dot(t, y):
        y = Q @ y + q_n * Kh7 * y[0]**7
        return y
    
    result = scipy.integrate.solve_ivp(y_dot, [0, tmax], y_0, method="Radau", dense_output=True)
    return result
    

def velocity_sweep(structural_section, ρ, Kh7, h_0, vs):
    # Initial solution at v_f + 1 [m/s]

    aeroelastic_section = AAA.Qmatrix.NonlinearAeroelasticSection(structural_section, ρ, vs[0], Kh7)

    # Propagate first for 20 seconds of to obtain a good initial state vector
    sol = solve(aeroelastic_section, [h_0, 0, 0], 20)

    n = 2**20
    t = np.linspace(10, 20, n)
    dt = t[1] - t[0]
    y = sol.sol(t)
    h = y[0, :]
    Y = scipy.fft.rfft(h)
    Y_mag = 2 / n * abs(Y)
    f = scipy.fft.rfftfreq(n, dt) * 2 * np.pi

    ω_LCO = f[np.argmax(Y_mag)]

    last_y = y[:, -1]

    
    As = []
    for v in vs:
        aeroelastic_section = AAA.Qmatrix.NonlinearAeroelasticSection(structural_section, ρ, v, Kh7)
        sol = solve(aeroelastic_section, last_y, 2)
        n = 2**16

        t = np.linspace(1, 2, n)
        dt = t[1] - t[0]
        y = sol.sol(t)
        h = y[0, :]
        Y = scipy.fft.rfft(h)
        Y_mag = 2 / n * abs(Y)
        f = scipy.fft.rfftfreq(n, dt) * 2 * np.pi

        ω_LCO = f[np.argmax(Y_mag)]
        A_LCO = (max(h) - min(h))/2
        As.append(A_LCO)
        print(v, A_LCO)

        last_y = y[:, -1]

        # plt.subplot(211)
        # plt.plot(t, y[0, :])
        # plt.subplot(212)
        # plt.plot(f[f < 100], Y_mag[f<100])
        # plt.show()
        # print(v, ω_LCO)
        # print(ω_LCO)

    plt.plot(vs, As, "o--")
    plt.grid()
    plt.xlabel("v [m/s]")
    plt.ylabel(r"$A_h$ [m]")
    plt.show()
    
