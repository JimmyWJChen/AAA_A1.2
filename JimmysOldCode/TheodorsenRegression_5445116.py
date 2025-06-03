import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.optimize

# k = np.linspace(1e-10, 10, 10000)
k = np.logspace(-3, 3, 10000)

jones = 1 - 0.165 * k / (k - 0.0455j) - 0.335 * k / (k - 0.3j)

J0 = scipy.special.jv(0, k)
J1 = scipy.special.jv(1, k)
Y0 = scipy.special.yn(0, k)
Y1 = scipy.special.yn(1, k)

denom = ((J1 + Y0)**2 + (Y1 - J0)**2)
F = (J1 * (J1 + Y0) + Y1 * (Y1 - J0)) / denom
G = -(Y1 * Y0 + J1 * J0) / denom
C = F + 1j*G

def f(k, p):
    result = 1
    
    half_point = len(p) // 2
    a_s = p[:half_point]
    p_s = p[half_point:]

    assert len(a_s) == len(p_s)

    for i, a in enumerate(a_s):
        p = p_s[i]
        result += a * k / (k + p*1j)

    return result

def residual(p, y, k):
    y_i = f(k, p)
    a = np.abs(y - y_i)
    return a


popt, pcov = scipy.optimize.leastsq(residual, 6*[-0.2], args=(C, k), maxfev=100000)
print(popt)



k = np.logspace(-2, 0.4, 10)

jones = 1 - 0.165 * k / (k - 0.0455j) - 0.335 * k / (k - 0.3j)

J0 = scipy.special.jv(0, k)
J1 = scipy.special.jv(1, k)
Y0 = scipy.special.yn(0, k)
Y1 = scipy.special.yn(1, k)

denom = ((J1 + Y0)**2 + (Y1 - J0)**2)
F = (J1 * (J1 + Y0) + Y1 * (Y1 - J0)) / denom
G = -(Y1 * Y0 + J1 * J0) / denom
C = F + 1j*G

mine = f(k, popt)

plt.figure(figsize=(10, 8))
for i, k_i in enumerate(k):
    plt.text(F[i], G[i] + 0.025, f"k = {k_i:#.03g}", horizontalalignment = "center", verticalalignment = "center", fontsize=12)
plt.plot(F, G, "o--", label="Theodorsen", color="blue")
plt.plot(jones.real, jones.imag, "o--", label = "Jones", color= "green")
plt.plot(mine.real, mine.imag, "o--", label="3 term regression", color = "red")

plt.legend()
plt.gca().set_aspect("equal")
plt.xlim(0.5, 1)
plt.ylim(-0.3, 0.0)

plt.xlabel("Real(S(k))")
plt.ylabel("Im(S(k))")
plt.grid()
plt.savefig("output/Q10Theodorsen.pdf", bbox_inches = "tight")
plt.show()