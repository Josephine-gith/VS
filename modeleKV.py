import numpy as np
import matplotlib.pyplot as plt

## Modèle de Kelvin Voigt

# Temps final N*dt
N = 10**4  # Nombre d'itérations
dt = 1e-3  # Pas de temps, en sec

Di = 0.2  # Nombre de dissipation

# Paramètres de l'expérience
h0 = 0.1  # m
r0 = 0.1  # m
M = 0.05  # kg

# Propriétés du matériau
rho = 1e3  # kg/m3
k = 40  # Pa.s
G = 50  # Pa
tau0 = 30  # Pa
sigma = 5e-2  # Pa.m
m = 1

g = 9.81  # m/s2

eta = Di * tau0 * (g / h0) ** (1 / 2)

# Initialisation
R = np.zeros(N + 1)
R[0] = r0
U = np.zeros(N + 1)
Gamma = np.zeros(N + 1)

# Itération
for i in range(N):
    R[i + 1] = dt * U[i] + R[i]
    gamma_p = U[i] * (R[i] / r0) ** 2 / h0
    tauT = (
        (k + eta) * abs(gamma_p) ** m * np.sign(gamma_p)
        + tau0
        + G * Gamma[i]
        - rho * g * h0 * (r0 / R[i]) ** 2
        - M * g / (np.pi * R[i])
        + sigma * r0 * (r0 / R[i] - 1)
    )
    Gamma[i + 1] = dt * gamma_p + Gamma[i]
    U[i + 1] = -dt * (R[i] / r0) ** 2 / (rho * h0) * tauT + U[i]

# Sortie
print(f"Le rayon au bout de {N * dt}s est {R[N]:.4f} m.")

T = dt * np.arange(N + 1)

plt.figure()
plt.subplot(1, 3, 1)
plt.plot(T, R, label="r(t)")
plt.title("r(t)")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.xlabel("t (en s)")
plt.ylabel("r (en m)")

plt.subplot(1, 3, 2)
plt.plot(T, U, label="u(t)")
plt.title("u(t)")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.xlabel("t (en s)")
plt.ylabel("u (en m/s)")

plt.subplot(1, 3, 3)
plt.plot(T, Gamma, label="gamma(t)")
plt.title("gamma(t)")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.xlabel("t (en s)")
plt.ylabel("gamma")

plt.tight_layout()
plt.show()
