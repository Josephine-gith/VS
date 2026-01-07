import numpy as np
import matplotlib.pyplot as plt

## Modèle de Hescher-Bukley + Maxwell

# Temps final N*dt
N = 10**5  # Nombre d'itérations
dt = 1e-4  # Pas de temps, en sec

# Paramètres de l'expérience
h0 = 0.1  # m
r0 = 0.1  # m
M = 0.05  # kg

# Propriétés du matériau
rho = 1e3  # kg/m3
k = 40  # Pa.s
G = 1e2  # Pa
Lambda = k / G
tau0 = 15  # Pa
Gamma0 = tau0 / G
sigma = 5e-2  # Pa.m
m = 1

g = 9.81  # m/s2

# Initialisation
R = np.zeros(N + 1)
R[0] = r0
U = np.zeros(N + 1)
Gamma_rev = np.zeros(N + 1)

# Itération
for i in range(N):
    R[i + 1] = dt * U[i] + R[i]
    tauT = (
        G * Gamma_rev[i]
        - rho * g * h0 * (r0 / R[i]) ** 2
        - M * g / (np.pi * R[i])
        + sigma * r0 * (r0 / R[i] - 1)
    )
    U[i + 1] = -dt * (R[i] / r0) ** 2 / (rho * h0) * tauT + U[i]
    Gamma_rev[i + 1] = (
        dt
        * (
            U[i] * (R[i] / r0) ** 2 / h0
            - max(0, (abs(Gamma_rev[i]) - Gamma0) ** (1 / m))
            / Lambda ** (1 / m)
            * np.sign(Gamma_rev[i])
        )
        + Gamma_rev[i]
    )

# Sortie
print(f"Le rayon au bout de {N * dt}s est {R[N]:.4f} m.")

T = dt * np.arange(N + 1)

plt.figure()
plt.subplot(1, 3, 1)
plt.plot(T, R, label="r(t)")
plt.title("r(t)")
plt.grid()

plt.subplot(1, 3, 2)
plt.plot(T, U, label="u(t)")
plt.title("u(t)")
plt.grid()

plt.subplot(1, 3, 3)
plt.plot(T, Gamma_rev, label="gamma_rev(t)")
plt.title("gamma_rev(t)")
plt.grid()


plt.show()
