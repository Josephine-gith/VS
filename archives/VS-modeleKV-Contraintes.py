import numpy as np
import matplotlib.pyplot as plt

## Modèle de Kelvin Voigt

# Temps final N*dt
N = 2 * 10**3  # Nombre d'itérations
dt = 1e-3  # Pas de temps, en sec

# Paramètres de l'expérience
h0 = 0.1  # m
r0 = 0.1  # m
M = 0.5  # kg

# Propriétés du matériau
rho = 1e3  # kg/m3
k = 40  # Pa.s
G = 50  # Pa
tau0 = 30  # Pa
sigma = 5e-2  # Pa.m

g = 9.81  # m/s2

# Grandeurs
R = np.zeros(N + 1)
U = np.zeros(N + 1)
Gamma = np.zeros(N + 1)
Contraintes = np.zeros((N + 1, 6))
Contraintes_leg = [
    "Visqueuse",
    "Plastique (contrainte seuil)",
    "Elastique",
    "Compression",
    "Capillaire",
    "Total (= inertie)",
]

# Initialisation
R[0] = r0

tauVisc = 0
tauElas = 0
tauPoids = -rho * g * h0
tauComp = -M * g / (np.pi * r0)
tauCapi = 0
tauT = tauVisc + tau0 + tauElas + tauPoids + tauComp + tauCapi
Contraintes[0, :] = np.array((tauVisc, tau0, tauElas, -tauComp, tauCapi, tauT)) / abs(
    tauPoids
)

# Itération
for i in range(N):
    R[i + 1] = dt * U[i] + R[i]

    tauVisc = k * U[i] * (R[i] / r0) ** 2 / h0
    tauElas = G * Gamma[i]
    tauPoids = -rho * g * h0 * (r0 / R[i]) ** 2
    tauComp = -M * g / (np.pi * R[i])
    tauCapi = sigma * r0 * (r0 / R[i] - 1)
    tauT = tauVisc + tau0 + tauElas + tauPoids + tauComp + tauCapi
    Contraintes[i + 1, :] = np.array(
        (tauVisc, tau0, tauElas, -tauComp, tauCapi, tauT)
    ) / abs(tauPoids)

    Gamma[i + 1] = dt * U[i] * (R[i] / r0) ** 2 / h0 + Gamma[i]
    U[i + 1] = -dt * (R[i] / r0) ** 2 / (rho * h0) * tauT + U[i]

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
plt.plot(T, Gamma, label="gamma(t)")
plt.title("gamma(t)")
plt.grid()

plt.show()


plt.plot(T, Contraintes, label=Contraintes_leg)
plt.title("Contribution de chaque contrainte, normalisée par le poids")
plt.xlabel("Temps (en sec)")
plt.ylabel("Contrainte / tauPoids")
plt.legend()
plt.grid()
plt.show()
