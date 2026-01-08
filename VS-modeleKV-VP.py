import numpy as np
import matplotlib.pyplot as plt

## Régimes visqueux et plastique
G = 0

# Modèle de Kelvin Voigt


def r_fin(h0=0.1, r0=0.1, rho=1.0e3, k=40.0, tau0=15.0, dt=1e-4, N=10**5):
    M = 0  # car dam break sans compression supplémentaire
    sigma = 0  # négligé

    g = 9.81  # m/s2

    # Initialisation
    R = np.zeros(N + 1)
    R[0] = r0
    U = np.zeros(N + 1)
    Gamma = np.zeros(N + 1)

    # Itération
    for i in range(N):
        R[i + 1] = dt * U[i] + R[i]
        tauT = (
            k * U[i] * (R[i] / r0) ** 2 / h0
            + tau0
            + G * Gamma[i]
            - rho * g * h0 * (r0 / R[i]) ** 2
            - M * g / (np.pi * R[i])
            + sigma * r0 * (r0 / R[i] - 1)
        )
        Gamma[i + 1] = dt * U[i] * (R[i] / r0) ** 2 / h0 + Gamma[i]
        U[i + 1] = -dt * (R[i] / r0) ** 2 / (rho * h0) * tauT + U[i]

    # Sortie
    uc = h0 * (rho * g * h0 - tau0) / k
    Ga = rho * g * h0 / (k * uc / h0)
    x1 = (Ga * h0 / r0) ** (1 / 5)

    Bn = tau0 / (rho * g * h0)
    x2 = (h0 / (r0 * Bn)) ** (1 / 3)

    y = R[-1] / r0 - 1

    return x1, x2, y, U[-1]


## R final pour différents tau0 à Bn fixé, pour les deux régimes


def fixe_Bn(h0, Bn, rho=1e3, g=9.81):
    return Bn * rho * g * h0


def fixe_Ga(h0, Ga, rho=1e3, g=9.81, uc=1):  # Quelle val de Uc ?
    return rho * g * h0**2 / (uc * Ga)


H0 = [0.05, 0.1, 0.3, 0.8, 1.0]
Sim_visc = np.zeros((len(H0), 4))
Sim_plas = np.zeros((len(H0), 4))

for i, h0 in enumerate(H0):
    Sim_visc[i] = r_fin(h0=h0, tau0=fixe_Bn(h0, 0.003), k=50)
    Sim_plas[i] = r_fin(h0=h0, k=10, tau0=20)

print(Sim_visc, Sim_plas)

plt.figure()
plt.subplot(1, 2, 1)
plt.scatter(Sim_visc[:, 0], Sim_visc[:, 2])
plt.title("Bn=0.003, régime visqueux")
plt.xlabel("(Ga * h0 / r0) ** (1 / 5)")
plt.ylabel("r_fin/r0 - 1")
plt.grid()

plt.subplot(1, 2, 2)
plt.scatter(Sim_plas[:, 1], Sim_plas[:, 2])
plt.title("Ga=50, régime plastique")
plt.xlabel("(h0 / (r0 * Bn)) ** (1 / 3)")
plt.ylabel("r_fin/r0 - 1")
plt.grid()
plt.show()
