import numpy as np
import matplotlib.pyplot as plt

## Régimes visqueux et élastique
tau0 = 10

# Modèle de Kelvin Voigt


def r_fin(h0=0.1, r0=0.1, rho=1e3, k=40, G=50.0, dt=1e-3, N=2 * 10**3):
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
    uc = h0 * (rho * g * h0 - tau0 - G * R[-1] ** 3 / (h0 * r0**2)) / k
    Ga = rho * g * h0 / (k * uc / h0)
    x1 = (Ga * h0 / r0) ** (1 / 5)

    """
    Bn = tau0 / (rho * g * h0)
    x2 = (h0 / (r0 * Bn)) ** (1 / 3)
    """

    El = G * r0 / (rho * g * h0**2)
    x2 = (h0 / (r0 * El)) ** (1 / 6)

    y = R[-1] / r0 - 1

    return x1, x2, y, U[-1]


## R final pour différents k à Bn fixé, pour les deux régimes


def fh0(G, El, rho=1e3, g=9.81, r0=0.1):
    return (G * r0 / (rho * g * El)) ** (1 / 2)


Glist = [5, 10, 20, 35, 55, 100]
Sim_visc = np.zeros((len(Glist), 4))
Sim_elas = np.zeros((len(Glist), 4))

for n, G in enumerate(Glist):
    Sim_visc[n] = r_fin(G=G, h0=fh0(G, 0.01))
    Sim_elas[n] = r_fin(G=G, h0=fh0(G, 0.1))


print(Sim_visc, Sim_elas)

plt.figure()
plt.subplot(1, 2, 1)
plt.scatter(Sim_visc[:, 0], Sim_visc[:, 2])
plt.title("El=0.003, régime visqueux")
plt.xlabel("(Ga * h0 / r0) ** (1 / 5)")
plt.ylabel("r_fin/r0 - 1")
plt.grid()

plt.subplot(1, 2, 2)
plt.scatter(Sim_elas[:, 1], Sim_elas[:, 2])
plt.title("El=0.3, régime élastique")
plt.xlabel("(h0 / (r0 * El)) ** (1 / 6)")
plt.ylabel("r_fin/r0 - 1")
plt.grid()
plt.show()
