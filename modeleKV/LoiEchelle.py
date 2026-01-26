import numpy as np
import matplotlib.pyplot as plt

## Régimes visqueux, plastique et élastique

# Modèle de Kelvin Voigt


def r_fin(
    h0=0.1, r0=0.1, rho=1.0e3, k=40.0, tau0=15.0, G=1.0e2, Di=0.2, m=1, dt=1e-4, N=10**5
):
    M = 0  # car dam break sans compression supplémentaire
    sigma = 0  # négligé

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
            - sigma / r0 * (r0 / R[i] - 1)
        )
        Gamma[i + 1] = dt * gamma_p + Gamma[i]
        U[i + 1] = -dt * (R[i] / r0) ** 2 / (rho * h0) * tauT + U[i]

    # Sortie
    uc = h0 * (rho * g * h0 - tau0 - G * r0 / h0) / k
    Ga = rho * g * h0 / (k * uc / h0)
    x1 = (Ga * h0 / r0) ** (1 / (2 * m + 3))

    Bn = tau0 / (rho * g * h0)
    x2 = (h0 / (r0 * Bn)) ** (1 / 3)

    El = G * r0 / (rho * g * h0**2)
    x3 = (h0 / (r0 * El)) ** (1 / 6)

    y1 = R[-1] / r0 - 1
    y2 = max(R) / r0 - 1

    """
    # Trace R(t)
    T = dt * np.arange(N + 1)
    plt.plot(T, R)
    plt.title('R(t)')
    plt.grid()
    plt.show()
    """
    return x1, x2, x3, y1, y2, U[-1]


# Simulations pour chaque régime
H0 = np.array([0.05, 0.1, 0.25, 0.3, 0.50, 0.65, 0.8, 1.0])
Sim_visc = np.zeros((len(H0), 6))
Sim_plas = np.zeros((len(H0), 6))
Sim_elas = np.zeros((len(H0), 6))

for i, h0 in enumerate(H0):
    Sim_visc[i] = r_fin(h0=h0, k=50, tau0=1e-6, G=1e-6)
    Sim_plas[i] = r_fin(h0=h0, tau0=20, k=1e-6, G=1e-10)
    Sim_elas[i] = r_fin(h0=h0, G=300, tau0=1e-6, k=1e-8, N=10**5)


# Pour vérifier que U_fin est bien ~ 0
# print(Sim_visc[:, 5], Sim_plas[:, 5], Sim_elas[:, 5])

# Trace les rayons finaux (ou max) normalisés
plt.figure()
plt.subplot(1, 3, 1)
plt.scatter(Sim_visc[:, 0], Sim_visc[:, 3])
plt.title("Régime visqueux (Bn=1e-6 - El=1e-6)")
plt.xlabel("(Ga * h0 / r0) ** (1 / 5)")
plt.ylabel("r_fin/r0 - 1")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

plt.subplot(1, 3, 2)
plt.scatter(Sim_plas[:, 1], Sim_plas[:, 3])
plt.title("Régime plastique (k=1e-3 - El=1e-6)")
plt.xlabel("(h0 / (r0 * Bn)) ** (1 / 3)")
plt.ylabel("r_fin/r0 - 1")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

plt.subplot(1, 3, 3)
plt.scatter(Sim_plas[:, 2], Sim_plas[:, 4])
plt.title("Régime élastique (k=1e-3 - Bn=0.003)")
plt.xlabel("(h0 / (r0 * El)) ** (1 / 6)")
plt.ylabel("r_max/r0 - 1")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

plt.tight_layout()
plt.show()
