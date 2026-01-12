import numpy as np
import matplotlib.pyplot as plt

## Modèle de Kelvin Voigt


def r_fin(
    h0=0.1,
    r0=0.1,
    rho=1.0e3,
    k=40.0,
    tau0=30.0,
    G=50.0,
    Di=0.2,
    m=1,
    dt=1e-4,
    N=2 * 10**4,
):
    M = 0  # car dam break sans compression supplémentaire
    sigma = 0  # négligé

    g = 9.81  # m/s2
    eta = Di * tau0 * (g / h0) ** (1 / 2)

    # Initialisation
    R = np.zeros(N + 1)
    U = np.zeros(N + 1)
    Gamma = np.zeros(N + 1)
    Contraintes = np.zeros((N + 1, 7))

    R[0] = r0
    tauVisc = 0
    tauElas = 0
    tauPoids = -rho * g * h0
    tauComp = -M * g / (np.pi * r0)
    tauCapi = 0
    tauDi = 0
    tauT = tauVisc + tau0 + tauElas + tauPoids + tauComp + tauCapi + tauDi
    Contraintes[0, :] = np.array(
        (tauVisc, tau0, tauElas, -tauComp, tauCapi, tauDi, -tauT)
    ) / abs(tauPoids)

    # Itération
    for i in range(N):
        R[i + 1] = dt * U[i] + R[i]
        gamma_p = U[i] * (R[i] / r0) ** 2 / h0

        tauVisc = k * abs(gamma_p) ** m * np.sign(gamma_p)
        tauElas = G * Gamma[i]
        tauPoids = -rho * g * h0 * (r0 / R[i]) ** 2
        tauComp = -M * g / (np.pi * R[i])
        tauCapi = sigma * r0 * (r0 / R[i] - 1)
        tauDi = eta * gamma_p
        tauT = tauVisc + tau0 + tauElas + tauPoids + tauComp + tauCapi + tauDi
        Contraintes[i + 1, :] = np.array(
            (tauVisc, tau0, tauElas, -tauComp, tauCapi, tauDi, -tauT)
        ) / abs(tauPoids)

        Gamma[i + 1] = dt * gamma_p + Gamma[i]
        U[i + 1] = -dt * (R[i] / r0) ** 2 / (rho * h0) * tauT + U[i]

    T = dt * np.arange(N + 1)

    return T, Contraintes


# Simulation pour chaque régime

h0 = 0.25
Sim_visc = r_fin(h0=h0, k=50, tau0=1e-6, G=1e-6)
Sim_plas = r_fin(h0=h0, tau0=20, k=1e-6, G=1e-10)
Sim_elas = r_fin(h0=h0, G=300, tau0=1e-6, k=1e-8)
Sim_mixt = r_fin(h0=h0, G=50, tau0=30, k=40)


Contraintes_leg = [
    "Visqueuse",
    "Plastique (contrainte seuil)",
    "Elastique",
    "Compression",
    "Capillaire",
    "Dissipatif",
    "Total (= inertie)",
]

fig, axs = plt.subplots(2, 2)

plt.subplot(221)
plt.plot(Sim_visc[0], Sim_visc[1])
plt.title("Régime visqueux")
plt.grid()
plt.subplot(222)
plt.plot(Sim_plas[0], Sim_plas[1])
plt.title("Régime plastique")
plt.grid()
plt.subplot(223)
plt.plot(Sim_elas[0], Sim_elas[1])
plt.title("Régime élastique")
plt.grid()
plt.subplot(224)
plt.plot(Sim_mixt[0], Sim_mixt[1], label=Contraintes_leg)
plt.title("Régime mixte")
plt.grid()

plt.suptitle("Contribution de chaque contrainte, normalisée par le poids")
fig.supxlabel("Temps (en sec)")
fig.supylabel("Contrainte / tauPoids")
fig.legend()


plt.show()
