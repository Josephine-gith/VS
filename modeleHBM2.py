import numpy as np
import matplotlib.pyplot as plt

## Modèle de Hescher-Bukley + Maxwell

# -----------------------------
# Paramètres numériques
# -----------------------------
N = 2 * 10**5
dt = 1e-4

# -----------------------------
# Paramètres géométriques
# -----------------------------
h0 = 0.1
r0 = 0.1
M = 0.5

# -----------------------------
# Propriétés matériau
# -----------------------------
rho = 1e3
G = 20
k = 50
Lambda = k / G

tau0 = 5
Gamma0 = tau0 / G

sigma = 5e-2
eta = 50  # dissipation cinématique
epsR = 0.2 * r0  # régularisation géométrique

g = 9.81

# -----------------------------
# Initialisation
# -----------------------------
R = np.zeros(N + 1)
U = np.zeros(N + 1)
Gamma = np.zeros(N + 1)

R[0] = r0


# -----------------------------
# Fonctions auxiliaires
# -----------------------------
def safe_factor(R):
    return (R / r0) ** 2 / (1 + (R / r0) ** 2)


def gamma_plastic(Gamma):
    # Régularisation C¹ du seuil
    return (1 / Lambda) * Gamma / np.sqrt(Gamma**2 + Gamma0**2)


# -----------------------------
# Boucle temporelle
# -----------------------------
for i in range(N):
    # --- Géométrie (explicite)
    R[i + 1] = max(R[i] + dt * U[i], epsR)

    fac = safe_factor(R[i])

    # --- Forces externes
    Fext = (
        rho * g * h0 * (r0 / R[i]) ** 2
        + M * g / (np.pi * R[i])
        - sigma * r0 * (r0 / R[i] - 1)
    )

    # --- Schéma implicite couplé U / Gamma
    # U^{n+1} = U^n - a (G Γ^{n+1} - Fext) - d U^{n+1}
    # Γ^{n+1} = Γ^n + b U^{n+1} - dt Γ_pl(Γ^{n+1})

    a = dt * fac / (rho * h0)
    b = dt * fac / h0
    d = dt * eta / (rho * h0)

    # Newton sur Gamma
    Gamma_new = Gamma[i]
    for _ in range(5):
        U_new = (U[i] - a * (G * Gamma_new - Fext)) / (1 + d)
        f = Gamma_new - Gamma[i] - b * U_new + dt * gamma_plastic(Gamma_new)
        df = (
            1
            + dt / Lambda * (Gamma0**2 / (Gamma_new**2 + Gamma0**2) ** (3 / 2))
            + b * a * G / (1 + d)
        )
        Gamma_new -= f / df

    Gamma[i + 1] = Gamma_new
    U[i + 1] = (U[i] - a * (G * Gamma[i + 1] - Fext)) / (1 + d)

# -----------------------------
# Énergie
# -----------------------------
T = dt * np.arange(N + 1)
E = 0.5 * rho * h0 * U**2 + 0.5 * G * Gamma**2

# -----------------------------
# Affichage
# -----------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(T, R)
plt.title("r(t)")
plt.grid()

plt.subplot(1, 3, 2)
plt.plot(T, U)
plt.title("u(t)")
plt.grid()

plt.subplot(1, 3, 3)
plt.plot(T, Gamma)
plt.title("gamma(t)")
plt.grid()

plt.figure()
plt.plot(T, E)
plt.title("Énergie totale (décroissante)")
plt.grid()

plt.tight_layout()
plt.show()

print(f"Rayon final : {R[-1]:.4f} m")
