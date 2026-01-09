import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def modeleKV_solve(
    h0,
    r0,
    rho,
    k,
    Bn,
    M=0,
    G=50,
    sigma=5e-2,
    m=1,
    g=9.81,
    Di=0.2,
    t_final=10.0,
):
    t_eval = np.linspace(0, t_final, 1000)
    tau0 = Bn * rho * g * h0
    eta = Di * tau0 * (g / h0) ** 0.5

    # Système d'EDO
    def kelvin_voigt_ode(t, y):
        R, U, Gamma = y

        # Protection numérique
        R = max(R, 1e-6)

        gamma_p = U * (R / r0) ** 2 / h0

        tauT = (
            (k + eta) * np.abs(gamma_p) ** m * np.sign(gamma_p)
            + tau0
            + G * Gamma
            - rho * g * h0 * (r0 / R) ** 2
            - M * g / (np.pi * R)
            + sigma * r0 * (r0 / R - 1)
        )

        dRdt = U
        dGammadt = gamma_p
        dUdt = -((R / r0) ** 2) / (rho * h0) * tauT

        return [dRdt, dUdt, dGammadt]

    # Conditions initiales
    y0 = [r0, 0.0, 0.0]

    # Résolution
    sol = solve_ivp(
        kelvin_voigt_ode,
        t_span=(0, t_final),
        y0=y0,
        method="BDF",  # Méthode implicite stable
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-9,
    )

    R, U, Gamma = sol.y
    T = sol.t

    h_inf = h0 * (r0 / R[-1]) ** 2

    return 1 - h_inf / h0, T, R, U, Gamma


def modeleKV_Euler(
    h0, r0, rho, k, Bn, sigma=5e-2, m=1, g=9.81, G=50, M=0, Di=0, dt=1e-3, N=10**4
):
    tau0 = Bn * rho * g * h0
    eta = Di * tau0 * (g / h0) ** (1 / 2)
    dt_eff = dt / (1 + k / 100)

    # Initialisation
    R = np.zeros(N + 1)
    R[0] = r0
    U = np.zeros(N + 1)
    Gamma = np.zeros(N + 1)

    # Itération
    for i in range(N):
        R[i + 1] = dt_eff * U[i] + R[i]
        gamma_p = U[i] * (R[i] / r0) ** 2 / h0
        tauT = (
            (k + eta) * abs(gamma_p) ** m * np.sign(gamma_p)
            + tau0
            + G * Gamma[i]
            - rho * g * h0 * (r0 / R[i]) ** 2
            - M * g / (np.pi * R[i])
            + sigma * r0 * (r0 / R[i] - 1)
        )
        Gamma[i + 1] = dt_eff * gamma_p + Gamma[i]
        U[i + 1] = -dt_eff * (R[i] / r0) ** 2 / (rho * h0) * tauT + U[i]

    h_inf = h0 * (r0 / R[N]) ** 2

    return 1 - h_inf / h0


if __name__ == "__main__":
    # Paramètres

    h0 = 0.1
    r0 = 0.1
    M = 0.05

    rho = 1e3
    k = 40
    G = 50
    tau0 = 30
    sigma = 5e-2
    m = 1
    g = 9.81

    Di = 0.2
    Bn = tau0 / (rho * g * h0)

    t_final = 10.0  # N*dt = 10s

    a, T, R, U, Gamma = modeleKV_solve(h0, r0, rho, k, Bn)

    print(f"Rayon final : {R[-1]:.4f} m")

    # Tracés
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

    plt.tight_layout()
    plt.show()
