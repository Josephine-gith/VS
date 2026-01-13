import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def modeleKV_solve(
    h0,
    r0,
    rho,
    k,
    Bn,
    a=1,
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

        gamma_p = a * U * (R / r0) ** 2 / h0

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

    c, T, R, U, Gamma = modeleKV_solve(h0, r0, rho, k, Bn)

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
