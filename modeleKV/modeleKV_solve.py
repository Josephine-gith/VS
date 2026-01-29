import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def modeleKV_solve(
    h0,
    r0,
    rho,
    k,
    Bn,
    a=1.0,
    M=0.0,
    G=50.0,
    sigma=5.0e-2,
    m=1.0,
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
        if np.isnan(gamma_p) or np.isinf(gamma_p):
            gamma_p = 0.0

        tauT = (
            (k + eta) * np.abs(gamma_p) ** m * np.sign(gamma_p)
            + tau0
            + G * Gamma
            - rho * g * h0 * (r0 / R) ** 2
            - M * g / (np.pi * R)
            - sigma / r0 * (r0 / R - 1)
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

    h0 = 0.005
    r0 = 0.0025
    ML = [0.2]

    rho = 1.28e3
    k = 90
    G = 907
    tau0 = 15
    sigma = 5e-2
    m = 0.4
    g = 9.81

    Di = 0
    Bn = tau0 / (rho * g * h0)

    t_final = 30.0  # N*dt = 10s

    RL = []
    HL = []
    UL = []
    T = []

    for n, M in enumerate(ML):
        c, T, R, U, Gamma = modeleKV_solve(
            h0, r0, rho, k, Bn, M=M, G=G, sigma=sigma, m=m, g=g, Di=Di, t_final=t_final
        )
        RL.append(R)
        HL.append(h0 * (r0 / R) ** 2)
        UL.append(U)

    # print(f"Rayon final : {R[-1]:.4f} m")

    # Tracés
    plt.figure()
    plt.subplot(1, 2, 1)
    for n in range(len(ML)):
        plt.plot(T, 1000 * RL[n], label=f"M={1000 * ML[n]}g")
    plt.title("r(t)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.xlabel("t (en s)")
    plt.ylabel("r (en mm)")
    plt.legend()

    """
    plt.subplot(1, 2, 2)
    for n in range(len(ML)):
        plt.plot(T, UL[n], label=f"M={1000 * ML[n]}g")
    plt.title("u(t)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.xlabel("t (en s)")
    plt.ylabel("u (en m/s)")
    plt.legend()
    """
    plt.subplot(1, 2, 2)
    for n in range(len(ML)):
        plt.plot(T, 1000 * HL[n], label=f"M={1000 * ML[n]}g")
    plt.title("h(t)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.xlabel("t (en s)")
    plt.ylabel("h (en mm)")
    plt.legend()

    # plt.tight_layout()
    plt.show()
