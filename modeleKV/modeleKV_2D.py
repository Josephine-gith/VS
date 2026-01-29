import numpy as np
import matplotlib.pyplot as plt

## Modèle de Kelvin Voigt
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def modeleKV_solve(
    h0,
    r0,
    rho,
    k,
    tau0,
    a=1,
    M=0.0,
    G=50.0,
    sigma=5.0e-2,
    m=1.0,
    g=9.81,
    Di=0.2,
    t_final=10.0,
):
    t_eval = np.linspace(0, t_final, 1000)
    eta = Di * tau0 * (g / h0) ** 0.5

    # Système d'EDO
    def kelvin_voigt_ode(t, y):
        R, U, Gamma = y

        # Protection numérique
        R = max(R, 1e-6)

        gamma_p = a * U * R / (r0 * h0)
        if np.isnan(gamma_p) or np.isinf(gamma_p):
            gamma_p = 0.0

        tauT = (
            (k + eta) * np.abs(gamma_p) ** m * np.sign(gamma_p)
            + tau0
            + G * Gamma
            - rho * g * h0 * r0 / R
            - M * g / (2 * R)
        )

        dRdt = U
        dGammadt = gamma_p
        dUdt = -R / (rho * h0 * r0) * tauT

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

    return T, R, U, Gamma


if __name__ == "__main__":
    # Paramètres

    h0 = 0.15
    r0 = 0.02
    ML = [0]

    rho = 1e-3
    k = 1
    G = 0
    tau0 = 0.01
    sigma = 0
    m = 1
    g = 981

    Di = 0

    t_final = 4000.0  # N*dt = 10s

    RL = []
    HL = []
    UL = []
    T = []

    for n, M in enumerate(ML):
        T, R, U, Gamma = modeleKV_solve(
            h0, r0, rho, k, tau0, M=M, G=G, sigma=sigma, m=m, g=g, Di=Di, t_final=t_final
        )
        RL.append(R)
        HL.append(h0 * r0 / R)
        UL.append(U)

    # print(f"Rayon final : {R[-1]:.4f} m")

    # Tracés
    plt.figure()
    plt.subplot(1, 2, 1)
    for n in range(len(ML)):
        plt.plot(T, RL[n], label=f"M={1000 * ML[n]}g")
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
        plt.plot(T, HL[n], label=f"M={1000 * ML[n]}g")
    plt.title("h(t)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.xlabel("t (en s)")
    plt.ylabel("h (en mm)")
    plt.legend()

    # plt.tight_layout()
    plt.show()
