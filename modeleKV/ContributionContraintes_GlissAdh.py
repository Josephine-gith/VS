import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
from matplotlib.lines import Line2D


def modeleKV_solve_g(
    h0,
    r0,
    rho,
    k,
    Bn,
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
    tau0 = Bn * rho * g * h0
    eta = Di * tau0 * (g / h0) ** 0.5

    # Système d'EDO
    def kelvin_voigt_ode(t, y):
        R, U, Gamma = y

        # Protection numérique
        R = max(R, 1e-6)

        gamma_p = a * U / R
        if np.isnan(gamma_p) or np.isinf(gamma_p):
            gamma_p = 0.0

        tauT = (
            (k + eta) * np.abs(gamma_p) ** m * np.sign(gamma_p)
            + tau0
            + G * Gamma
            - rho * g * h0 * (r0 / R) ** 2
            - M * g / (np.pi * R)
            + sigma / r0 * (r0 / R - 1)
        )

        dRdt = U
        dGammadt = gamma_p
        dUdt = -((R / r0) ** 2) / (rho * h0) * tauT

        return [dRdt, dUdt, dGammadt]

    def calcul_contraintes(T, R, U, Gamma, params):
        h0, r0, rho, k, Bn, a, M, G, sigma, m, g, Di = params

        tau0 = Bn * rho * g * h0
        eta = Di * tau0 * (g / h0) ** 0.5

        Contraintes = []

        for Rk, Uk, Gammak in zip(R, U, Gamma):
            gamma_p = a * Uk / Rk

            tauVisc = k * abs(gamma_p) ** m * np.sign(gamma_p)
            # tauPlas = tau0
            tauElas = G * Gammak
            tauPoids = -rho * g * h0 * (r0 / Rk) ** 2
            tauComp = -M * g / (np.pi * Rk)
            tauCapi = -sigma / r0 * (r0 / Rk - 1)
            tauDi = eta * gamma_p

            tauT = tauVisc + tau0 + tauElas + tauPoids + tauComp + tauCapi + tauDi

            Contraintes.append(
                np.array((tauVisc, tau0, tauElas, tauCapi, -tauT))
                / abs(tauComp + tauPoids)
            )

        return np.array(Contraintes)

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

    params = (h0, r0, rho, k, Bn, 1, M, G, sigma, m, g, Di)
    C = calcul_contraintes(T, R, U, Gamma, params)

    return T, R, C


def modeleKV_solve(
    h0,
    r0,
    rho,
    k,
    Bn,
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
            - sigma * (r0 / R - 1) / r0
        )

        dRdt = U
        dGammadt = gamma_p
        dUdt = -((R / r0) ** 2) / (rho * h0) * tauT

        return [dRdt, dUdt, dGammadt]

    def calcul_contraintes(T, R, U, Gamma, params):
        h0, r0, rho, k, Bn, a, M, G, sigma, m, g, Di = params

        tau0 = Bn * rho * g * h0
        eta = Di * tau0 * (g / h0) ** 0.5

        Contraintes = []

        for Rk, Uk, Gammak in zip(R, U, Gamma):
            gamma_p = a * Uk * (Rk / r0) ** 2 / h0

            tauVisc = k * abs(gamma_p) ** m * np.sign(gamma_p)
            tauElas = G * Gammak
            tauPoids = -rho * g * h0 * (r0 / Rk) ** 2
            tauComp = -M * g / (np.pi * Rk)
            tauCapi = sigma * (r0 / Rk - 1) / r0
            tauDi = eta * gamma_p

            tauT = tauVisc + tau0 + tauElas + tauPoids + tauComp + tauCapi + tauDi

            Contraintes.append(
                np.array((tauVisc, tau0, tauElas, tauCapi, -tauT))
                / abs(tauComp + tauPoids)
            )

        return np.array(Contraintes)

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

    params = (h0, r0, rho, k, Bn, 1, M, G, sigma, m, g, Di)
    C = calcul_contraintes(T, R, U, Gamma, params)

    return T, R, C


if __name__ == "__main__":
    # Paramètres

    h0 = 0.005
    r0 = 0.0025
    M = 0.01
    rho = 1.28e3
    k = 90
    G = 100
    tau0 = 15
    sigma = 7.2e-2
    m = 0.4
    g = 9.81

    Di = 0
    Bn = tau0 / (rho * g * h0)

    t_final = 20

    T, R, C = modeleKV_solve(
        h0, r0, rho, k, Bn, M=M, G=G, sigma=sigma, m=m, g=g, Di=Di, t_final=t_final
    )
    Tg, Rg, Cg = modeleKV_solve_g(
        h0, r0, rho, k, Bn, M=M, G=G, sigma=sigma, m=m, g=g, Di=Di, t_final=t_final
    )

    df_Contraintes = pd.DataFrame(
        C,
        columns=[
            "Visqueuse",
            "Plastique",
            "Elastique",
            "Capillaire",
            "Inertielle",
        ],
    )
    df_Contraintes_g = pd.DataFrame(
        Cg,
        columns=[
            "Visqueuse",
            "Plastique",
            "Elastique",
            "Capillaire",
            "Inertielle",
        ],
    )

    # Tracés

    for col in df_Contraintes.columns:
        # Courbe sans glissement (trait plein)
        (line,) = plt.plot(T, df_Contraintes[col], label=col)
        color = line.get_color()

        # Courbe avec glissement (pointillé, même couleur)
        plt.plot(T, df_Contraintes_g[col], "--", color=color)
    leg1 = plt.legend(title="Type de contrainte", loc="upper right")
    style_legend = [
        Line2D([0], [0], color="black", linestyle="-", label="Sans glissement"),
        Line2D([0], [0], color="black", linestyle="--", label="Avec glissement"),
    ]

    leg2 = plt.legend(handles=style_legend, title="Modèle", loc="right")

    plt.gca().add_artist(leg1)

    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.xlabel("t (en s)")
    plt.ylabel("contrainte / contrainte de compression+poids")

    # plt.tight_layout()
    plt.title("Contribution de chaque contrainte")
    plt.show()
