import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
from matplotlib.lines import Line2D


def modeleKV_solve(
    h0,
    r0,
    rho,
    k,
    Bn,
    a=1,
    M=0.0,
    G=0.0,
    m=1.0,
    g=9.81,
    Di=0.2,
    t_final=10.0,
    glissement=False,
):
    t_eval = np.linspace(0, t_final, 1000)
    tau0 = Bn * rho * g * h0
    eta = Di * tau0 * (g / h0) ** 0.5

    if glissement:
        kapp = 3 ** ((m + 1) / 2) * k
        tau0app = np.sqrt(3) * tau0
    else:
        kapp = k
        tau0app = tau0

    # Système d'EDO sur H (Hauteur) et V (Vitesse verticale dH/dt)
    def kelvin_voigt_ode_h(t, y):
        H, V, Gamma = y

        # 1. Protection numérique et Géométrie
        H = max(H, 1e-9)
        R = r0 * np.sqrt(h0 / H)

        # 2. Conversion de la vitesse verticale V en vitesse radiale U équivalente
        # Relation issue de la conservation de la masse : U = -(R / 2H) * V
        U = -(R / (2 * H)) * V

        # 3. Calcul du cisaillement (gamma_p)
        if glissement:
            gamma_p = a * U / R
        else:
            gamma_p = a * U / H

        if np.isnan(gamma_p) or np.isinf(gamma_p):
            gamma_p = 0.0

        # 4. Calcul de tauT (Somme des contraintes)
        Geom_factor = (2.0 * R) / (3.0 * H)

        tauT = (
            (
                (kapp + eta) * np.abs(gamma_p) ** m * np.sign(gamma_p)
                + tau0app
                + G * Gamma
            )
            * Geom_factor
            - rho * g * H
            - M * g / (np.pi * R**2)
        )

        # Seuil de mise en mouvement (Bingham/Herschel-Bulkley)
        if (
            abs(G * Gamma * Geom_factor - rho * g * H - M * g / (np.pi * R**2))
            < tau0app * Geom_factor
        ):
            tauT = 0
            gamma_p = 0

        dHdt = V
        dGammadt = gamma_p

        # 5. Équation du mouvement : rho * h * d²h/dt² = tauT

        dVdt = tauT / (rho * H)

        return [dHdt, dVdt, dGammadt]

    def calcul_contraintes(T, H, V, Gamma, params):
        h0, r0, rho, k, Bn, a, M, G, m, g, Di = params

        tau0 = Bn * rho * g * h0
        eta = Di * tau0 * (g / h0) ** 0.5

        if glissement:
            kapp = 3 ** ((m + 1) / 2) * k
            tau0app = np.sqrt(3) * tau0
        else:
            kapp = k
            tau0app = tau0

        Contraintes = []

        for Hk, Vk, Gammak in zip(H, V, Gamma):
            Rk = r0 * np.sqrt(h0 / Hk)
            Uk = -(Rk / (2 * Hk)) * Vk
            Geom_factor = (2.0 * Rk) / (3.0 * Hk)

            if glissement:
                gamma_p = a * Uk / Rk
            else:
                gamma_p = a * Uk * (Rk / r0) ** 2 / h0

            tauVisc = kapp * abs(gamma_p) ** m * np.sign(gamma_p) * Geom_factor
            tauPoids = -rho * g * Hk
            tauComp = -M * g / (np.pi * Rk**2)
            tauDi = eta * gamma_p
            tauPlas = tau0app * Geom_factor

            tauT = tauVisc + tauPlas + tauPoids + tauComp + tauDi

            Contraintes.append(
                np.array((tauVisc, tauPlas, -tauT)) / abs(tauComp + tauPoids)
            )

        return np.array(Contraintes)

    # Conditions initiales
    y0 = [r0, 0.0, 0.0]

    # Résolution
    sol = solve_ivp(
        kelvin_voigt_ode_h,
        t_span=(0, t_final),
        y0=y0,
        method="BDF",  # Méthode implicite stable
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-9,
    )

    H, V, Gamma = sol.y
    T = sol.t

    R = r0 * np.sqrt(h0 / H)

    params = (h0, r0, rho, k, Bn, 1, M, G, m, g, Di)
    C = calcul_contraintes(T, H, V, Gamma, params)

    return T, R, C


if __name__ == "__main__":
    # Paramètres

    mg = 50e-6
    r0 = 0.005

    M = 0.05

    rho = 1.27e3
    k = 90
    G = 0
    tau0 = 15
    m = 0.4
    g = 9.81

    Di = 0.1
    a, ag = 5, 5
    h0 = mg / (rho * np.pi * r0**2)
    Bn = tau0 / (rho * g * h0)
    t_final = 5e4

    T, R, C = modeleKV_solve(
        h0, r0, rho, k, Bn, M=M, G=G, m=m, g=g, Di=Di, t_final=t_final
    )
    Tg, Rg, Cg = modeleKV_solve(
        h0, r0, rho, k, Bn, M=M, G=G, m=m, g=g, Di=Di, t_final=t_final, glissement=True
    )

    df_Contraintes = pd.DataFrame(
        C,
        columns=[
            "Visqueuse",
            "Plastique",
            "Inertielle",
        ],
    )
    df_Contraintes_g = pd.DataFrame(
        Cg,
        columns=[
            "Visqueuse",
            "Plastique",
            "Inertielle",
        ],
    )

    # Tracés

    for col in df_Contraintes.columns:
        Tplot = T / 3600
        # Courbe sans glissement (trait plein)
        (line,) = plt.plot(Tplot, df_Contraintes[col], label=col)
        color = line.get_color()

        # Courbe avec glissement (pointillé, même couleur)
        plt.plot(Tplot, df_Contraintes_g[col], "--", color=color)
    leg1 = plt.legend(title="Type de contrainte", loc="upper right")
    style_legend = [
        Line2D([0], [0], color="black", linestyle="-", label="Sans glissement"),
        Line2D([0], [0], color="black", linestyle="--", label="Avec glissement"),
    ]

    leg2 = plt.legend(handles=style_legend, title="Modèle", loc="right")

    plt.gca().add_artist(leg1)

    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.xlabel("t (en h)")
    plt.ylabel("contrainte / contrainte de compression+poids")

    # plt.tight_layout()
    plt.title("Contribution de chaque contrainte")
    plt.show()
