import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def modeleKV_solve_h(
    h0,
    r0,
    rho,
    k,
    Bn,
    a=1.0,
    M=0.0,
    G=0.0,
    m=1.0,
    g=9.81,
    Di=0.2,
    t_final=1e5,
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

        # Seuil de mise en mouvement (Herschel-Bulkley)
        if (
            abs(
                G * Gamma * Geom_factor
                - rho * g * h0 * (r0 / R) ** 2
                - M * g / (np.pi * R**2)
            )
            < tau0app * Geom_factor
        ):
            tauT = 0
            gamma_p = 0

        dHdt = V
        dGammadt = gamma_p

        # 5. Équation du mouvement : rho * h * d²h/dt² = tauT

        dVdt = tauT / (rho * H)

        return [dHdt, dVdt, dGammadt]

    # Conditions initiales
    # H = h0, V = 0 (départ au repos), Gamma = 0
    y0 = [h0, 0.0, 0.0]

    # Résolution
    sol = solve_ivp(
        kelvin_voigt_ode_h,
        t_span=(0, t_final),
        y0=y0,
        method="BDF",
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-9,
    )

    H_sol, V_sol, Gamma_sol = sol.y
    T_sol = sol.t

    R_sol = r0 * np.sqrt(h0 / H_sol)

    h_inf = H_sol[-1]

    return 1 - h_inf / h0, T_sol, R_sol, V_sol, Gamma_sol


if __name__ == "__main__":
    
    # Paramètres d'expériences
    mg = 50e-6
    r0 = 0.005
    g = 9.81
    t_final = 2e6

    # Masses de compression
    ML = [0.020, 0.050, 0.1]

    # Propriétés du fluide
    rho = 1.27e3
    k = 90
    G = 0
    tau0 = 15
    m = 0.4

    Di = 0.1
    a, ag = 5, 5
    h0 = mg / (rho * np.pi * r0**2)
    Bn = tau0 / (rho * g * h0)

    # Stocker les rayons et temps pour les M
    RL = []
    RL_gliss = []
    TL = []
    TL_gliss = []

    for n, M in enumerate(ML):
        # Sans glissement
        c, T, R, V, Gamma = modeleKV_solve_h(
            h0,
            r0,
            rho,
            k,
            Bn,
            M=M,
            G=G,
            m=m,
            g=g,
            Di=Di,
            t_final=t_final,
            a=a,
        )
        RL.append(R)
        TL.append(T)

        # Avec glissement
        c, Tg, Rg, Vg, Gammag = modeleKV_solve_h(
            h0,
            r0,
            rho,
            k,
            Bn,
            M=M,
            G=G,
            m=m,
            g=g,
            Di=Di,
            t_final=t_final,
            a=ag,
            glissement=True,
        )
        RL_gliss.append(Rg)
        TL_gliss.append(Tg)

    # Tracés
    plt.figure(figsize=(10, 5))

    # Plot R(t)
    plt.subplot(1, 2, 1)
    for n in range(len(ML)):
        plt.plot(TL[n], 1000 * RL[n], label=f"{1000 * ML[n]:.0f}g")
        plt.plot(
            TL_gliss[n], 1000 * RL_gliss[n], "--", color=plt.gca().lines[-1].get_color()
        )

    plt.title("Rayon R(t)")
    plt.xlabel("t (s)")
    plt.ylabel("R (mm)")
    plt.grid(True, which="both", linestyle="--")
    plt.legend()

    # Plot H(t)
    plt.subplot(1, 2, 2)
    for n in range(len(ML)):
        # Recalcul de H depuis R pour vérifier la cohérence
        H_calc = h0 * (r0 / RL[n]) ** 2
        H_calc_gliss = h0 * (r0 / RL_gliss[n]) ** 2

        plt.plot(TL[n], 1000 * H_calc, label=f"{1000 * ML[n]:.0f}g")
        plt.plot(
            TL_gliss[n],
            1000 * H_calc_gliss,
            "--",
            color=plt.gca().lines[-1].get_color(),
        )

    plt.title("Hauteur H(t)")
    plt.xlabel("t (s)")
    plt.ylabel("H (mm)")
    plt.grid(True, which="both", linestyle="--")

    plt.tight_layout()
    plt.show()
