import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.lines import Line2D

def modeleKV_solve_h(
    h0,
    r0,
    rho,
    k,
    Bn,
    a=1.0,
    M=0.0,
    G=50.0,
    sigma=7.2e-2,
    m=1.0,
    g=9.81,
    Di=0.2,
    t_final=10.0,
    glissement=False # J'ai fusionné les deux modèles avec un booléen pour simplifier
):
    t_eval = np.linspace(0, t_final, 1000)
    tau0 = Bn * rho * g * h0
    eta = Di * tau0 * (g / h0) ** 0.5

    # Système d'EDO sur H (Hauteur) et V (Vitesse verticale dH/dt)
    def kelvin_voigt_ode_h(t, y):
        H, V, Gamma = y

        # 1. Protection numérique et Géométrie
        H = max(H, 1e-9) # Empêche H d'atteindre 0 absolu
        R = r0 * np.sqrt(h0 / H) # Conservation du volume
        
        # 2. Conversion de la vitesse verticale V en vitesse radiale U équivalente
        # Relation issue de la conservation de la masse : U = -(R / 2H) * V
        U = -(R / (2 * H)) * V

        # 3. Calcul du cisaillement (gamma_p)
        # On garde tes formules originales, mais injectées avec le U calculé ci-dessus
        if glissement:
            gamma_p = a * U / R
        else:
            gamma_p = a * U * (R / r0) ** 2 / h0
            
        if np.isnan(gamma_p) or np.isinf(gamma_p):
            gamma_p = 0.0

        # 4. Calcul de tauT (Somme des contraintes)
        # Note: tauT contient (Résistance - Moteur). 
        # Si Moteur > Résistance, tauT est négatif.
        tauT = (
            (k + eta) * np.abs(gamma_p) ** m * np.sign(gamma_p)
            + tau0
            + G * Gamma
            - rho * g * h0 * (r0 / R) ** 2 # Pression hydrostatique fluide
            - M * g / (np.pi * R)          # Attention aux dimensions ici (voir note plus bas*)
            - sigma * (r0 / R - 1) / r0
        )

        # Seuil de mise en mouvement (Bingham/Herschel-Bulkley)
        if abs(rho * g * h0 * (r0 / R) ** 2+M * g / (np.pi * R)) < tau0:
            tauT = tau0
            # Si on est sous le seuil, le fluide agit comme un solide -> forte décélération du mouvement
            # Pour éviter les instabilités numériques d'arrêt brusque, on peut laisser la physique 
            # freiner naturellement ou forcer V vers 0 si très faible.
        
        dHdt = V
        dGammadt = gamma_p
        
        # 5. Équation du mouvement : rho * h * d²h/dt² = tauT
        # Analyse du signe : 
        # Si Masse dominante -> tauT < 0.
        # On veut que la plaque descende (V diminue, dVdt < 0).
        # Donc dVdt doit être du même signe que tauT.
        # Facteur correctif : le terme inertiel complet pour le "Squeeze flow" est complexe,
        # mais ton modèle approximé rho*h*accel est acceptable ici.
        
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
    
    # Recalcul de R pour l'affichage
    R_sol = r0 * np.sqrt(h0 / H_sol)

    h_inf = H_sol[-1]

    # On retourne R_sol pour garder la cohérence avec ton script de plot
    return 1 - h_inf / h0, T_sol, R_sol, V_sol, Gamma_sol

if __name__ == "__main__":
    # Paramètres identiques à ton script
    mg = 50e-6
    r0 = 0.005
    ML = [0.020, 0.040, 0.1]
    rho = 1.27e3
    k = 90
    G = 907
    tau0 = 15
    sigma = 7.2e-2
    m = 0.4
    g = 9.81
    Di = 0
    h0 = mg / (rho * np.pi * r0**2)
    Bn = tau0 / (rho * g * h0)
    t_final = 100000.0 # Réduit pour tester rapidement, remets 1e7 si besoin

    RL = []
    RL_gliss = []
    HL = []     # Pour stocker les hauteurs directes
    T_list = []

    for n, M in enumerate(ML):
        # Sans glissement
        c, T, R, V, Gamma = modeleKV_solve_h(
            h0, r0, rho, k, Bn, M=M, G=G, sigma=sigma, m=m, g=g, Di=Di, t_final=t_final, glissement=False
        )
        RL.append(R)
        T_list.append(T) # On stocke le temps car solve_ivp peut varier légèrement les pas
        
        # Avec glissement
        c, T_g, Rg, Vg, Gammag = modeleKV_solve_h(
            h0, r0, rho, k, Bn, M=M, G=G, sigma=sigma, m=m, g=g, Di=Di, t_final=t_final, glissement=True
        )
        RL_gliss.append(Rg)

    # Tracés
    plt.figure(figsize=(10, 5))
    
    # Plot R(t)
    plt.subplot(1, 2, 1)
    for n in range(len(ML)):
        plt.plot(T_list[n], 1000 * RL[n], label=f"{1000 * ML[n]:.0f}g")
        plt.plot(T_list[n], 1000 * RL_gliss[n], "--", color=plt.gca().lines[-1].get_color())
    
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
        
        plt.plot(T_list[n], 1000 * H_calc, label=f"{1000 * ML[n]:.0f}g")
        plt.plot(T_list[n], 1000 * H_calc_gliss, "--", color=plt.gca().lines[-1].get_color())

    plt.title("Hauteur H(t)")
    plt.xlabel("t (s)")
    plt.ylabel("H (mm)")
    plt.grid(True, which="both", linestyle="--")
    
    plt.tight_layout()
    plt.show()