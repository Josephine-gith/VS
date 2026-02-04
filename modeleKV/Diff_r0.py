from modeleKV_solve_GlissAdh import modeleKV_solve, modeleKV_solve_gliss
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# Paramètres

mg = 50.0e-6
R0 = [0.003, 0.009]

M = 0.1

rho = 1.27e3

k = 90
G = 907
tau0 = 15
sigma = 7.2e-2
m = 0.4
g = 9.81

Di = 0.1


t_final = 2000000.0

RL = []
RL_gliss = []
T = []

for r0 in R0:
    h0 = mg / (rho * np.pi * r0**2)
    Bn = tau0 / (rho * g * h0)

    c, T, R, U, Gamma = modeleKV_solve(
        h0, r0, rho, k, Bn, M=M, G=G, sigma=sigma, m=m, g=g, Di=Di, t_final=t_final
    )
    c, T, Rg, Ug, Gamma = modeleKV_solve_gliss(
        h0, r0, rho, k, Bn, M=M, G=G, sigma=sigma, m=m, g=g, Di=Di, t_final=t_final
    )
    RL.append(R)
    RL_gliss.append(Rg)

# Tracés
plt.figure()
plt.subplot(1, 2, 1)
for n in range(len(R0)):
    (line,) = plt.plot(T, 1000 * RL[n], label=f"{1000 * R0[n]}mm")
    color = line.get_color()

    # Courbe avec glissement (pointillé, même couleur)
    plt.plot(T, 1000 * RL_gliss[n], "--", color=color)
leg1 = plt.legend(title="Hauteur h0", loc="lower right")
style_legend = [
    Line2D([0], [0], color="black", linestyle="-", label="Sans glissement"),
    Line2D([0], [0], color="black", linestyle="--", label="Avec glissement"),
]

leg2 = plt.legend(handles=style_legend, title="Modèle", loc="lower center")

plt.gca().add_artist(leg1)
plt.title("R(t)")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.xlabel("t (en s)")
plt.ylabel("r (en mm)")

plt.subplot(1, 2, 2)
for n in range(len(R0)):
    (line,) = plt.plot(
        T, 1000 * mg / (rho * np.pi * RL[n] ** 2), label=f"{1000 * R0[n]}mm"
    )
    color = line.get_color()

    # Courbe avec glissement (pointillé, même couleur)
    plt.plot(T, 1000 * mg / (rho * np.pi * RL_gliss[n] ** 2), "--", color=color)
leg1 = plt.legend(title="Hauteur h0", loc="upper right")
style_legend = [
    Line2D([0], [0], color="black", linestyle="-", label="Sans glissement"),
    Line2D([0], [0], color="black", linestyle="--", label="Avec glissement"),
]

leg2 = plt.legend(handles=style_legend, title="Modèle", loc="upper center")

plt.gca().add_artist(leg1)

plt.title("H(t)")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.xlabel("t (en s)")
plt.ylabel("h (en mm)")

# plt.tight_layout()
plt.show()
