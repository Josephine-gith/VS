import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from modeleKV_solve_GlissAdh import modeleKV_solve, modeleKV_solve_gliss


# Paramètres

t_final = 10.0  # N*dt = 10s

h0 = 0.2
r0 = 0.1
M = 0
rho = 1e3
k = 20
G = 0
sigma = 0
g = 9.81
Di = 0.2
Tau0 = np.logspace(-4, 1.9, 50)
m = 1

Bn = Tau0 / (rho * g * h0)
xBn = (h0 / (Bn * r0)) ** (1 / 3)
Ga = rho * g * h0 / (rho * g * h0 - Tau0 - G * r0 / h0)


rows_g = []
xGa_g = (Ga * (h0 / r0) ** (1 - m)) ** (1 / (3 - m))
rows = []
xGa = xGa = (Ga * h0 / r0) ** (1 / (2 * m + 3))


for i, tau0 in enumerate(Tau0):
    T, R, U, Gamma = modeleKV_solve(h0, r0, rho, k, tau0, m=m, g=g, Di=Di)
    T, Rg, Ug, Gammag = modeleKV_solve_gliss(h0, r0, rho, k, tau0, m=m, g=g, Di=Di)

    uc_g = r0 * ((rho * g * h0 - tau0 - G * r0 / h0) / k) ** (1 / m)
    ucr_g = h0 * (
        (rho * g * h0) ** (2 - m / 3)
        / (k * tau0 ** (1 - m / 3) * (h0 / r0) ** (2 * m / 3))
    ) ** (1 / m)

    uc = h0 * ((rho * g * h0 - tau0 - G * r0 / h0) / k) ** (1 / m)
    ucr = (
        tau0 ** (2 / 3 + 1 / m)
        * h0
        / (k ** (1 / m) * (rho * g * h0**2 / r0) ** (2 / 3))
    )

    rows.append(
        {
            "T": T.tolist(),
            "r_inf": R[-1],
            "g": g,
            "rho": rho,
            "h0": h0,
            "r0": r0,
            "k": k,
            "tau0": tau0,
            "m": m,
            "h0/r0": h0 / r0,
            "r_inf/r0": R[-1] / r0,
            "h_inf": h0 * (r0 / R[-1]) ** 2,
            "h_inf/h0": (r0 / R[-1]) ** 2,
            "Uc": uc,
            "Ucr": ucr,
            "Uc/Ucr": uc / ucr,
            "Ga": Ga[i],
            "Bn": Bn[i],
            "scaling visqueux": xGa[i],
            "(1/Bn*h0/r0)**1/3": xBn[i],
        }
    )

    rows_g.append(
        {
            "T": T.tolist(),
            "r_inf": Rg[-1],
            "g": g,
            "rho": rho,
            "h0": h0,
            "r0": r0,
            "k": k,
            "tau0": tau0,
            "m": m,
            "h0/r0": h0 / r0,
            "r_inf/r0": Rg[-1] / r0,
            "h_inf": h0 * (r0 / Rg[-1]) ** 2,
            "h_inf/h0": (r0 / Rg[-1]) ** 2,
            "Uc": uc_g,
            "Ucr": ucr_g,
            "Uc/Ucr": uc_g / ucr_g,
            "Ga": Ga[i],
            "Bn": Bn[i],
            "scaling visqueux": xGa_g[i],
            "(1/Bn*h0/r0)**1/3": xBn[i],
        }
    )

df1 = pd.DataFrame(rows)
df_g = pd.DataFrame(rows_g)

file_name = "Simulations.xlsx"
df1.to_excel(file_name)

# Tracés
plt.figure()

X1 = df1["(1/Bn*h0/r0)**1/3"] / df1["scaling visqueux"]
Y1 = df1["r_inf/r0"] / df1["scaling visqueux"]

Xg = df_g["(1/Bn*h0/r0)**1/3"] / df_g["scaling visqueux"]
Yg = df_g["r_inf/r0"] / df_g["scaling visqueux"]

plt.plot(X1, Y1, label="Adhérence")
plt.plot(Xg, Yg, label="Glissement")
plt.ylabel("r_inf/r0 / scaling visqueux")
plt.xlabel("nombre d'effondrement")
plt.xscale("log")
plt.yscale("log")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()

plt.tight_layout()
plt.show()
