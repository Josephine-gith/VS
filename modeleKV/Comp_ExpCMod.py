import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from modeleKV_solve_GlissAdh import modeleKV_solve, modeleKV_solve_gliss

file_path = "Experiences/Mesures brutes.xlsx"
df = pd.read_excel(file_path, sheet_name="Feuil1")

df["h0 [m]"] = 4e6 * df["m [kg]"] / (df["rho [kg/m3]"] * np.pi * df["D_0 [mm]"] ** 2)
df["Bn"] = df["tau0"] / (df["rho [kg/m3]"] * df["g"] * df["h0 [m]"])


df["1-h_inf/h0"] = df.apply(
    lambda r: modeleKV_solve(
        r["h0 [m]"],
        r["D_0 [mm]"] * 1e-3 / 2,
        r["rho [kg/m3]"],
        r["k"],
        r["Bn"],
        G=r["G"],
        m=r["m"],
        M=r["M [kg]"],
        a=0.025,
    )[0],
    axis=1,
)

df["1-h_inf/h0 gliss"] = df.apply(
    lambda r: modeleKV_solve_gliss(
        r["h0 [m]"],
        r["D_0 [mm]"] * 1e-3 / 2,
        r["rho [kg/m3]"],
        r["k"],
        r["Bn"],
        G=r["G"],
        m=r["m"],
        M=r["M [kg]"],
        a=0.06,
    )[0],
    axis=1,
)

df["D_max KV"] = df["D_0 [mm]"] / (1 - df["1-h_inf/h0"]) ** (1 / 2)
df["D_max KV gliss"] = df["D_0 [mm]"] / (1 - df["1-h_inf/h0 gliss"]) ** (1 / 2)

X1 = np.linspace(min(df["D_max [mm]"]), max(df["D_max [mm]"]))
x, y = df["D_max KV"], df["D_max [mm]"]
xg, yg = df["D_max KV gliss"], df["D_max [mm]"]

a, b = np.polyfit(x, y, 1)
ag, bg = np.polyfit(xg, yg, 1)

plt.plot(X1, X1, "--", label="x=y", color="red")

# plt.scatter(x, y, label='non-glissement, a=0.025')
# plt.plot(X1, a*X1+b, label=f'Réglin : y={a:.3g}x+{b:.3g}')

plt.scatter(xg, yg, color="green", label="glissement, a=0.06")
plt.plot(X1, ag * X1 + bg, label=f"Réglin : y={ag:.3g}x+{bg:.3g}", color="green")

plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.xlabel("D max modèle [mm]")
plt.ylabel("D max expérimental [mm]")
plt.legend()
plt.show()
