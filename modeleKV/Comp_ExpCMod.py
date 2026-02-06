import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from modeleKV_solve_GlissAdh import modeleKV_solve, modeleKV_solve_gliss

file_path = "Experiences/Mesures mayonnaise.xlsx"
df = pd.read_excel(file_path, sheet_name="Feuil1")

# Pour n'avoir que la série 2 de mayonnaise (30min d'étalement) :
# df = df[df["m [kg]"].isin([0.000055, 0.000029967])]


df["r0 [m]"] = df["D_0 [mm]"] * 1e-3 / 2
df["h0 [m]"] = df["m [kg]"] / (df["rho [kg/m3]"] * np.pi * df["r0 [m]"] ** 2)
df["Bn"] = df["tau0"] / (df["rho [kg/m3]"] * df["g"] * df["h0 [m]"])

a_ng, a_g = 1, 1
t_final = 1000000000000


df["1-h_inf/h0"] = df.apply(
    lambda r: modeleKV_solve(
        r["h0 [m]"],
        r["r0 [m]"],
        r["rho [kg/m3]"],
        r["k"],
        r["Bn"],
        G=r["G"],
        m=r["m"],
        M=r["M [kg]"] + 5e-3,
        a=a_ng,
        Di=0.1,
        t_final=t_final,
    )[0],
    axis=1,
)


df["1-h_inf/h0 gliss"] = df.apply(
    lambda r: modeleKV_solve_gliss(
        r["h0 [m]"],
        r["r0 [m]"],
        r["rho [kg/m3]"],
        r["k"],
        r["Bn"],
        G=r["G"],
        m=r["m"],
        M=r["M [kg]"] + 5e-3,
        a=a_g,
        Di=0.1,
        t_final=t_final,
    )[0],
    axis=1,
)

df["D_max KV"] = df["D_0 [mm]"] / (1 - df["1-h_inf/h0"]) ** (1 / 2)
df["D_max KV gliss"] = df["D_0 [mm]"] / (1 - df["1-h_inf/h0 gliss"]) ** (1 / 2)

X1 = np.linspace(min(df["D_max KV"]), max(df["D_max KV"]))
x, y = df["D_max KV"], df["D_max [mm]"]
xg, yg = df["D_max KV gliss"], df["D_max [mm]"]

a, b = np.polyfit(x, y, 1)
ag, bg = np.polyfit(xg, yg, 1)

plt.plot(X1, X1, "--", label="x=y", color="red")

plt.scatter(x, y, label=f"non-glissement, a={a_ng}")
plt.plot(X1, a * X1 + b, label=f"Réglin : y={a:.3g}x+{b:.3g}")

plt.scatter(xg, yg, color="green", label=f"glissement, a={a_g}")
plt.plot(X1, ag * X1 + bg, label=f"Réglin : y={ag:.3g}x+{bg:.3g}", color="green")

plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.xlabel("D max modèle [mm]")
plt.ylabel("D max expérimental [mm]")
plt.legend()
plt.title(file_path)
plt.show()
