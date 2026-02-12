import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from modele import modeleKV_solve_h

# Import et mise en forme des données expérimentales

# pour colle uniquement, avec t_final = 1h30
file_path = "Experiences/Mesures brutes colle 2.xlsx"

# ou bien, pour colle et mayonnaise, avec t_final pas bien connu :
# file_path = "Experiences/Toutes Mesures.xlsx"

df = pd.read_excel(file_path, sheet_name="Feuil1")

df["r0 [m]"] = df["D_0 [mm]"] * 1e-3 / 2
df["h0 [m]"] = df["m [kg]"] / (df["rho [kg/m3]"] * np.pi * df["r0 [m]"] ** 2)
df["Bn"] = df["tau0"] / (df["rho [kg/m3]"] * df["g"] * df["h0 [m]"])

a = 5  # Choix du préfacteur du taux de déformation

# Calcul du modèle pour chaque mesure expérimentale
# càd pour chaque ligne du tableau

df["1-h_inf/h0"] = df.apply(
    lambda r: modeleKV_solve_h(
        r["h0 [m]"],
        r["r0 [m]"],
        r["rho [kg/m3]"],
        r["k"],
        r["Bn"],
        G=r["G"],
        m=r["m"],
        M=r["M [kg]"],
        a=a,
        Di=0.1,
        t_final=r["t_final [s]"],
        glissement=(r["matériau"] == "mayonnaise"),
    )[0],
    axis=1,
)

df["D_max KV"] = df["D_0 [mm]"] / (1 - df["1-h_inf/h0"]) ** (1 / 2)


# Tracés

X1 = np.linspace(min(df["D_max KV"]), max(df["D_max KV"]))
plt.plot(X1, X1, "--", label="x=y", color="red")

# Régression linéaire
# Non-glissement
x, y = (
    df[df["matériau"] == "colle"]["D_max KV"],
    df[df["matériau"] == "colle"]["D_max [mm]"],
)
a, b = np.polyfit(x, y, 1)

plt.scatter(x, y, label="colle")
plt.plot(X1, a * X1 + b, label=f"Réglin : y={a:.3g}x+{b:.3g}")

# Glissement
"""
xg, yg = (
    df[df["matériau"] == "mayonnaise"]["D_max KV"],
    df[df["matériau"] == "mayonnaise"]["D_max [mm]"],
)
ag, bg = np.polyfit(xg, yg, 1)
plt.scatter(xg, yg, color="green", label="mayonnaise")
plt.plot(X1, ag * X1 + bg, label=f"Réglin : y={ag:.3g}x+{bg:.3g}", color="green")
"""

plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.xlabel("D max modèle [mm]")
plt.ylabel("D max expérimental [mm]")
plt.title("Dmax expérimental en fonction du Dmax modèle, pour la colle")
print(f"Pour un temps final = {df['t_final [s]'].values[0]:.2g} s")
plt.legend()
plt.show()
