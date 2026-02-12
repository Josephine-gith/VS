import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from modele import modeleKV_solve_h

# Import et mise en forme des données expérimentales
file_path = "Experiences/Mesures temporelles.xlsx"
df0 = pd.read_excel(file_path, sheet_name="Feuil1")

dfC = df0[df0["matériau"] == "colle"].drop("matériau", axis="columns")
dfM = df0[df0["matériau"] == "mayonnaise"].drop("matériau", axis="columns")

dfC_init = dfC.iloc[[0]]
dfM_init = dfM.iloc[[0]]

# Données initiales pour alimenter le modèle
r0C = dfC_init["D [mm]"].values[0] * 1e-3 / 2
h0C = dfC_init["m [kg]"].values[0] / (
    dfC_init["rho [kg/m3]"].values[0] * np.pi * r0C**2
)
BnC = dfC_init["tau0"].values[0] / (
    dfC_init["rho [kg/m3]"].values[0] * dfC_init["g"].values[0] * h0C
)

r0M = dfM_init["D [mm]"].values[0] * 1e-3 / 2
h0M = dfM_init["m [kg]"].values[0] / (
    dfM_init["rho [kg/m3]"].values[0] * np.pi * r0M**2
)
BnM = dfM_init["tau0"].values[0] / (
    dfM_init["rho [kg/m3]"].values[0] * dfM_init["g"].values[0] * h0M
)

a_ngC, a_gC = 6,5
a_ngM, a_gM = 5,5

t_final = dfC["t [min]"].iloc[len(dfC.index) - 1] * 60

# Calcul du modèle pour la colle non glissante
c, TC, RC, U, Gamma = modeleKV_solve_h(
    h0C,
    r0C,
    dfC_init["rho [kg/m3]"].values[0],
    dfC_init["k"].values[0],
    BnC,
    G=dfC_init["G"].values[0],
    m=dfC_init["m"].values[0],
    M=dfC_init["M [kg]"].values[0] + 5e-3,
    a=a_ngC,
    Di=0.1,
    t_final=t_final,
)

"""
# Calcul du modèle pour la colle glissante
c, TCg, RC_gliss, U, Gamma = modeleKV_solve_h(
    h0C,
    r0C,
    dfC_init["rho [kg/m3]"].values[0],
    dfC_init["k"].values[0],
    BnC,
    G=dfC_init["G"].values[0],
    m=dfC_init["m"].values[0],
    M=dfC_init["M [kg]"].values[0] + 5e-3,
    a=a_gC,
    Di=0.1,
    t_final=t_final,
    glissement=True,
)

# Calcul du modèle pour la mayonnaise non glissante
c, TM, RM, U, Gamma = modeleKV_solve_h(
    h0M,
    r0M,
    dfM_init["rho [kg/m3]"].values[0],
    dfM_init["k"].values[0],
    BnM,
    G=dfM_init["G"].values[0],
    m=dfM_init["m"].values[0],
    M=dfM_init["M [kg]"].values[0] + 5e-3,
    a=a_ngM,
    Di=0.1,
    t_final=t_final,
)"""

# Calcul du modèle pour la mayonnaise glissante
c, TMg, RM_gliss, U, Gamma = modeleKV_solve_h(
    h0M,
    r0M,
    dfM_init["rho [kg/m3]"].values[0],
    dfM_init["k"].values[0],
    BnM,
    G=dfM_init["G"].values[0],
    m=dfM_init["m"].values[0],
    M=dfM_init["M [kg]"].values[0] + 5e-3,
    a=a_gM,
    Di=0.1,
    t_final=t_final,
    glissement=True,
)


# Tracés

plt.scatter(dfC["t [min]"]/60, dfC["D [mm]"], label="Expérience colle", color="red")
plt.plot(TC / 3600, 2e3 * RC, label="Modèle colle", color="red")
# plt.plot(TCg / 60, 2e3 * RC_gliss, "--", label="Modèle glissant colle", color="red")

plt.scatter(dfM["t [min]"]/60, dfM["D [mm]"], label="Expérience mayo", color="blue")
# plt.plot(TM / 60, 2e3 * RM, label="Modèle non-glissant mayo", color="blue")
plt.plot(TMg / 3600, 2e3 * RM_gliss, "--", label="Modèle glissant mayo", color="blue")


plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.xlabel("t [h]")
plt.ylabel("D [mm]")
plt.title("Etalement de la colle et de la mayonnaise")
plt.legend()
plt.show()
