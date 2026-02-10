import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from modeleKV2_solve_GlissAdh import modeleKV_solve_h

file_path = "Experiences/Mesures temporelles.xlsx"
df0 = pd.read_excel(file_path, sheet_name="Feuil1")

dfC = df0[df0["matériau"] == "colle"].drop("matériau", axis="columns")
dfM = df0[df0["matériau"] == "mayonnaise"].drop("matériau", axis="columns")


dfC_init = dfC.iloc[[0]]
dfM_init = dfM.iloc[[0]]

# dfM_init.at[dfM_init.index[0], "rho [kg/m3]"] = 1.28e3


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

a_ngC, a_gC = 0.5, 0.5
a_ngM, a_gM = 0.04, 0.04
# t_final = dfC["t [min]"].iloc[len(dfC.index) - 1] * 60
t_final = 1e5

c, T, RC, U, Gamma = modeleKV_solve_h(
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

RC_gliss = modeleKV_solve_h(
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
)[2]

RM = modeleKV_solve_h(
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
)[2]

RM_gliss = modeleKV_solve_h(
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
)[2]


plt.scatter(dfC["t [min]"], dfC["D [mm]"], label="Expérience colle", color="red")
plt.plot(T / 60, 2e3 * RC, label="Modèle colle", color="red")
plt.plot(T / 60, 2e3 * RC_gliss, "--", label="Modèle glissant colle", color="red")

plt.scatter(dfM["t [min]"], dfM["D [mm]"], label="Expérience mayo", color="blue")
plt.plot(T / 60, 2e3 * RM, label="Modèle mayo", color="blue")
plt.plot(T / 60, 2e3 * RM_gliss, "--", label="Modèle glissant mayo", color="blue")


plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.xlabel("t [min]")
plt.ylabel("D [mm]")
plt.legend()
plt.show()
