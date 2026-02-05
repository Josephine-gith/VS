import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# Chargement des données

file_path_mayo = "Experiences/Mayo-03-02-26.xlsx"

dfm = pd.read_excel(
    file_path_mayo, sheet_name="03-02-26-Flow-curve-11", usecols="H,N", header=4
)

idxmin = dfm["[Pa].2"].idxmin()
gamma0 = dfm.loc[idxmin, "[1/s]"]
tau0 = dfm.loc[idxmin, "[Pa].2"]

xm = dfm["[1/s]"].to_numpy()
ym = dfm["[Pa].2"].to_numpy()


# Modèle (τ0 fixé)

def model(x, k, m, gamma0, tau0):
    return k * (x**m - gamma0**m) + tau0



# Ajustement pondéré

paramsm, covm = curve_fit(
    lambda x, k, m: model(x, k, m, gamma0, tau0),
    xm,
    ym,
    p0=[1, 0.5],
    sigma=ym,
    absolute_sigma=False,
    bounds=([0.0, 0.0], [np.inf, 3.0]),
)

km, mm = paramsm


# Résultats


print("=== Résultats du fit ===")
print(f"mayo : k = {km:.3g}, m = {mm:.3f}, tau0 = {tau0:.3f} Pa")


# Tracés

x_plotm = np.logspace(np.log10(xm.min()), np.log10(xm.max()), 300)

plt.figure(figsize=(7, 5))

plt.scatter(dfm["[1/s]"], dfm["[Pa].2"], label="Mayo données", s=15)


plt.plot(x_plotm, model(x_plotm, km, mm, gamma0, tau0), label="Fit Mayo")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Taux de cisaillement [1/s]")
plt.ylabel("Contrainte [Pa]")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()
