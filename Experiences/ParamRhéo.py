import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# Chargement des données


file_path = "Experiences/Kaolin47-52%-27&28-01-26-Flow curves.xlsx"

df50 = pd.read_excel(file_path, sheet_name="K50%-pot2-FC-1", usecols="H,N", header=4)

df52 = pd.read_excel(file_path, sheet_name="K52%-pot2-FC-1", usecols="H,N", header=4)


# Détection du plateau (contrainte seuil)
idx50 = df50["[Pa].2"].idxmin()
gamma0_50 = df50.loc[idx50, "[1/s]"]
tau0_50 = df50.loc[idx50, "[Pa].2"]

idx52 = df52["[Pa].2"].idxmin()
gamma0_52 = df52.loc[idx52, "[1/s]"]
tau0_52 = df52.loc[idx52, "[Pa].2"]

# Masque : on ne garde que la zone non plafonnée
mask50 = df50["[Pa].2"] > tau0_50
mask52 = df52["[Pa].2"] > tau0_52

x50 = pd.Series(df50.loc[mask50, "[1/s]"]).to_numpy()
y50 = pd.Series(df50.loc[mask50, "[Pa].2"]).to_numpy()

x52 = pd.Series(df52.loc[mask52, "[1/s]"]).to_numpy()
y52 = pd.Series(df52.loc[mask52, "[Pa].2"]).to_numpy()


# Modèle (τ0 fixé)


def model(x, k, m, gamma0, tau0):
    return k * (x**m - gamma0**m) + tau0


# Ajustement pondéré


params50, cov50 = curve_fit(
    lambda x, k, m: model(x, k, m, gamma0_50, tau0_50),
    x50,
    y50,
    p0=[1, 0.5],
    sigma=y50,
    absolute_sigma=False,
    bounds=([0.0, 0.0], [np.inf, 3.0]),
)

k50, m50 = params50

params52, cov52 = curve_fit(
    lambda x, k, m: model(x, k, m, gamma0_52, tau0_52),
    x52,
    y52,
    p0=[1, 0.5],
    sigma=y52,
    absolute_sigma=False,
    bounds=([0.0, 0.0], [np.inf, 3.0]),
)

k52, m52 = params52


# Résultats


print("=== Résultats du fit ===")
print(f"50 % : k = {k50:.3g}, m = {m50:.3f}, tau0 = {tau0_50:.3f} Pa")
print(f"52 % : k = {k52:.3g}, m = {m52:.3f}, tau0 = {tau0_52:.3f} Pa")


# Tracés


x_plot = np.logspace(
    np.log10(min(x50.min(), x52.min())), np.log10(max(x50.max(), x52.max())), 300
)

plt.figure(figsize=(7, 5))

plt.scatter(df50["[1/s]"], df50["[Pa].2"], label="50 % données", s=15)
plt.scatter(df52["[1/s]"], df52["[Pa].2"], label="52 % données", s=15)

plt.plot(x_plot, model(x_plot, k50, m50, gamma0_50, tau0_50), label="Fit 50 %")
plt.plot(x_plot, model(x_plot, k52, m52, gamma0_52, tau0_52), label="Fit 52 %")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Taux de cisaillement [1/s]")
plt.ylabel("Contrainte [Pa]")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()
