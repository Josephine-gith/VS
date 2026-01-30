import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# Chargement des données


file_path = "Experiences/Kaolin47-52%-27&28-01-26-Flow curves.xlsx"

df50 = pd.read_excel(file_path, sheet_name="K50%-pot2-FC-1", usecols="H,N", header=4)

df52 = pd.read_excel(file_path, sheet_name="K52%-pot2-FC-1", usecols="H,N", header=4)


# Détection du plateau (contrainte seuil)
tau0_50 = df50["[Pa].2"].min()
tau0_52 = df52["[Pa].2"].min()

# Masque : on ne garde que la zone non plafonnée
mask50 = df50["[Pa].2"] > tau0_50
mask52 = df52["[Pa].2"] > tau0_52

x50 = pd.Series(df50.loc[mask50, "[1/s]"]).to_numpy()
print(df50.loc[mask50, "[1/s]"], x50)
y50 = pd.Series(df50.loc[mask50, "[Pa].2"]).to_numpy()

x52 = pd.Series(df52.loc[mask52, "[1/s]"]).to_numpy()
y52 = pd.Series(df52.loc[mask52, "[Pa].2"]).to_numpy()


# Modèle (τ0 fixé)


def model(x, k, m, tau0):
    return k * x**m + tau0


# Ajustement pondéré


params50, cov50 = curve_fit(
    model,
    x50,
    y50,
    p0=[1, 0.5, tau0_50],
    sigma=y50,
    absolute_sigma=False,
    bounds=([0, 0, tau0_50 * 0.9], [np.inf, 3, tau0_50 * 1.1]),
)

k50, m50, tau0_fit50 = params50

params52, cov52 = curve_fit(
    model,
    x52,
    y52,
    p0=[1, 0.5, tau0_52],
    sigma=y52,
    absolute_sigma=False,
    bounds=([0, 0, tau0_52 * 0.9], [np.inf, 3, tau0_52 * 1.1]),
)

k52, m52, tau0_fit52 = params52


# Résultats


print("=== Résultats du fit ===")
print(f"50 % : k = {k50:.3g}, m = {m50:.3f}, tau0 = {tau0_fit50:.3f} Pa")
print(f"52 % : k = {k52:.3g}, m = {m52:.3f}, tau0 = {tau0_fit52:.3f} Pa")


# Tracés


x_plot = np.logspace(
    np.log10(min(x50.min(), x52.min())), np.log10(max(x50.max(), x52.max())), 300
)

plt.figure(figsize=(7, 5))

plt.scatter(df50["[1/s]"], df50["[Pa].2"], label="50 % données", s=15)
plt.scatter(df52["[1/s]"], df52["[Pa].2"], label="52 % données", s=15)

plt.plot(x_plot, model(x_plot, k50, m50, tau0_fit50), label="Fit 50 %")
plt.plot(x_plot, model(x_plot, k52, m52, tau0_fit52), label="Fit 52 %")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Taux de cisaillement [1/s]")
plt.ylabel("Contrainte [Pa]")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()
