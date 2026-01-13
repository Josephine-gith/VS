import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd


def modeleKV_solve(
    h0,
    r0,
    rho,
    k,
    tau0,
    a=1,
    M=0,
    G=50,
    sigma=5e-2,
    m=1,
    g=9.81,
    Di=0.2,
    t_final=10.0,
):
    eta = Di * tau0 * (g / h0) ** 0.5
    rho_h0 = rho * h0
    inv_r0 = 1.0 / r0

    # Système d'EDO
    def kelvin_voigt_ode(t, y):
        R, U, Gamma = y

        Rr2 = (R * inv_r0) ** 2
        gamma_p = a * U * Rr2 / h0

        tauT = (
            (k + eta) * np.abs(gamma_p) ** m * np.sign(gamma_p)
            + tau0
            + G * Gamma
            - rho_h0 * g * (r0 / R) ** 2
            - M * g / (np.pi * R)
            + sigma * r0 * (r0 / R - 1)
        )

        dRdt = U
        dUdt = -((R / r0) ** 2) / (rho_h0) * tauT
        dGammadt = gamma_p

        return [dRdt, dUdt, dGammadt]

    # Conditions initiales
    y0 = [r0, 0.0, 0.0]

    # Résolution
    sol = solve_ivp(
        kelvin_voigt_ode,
        t_span=(0, t_final),
        y0=y0,
        method="BDF",  # Méthode implicite stable
        rtol=1e-5,
        atol=[1e-9, 1e-7, 1e-9],
    )

    R, U, Gamma = sol.y
    T = sol.t

    return T, R, U, Gamma


# Paramètres
Tau0 = np.logspace(-2, 3, 50)

h0 = 0.2
r0 = 0.1
M = 0.0

rho = 1e3
k = 20
G = 0
# tau0 = 30
sigma = 0
m = 1
g = 9.81

Di = 0.2
Bn = Tau0 / (rho * g * h0)
xBn = (h0 / (Bn * r0)) ** (1 / 3)
Ga = rho * g * h0 / (rho * g * h0 - Tau0 - G * r0 / h0)
xGa = (Ga * h0 / r0) ** (1 / (2 * m + 3))

t_final = 10.0  # N*dt = 10s
rows = []

for i, tau0 in enumerate(Tau0):
    T, R, U, Gamma = modeleKV_solve(h0, r0, rho, k, tau0)

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
            "(Ga*h0/r0)**(1/2m+3)": xGa[i],
            "(1/Bn*h0/r0)**1/3": xBn[i],
        }
    )


df = pd.DataFrame(rows)

file_name = "Simulations.xlsx"
df.to_excel(file_name)

# Tracés
plt.figure()

X = df["(1/Bn*h0/r0)**1/3"] / df["(Ga*h0/r0)**(1/2m+3)"]
Y = df["r_inf/r0"] / df["(Ga*h0/r0)**(1/2m+3)"]

plt.plot(X, Y)
plt.ylabel("r_inf/r0 / (Ga*h0/r0)**(1/2m+3)")
plt.xlabel("(1/Bn*h0/r0)**1/3  /  (Ga*h0/r0)**(1/2m+3)")
plt.xscale("log")
plt.yscale("log")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

plt.tight_layout()
plt.show()
