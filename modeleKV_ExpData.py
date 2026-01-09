import numpy as np

from ExpData_Df import df_exp


def modeleKV(
    h0, r0, rho, k, Bn, sigma=5e-2, m=1, g=9.81, G=30, M=0, Di=1e-3, dt=1e-3, N=10**4
):
    tau0 = Bn * rho * g * h0
    eta = Di * tau0 * (g / h0) ** (1 / 2)

    # Initialisation
    R = np.zeros(N + 1)
    R[0] = r0
    U = np.zeros(N + 1)
    Gamma = np.zeros(N + 1)

    # Itération
    for i in range(N):
        R[i + 1] = dt * U[i] + R[i]
        gamma_p = U[i] * (R[i] / r0) ** 2 / h0
        tauT = (
            (k + eta) * abs(gamma_p) ** m * np.sign(gamma_p)
            + tau0
            + G * Gamma[i]
            - rho * g * h0 * (r0 / R[i]) ** 2
            - M * g / (np.pi * R[i])
            + sigma * r0 * (r0 / R[i] - 1)
        )
        Gamma[i + 1] = dt * gamma_p + Gamma[i]
        U[i + 1] = -dt * (R[i] / r0) ** 2 / (rho * h0) * tauT + U[i]

    h_inf = h0 * (r0 / R[N]) ** 2

    return 1 - h_inf / h0


df_final = df_exp.iloc[:10].copy()

df_final["1 - h_inf/h0 KV"] = df_exp.apply(
    lambda r: modeleKV(
        r["h0 [m]"], r["r0 [m]"], r["r [kg/m³]"], r["k [Pa s]"], r["Bn"]
    ),
    axis=1,
)

print(df_final.head())
