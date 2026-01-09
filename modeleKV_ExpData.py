import matplotlib.pyplot as plt

from ExpData_Df import df_exp
from modeleKV_solve import modeleKV_solve

df_final = df_exp.copy()

df_final["1 - h_inf/h0 KV"] = df_exp.apply(
    lambda r: modeleKV_solve(
        r["h0 [m]"], r["r0 [m]"], r["r [kg/m³]"], r["k [Pa s]"], r["Bn"], G=300
    )[0],
    axis=1,
)

df_final["err"] = df_final["1 - h_inf/h0 KV"] / df_final["1 - h_inf/h0"] - 1

# print(df_final.head())

plt.scatter(df_final["r0 [m]"] / df_final["h0 [m]"], df_final["err"])
plt.show()
