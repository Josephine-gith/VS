import matplotlib.pyplot as plt

from ExpData_Df import df_exp
from modeleKV_solve import modeleKV_solve


# Préparation des données

df_exp["h_inf/h0"] = 1 - df_exp["1 - h_inf/h0"]
df_exp["Uc/Ucr"] = (-1 + 1 / df_exp["Bn"]) ** (1 / df_exp["m"]) * (
    df_exp["h0 [m]"] / (df_exp["Bn"] * df_exp["r0 [m]"])
) ** (2 / 3)

df = df_exp.copy()
df["1 - h_inf/h0 KV"] = df.apply(
    lambda r: modeleKV_solve(
        r["h0 [m]"], r["r0 [m]"], r["r [kg/m³]"], r["k [Pa s]"], r["Bn"], a=60, G=0
    )[0],
    axis=1,
)
df["h_inf/h0 KV"] = 1 - df["1 - h_inf/h0 KV"]
df["err"] = df["1 - h_inf/h0 KV"] / df["1 - h_inf/h0"] - 1


# Styles par matériau

markers_list = ["o", "P", "X", "s", "D", "^", "v", "*"]
cmap = plt.cm.get_cmap("Set1")
colors_list = [cmap(i) for i in range(cmap.N)]
style = {
    m: (markers_list[i % len(markers_list)], colors_list[i % len(colors_list)])
    for i, m in enumerate(df["material"].unique())
}


# Graphe 1

fig = plt.figure(figsize=(12, 8))

ax1 = plt.subplot(121)
for m, (mk, c) in style.items():
    d = df[df["material"] == m]
    ax1.plot(d["Uc/Ucr"], d["h_inf/h0 KV"], mk, color=c, label=f"Mod {m}")
    ax1.plot(d["Uc/Ucr"], d["h_inf/h0"], mk, color=c, mfc="none", label=f"Exp {m}")

ax1.set_xscale("log")
ax1.set_xlabel("Uc/Ucr")
ax1.set_ylabel("h_inf/h0")
ax1.set_title("Hauteur finale normalisée exp et modèle KV")
ax1.grid(True, which="both", linestyle="--", linewidth=0.5)


# Graphe 2

ax2 = plt.subplot(122)
for m, (mk, c) in style.items():
    d = df[df["material"] == m]
    ax2.plot(d["Uc/Ucr"], d["err"], mk, color=c)

ax2.set_xscale("log")
ax2.set_xlabel("Uc/Ucr")
ax2.set_ylabel("(1 - h_inf/h0)th / (1 - h_inf/h0)exp - 1")
ax2.set_title("Erreur normalisée entre exp et modèle")
ax2.grid(True, which="both", linestyle="--", linewidth=0.5)

handles, labels = ax1.get_legend_handles_labels()
handles2 = handles[::2] + handles[1::2]
labels2 = labels[::2] + labels[1::2]

fig.legend(handles2, labels2, loc="lower center", ncol=2, frameon=False)

fig.subplots_adjust(bottom=0.35)
plt.show()
