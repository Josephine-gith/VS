import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import colormaps

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
cmap = colormaps["Set1"]
colors_list = [cmap(i) for i in range(cmap.N)]
style = {
    m: (markers_list[i % len(markers_list)], colors_list[i % len(colors_list)])
    for i, m in enumerate(df["material"].unique())
}


# Graphe 1

fig, ax = plt.subplots(figsize=(10, 6))

for m, (mk, c) in style.items():
    d = df[df["material"] == m]
    ax.plot(d["Uc/Ucr"], d["h_inf/h0 KV"], mk, color=c, label=m)
    ax.plot(d["Uc/Ucr"], d["h_inf/h0"], mk, color=c, mfc="none", label="_nolegend_")

ax.set_xscale("log")
ax.set_xlabel("Uc/Ucr")
ax.set_ylabel("h_inf/h0")
ax.set_title("Hauteur finale normalisée exp et modèle KV")
ax.grid(True, which="both", linestyle="--", linewidth=0.5)


# Graphe 2
"""
ax2 = plt.subplot(122)
for m, (mk, c) in style.items():
    d = df[df["material"] == m]
    ax2.plot(d["Uc/Ucr"], d["err"], mk, color=c)

ax2.set_xscale("log")
ax2.set_xlabel("Uc/Ucr")
ax2.set_ylabel("(1 - h_inf/h0)th / (1 - h_inf/h0)exp - 1")
ax2.set_title("Erreur normalisée entre exp et modèle")
ax2.grid(True, which="both", linestyle="--", linewidth=0.5)
"""
"""
handles, labels = plt.get_legend_handles_labels()
handles2 = handles[::2] + handles[1::2]
labels2 = labels[::2] + labels[1::2]

fig.legend(handles2, labels2, loc="lower center", ncol=2, frameon=False)
"""

leg1 = ax.legend(
    title="Matériau",
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),
    frameon=True,
    fontsize=9,
    title_fontsize=10,
    ncols=1
)

legend_symbols = [
    Line2D([0], [0], marker="o", color="k", linestyle="None", label="Modèle KV"),
    Line2D(
        [0],
        [0],
        marker="o",
        color="k",
        linestyle="None",
        markerfacecolor="none",
        label="Expérience",
    ),
]

leg2 = ax.legend(handles=legend_symbols, loc="upper right", frameon=True)

ax.add_artist(leg1)

fig.subplots_adjust(bottom=0.45)

#plt.tight_layout()
plt.show()
