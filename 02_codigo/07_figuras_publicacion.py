"""
07_figuras_publicacion.py
Figuras de alta resolución para publicación Q1 — todo en inglés.

Entrada:  03_salidas/modelos/datos_procesados.pkl
Salidas:  03_salidas/figuras_pub/
          fig01_sem_path_diagram.png
          fig02_factor_loadings_heatmap.png
          fig03_construct_correlations.png
          fig04_raincloud_gender.png
          fig05_likert_diverging.png
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.gridspec as gridspec

sys.path.append(str(__import__("pathlib").Path(__file__).parent))
from config import (
    ARCHIVO_PROCESADO, FIGURAS_PUB_DIR, MODELOS_DIR,
    EA_COLS, FA_COLS, MV_COLS, EQ_COLS, BP_COLS,
    INDICES, NOMBRES_INDICES_EN, NOMBRES_INDICES_CORTOS_EN,
    COLORES_INDICES, LISTA_COLORES_INDICES,
    COLORES_GENERO_EN, GENERO_ES_EN,
    DPI_PUB, FIGSIZE_GRANDE, FIGSIZE_MEDIANA, FIGSIZE_CUADRADA,
    FS_TITULO, FS_EJE, FS_TICK, FS_LEYENDA,
    ESTILO_BASE, NOTA_MUESTRA_EN, NOTA_SIG_EN,
)

plt.style.use(ESTILO_BASE)
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        FS_TICK,
    "axes.titlesize":   FS_TITULO,
    "axes.labelsize":   FS_EJE,
    "xtick.labelsize":  FS_TICK,
    "ytick.labelsize":  FS_TICK,
    "legend.fontsize":  FS_LEYENDA,
    "figure.dpi":       DPI_PUB,
    "savefig.dpi":      DPI_PUB,
    "savefig.bbox":     "tight",
    "savefig.facecolor":"white",
})

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def imprimir_seccion(titulo):
    print(f"\n{'='*60}")
    print(f"  {titulo}")
    print(f"{'='*60}")


def guardar_figura(fig, nombre: str):
    ruta = FIGURAS_PUB_DIR / nombre
    fig.savefig(ruta, dpi=DPI_PUB, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  ✅ Guardado: {nombre}")


def cronbach_alpha(data: pd.DataFrame) -> float:
    n = data.shape[1]
    if n < 2:
        return np.nan
    vt = data.sum(axis=1).var(ddof=1)
    return (n / (n - 1)) * (1 - data.var(axis=0, ddof=1).sum() / vt) if vt > 0 else np.nan


# =============================================================================
# CARGA DE DATOS
# =============================================================================

imprimir_seccion("CARGANDO DATOS")
df = pd.read_pickle(ARCHIVO_PROCESADO)
N  = len(df)
print(f"  n = {N}")

# Calcular correlaciones ítem-total como proxy de cargas
likert_freq   = {"Nunca":1,"Pocas veces":2,"Algunas veces":3,
                 "Frecuentemente":4,"Siempre o casi siempre":5}
likert_acuerdo = {"Totalmente en desacuerdo":1,"En desacuerdo":2,
                  "Ni de acuerdo ni en desacuerdo":3,
                  "De acuerdo":4,"Totalmente de acuerdo":5}

escalas = {
    "Stress (EA)":     (EA_COLS, "_num",   likert_freq,    COLORES_INDICES["IDX_EA"]),
    "Fatigue (FA)":    (FA_COLS, "_num",   likert_freq,    COLORES_INDICES["IDX_FA"]),
    "Motivation (MV)":(MV_COLS, "_num",   likert_acuerdo, COLORES_INDICES["IDX_MV"]),
    "Equity (EQ)":     (EQ_COLS, "_final", likert_acuerdo, COLORES_INDICES["IDX_EQ"]),
    "Well-being (BP)": (BP_COLS, "_num",   likert_acuerdo, COLORES_INDICES["IDX_BP"]),
}

# Leer coeficientes SEM si existen
sem_coefs = {}
ruta_sem  = MODELOS_DIR / "mod_sem_output_completo.txt"
if ruta_sem.exists():
    print(f"  ✅ Output SEM encontrado")
else:
    print(f"  ⚠️  Output SEM no encontrado — fig01 usará valores de ejemplo")

# Leer cargas factoriales si existen
cargas_dict = {}
ruta_tab10  = __import__("pathlib").Path(__file__).parent.parent / \
              "03_salidas" / "tablas" / "tab10_cargas_factoriales.txt"

# =============================================================================
# FIG01 — INTEGRATED PSYCHOMETRIC-STRUCTURAL PORTRAIT
# =============================================================================

imprimir_seccion("FIG01 — INTEGRATED PSYCHOMETRIC-STRUCTURAL PORTRAIT")

from matplotlib.lines import Line2D
from matplotlib.path  import Path as MplPath

# ── Cargas (item-total correlations) por escala
item_loadings_fig = {}
item_labels_fig   = {}

for esc_name, (cols, suf, mapa, color) in escalas.items():
    cols_ok = [c + suf for c in cols if c + suf in df.columns]
    if not cols_ok:
        continue
    data_  = df[cols_ok].dropna()
    total_ = data_.sum(axis=1)
    loads  = []
    for col in cols_ok:
        r, _ = stats.pearsonr(data_[col], total_ - data_[col])
        loads.append(r)
    item_loadings_fig[esc_name] = loads
    pref = esc_name.split("(")[1].replace(")", "")
    item_labels_fig[esc_name] = [f"{pref}{i+1}" for i in range(len(loads))]

# ── Alpha por escala
alphas_fig = {}
for esc_name, (cols, suf, mapa, color) in escalas.items():
    cols_ok = [c + suf for c in cols if c + suf in df.columns]
    alphas_fig[esc_name] = cronbach_alpha(df[cols_ok].dropna())

# ── Correlaciones entre constructos
idx_esc_map = {
    "Stress (EA)":     "IDX_EA",
    "Fatigue (FA)":    "IDX_FA",
    "Motivation (MV)": "IDX_MV",
    "Equity (EQ)":     "IDX_EQ",
    "Well-being (BP)": "IDX_BP",
}
corr_pairs_fig = [
    ("Stress (EA)",     "Fatigue (FA)"),
    ("Stress (EA)",     "Motivation (MV)"),
    ("Stress (EA)",     "Equity (EQ)"),
    ("Stress (EA)",     "Well-being (BP)"),
    ("Fatigue (FA)",    "Motivation (MV)"),
    ("Fatigue (FA)",    "Equity (EQ)"),
    ("Fatigue (FA)",    "Well-being (BP)"),
    ("Equity (EQ)",     "Motivation (MV)"),
    ("Equity (EQ)",     "Well-being (BP)"),
    ("Motivation (MV)", "Well-being (BP)"),
]
corr_vals = {}
for n1, n2 in corr_pairs_fig:
    i1 = idx_esc_map[n1]; i2 = idx_esc_map[n2]
    r, p = stats.pearsonr(df[i1].dropna(), df[i2][df[i1].notna()])
    corr_vals[(n1, n2)] = (r, p)

# ── Layout figure
fig, ax = plt.subplots(figsize=(24, 15))
ax.set_xlim(0, 24); ax.set_ylim(0, 15)
ax.axis("off"); ax.set_facecolor("white")

# Posiciones de nodos
NODES_P = {
    "Stress (EA)":     (7.5,  11.5),
    "Fatigue (FA)":    (7.5,   7.5),
    "Equity (EQ)":     (7.5,   3.5),
    "Motivation (MV)": (16.0,  10.5),
    "Well-being (BP)": (16.0,   4.5),
}
EXOG_P = {
    "Workload":   (1.2, 11.8),
    "Harassment": (1.2,  9.8),
    "Gender":     (1.2,  7.8),
    "Origin":     (1.2,  5.5),
    "Sleep":      (1.2,  3.5),
}
COLORS_N = {
    "Stress (EA)":     COLORES_INDICES["IDX_EA"],
    "Fatigue (FA)":    COLORES_INDICES["IDX_FA"],
    "Motivation (MV)": COLORES_INDICES["IDX_MV"],
    "Equity (EQ)":     COLORES_INDICES["IDX_EQ"],
    "Well-being (BP)": COLORES_INDICES["IDX_BP"],
}

CR   = 1.10   # radio del círculo del constructo
SMAX = 0.80   # longitud máxima del radio del ítem

# ── Arcos de correlación entre constructos
for n1, n2 in corr_pairs_fig:
    r, p = corr_vals[(n1, n2)]
    if abs(r) < 0.08:
        continue
    x1, y1 = NODES_P[n1]; x2, y2 = NODES_P[n2]
    c_edge  = "#C0392B" if r > 0 else "#2980B9"
    alpha_e = 0.15 + abs(r) * 0.65
    lw_e    = 0.8 + abs(r) * 5.5
    rad     = 0.25 if (n1 != "Stress (EA)" or n2 != "Fatigue (FA)") else -0.35

    patch = mpatches.FancyArrowPatch(
        posA=(x1, y1), posB=(x2, y2),
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-", color=c_edge,
        lw=lw_e, alpha=alpha_e, zorder=1
    )
    ax.add_patch(patch)

    # Etiqueta r en punto medio del arco
    mx = (x1 + x2) / 2 - (y2 - y1) * rad * 0.5
    my = (y1 + y2) / 2 + (x2 - x1) * rad * 0.5
    sig_r = ("***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "")
    ax.text(mx, my, f"r={r:.2f}{sig_r}",
            ha="center", va="center", fontsize=8.5,
            color=c_edge, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.0),
            zorder=4)

# ── Flechas estructurales del SEM
PATHS_STRUCT = [
    ("Stress (EA)",     "Motivation (MV)",  0.008, False),
    ("Fatigue (FA)",    "Motivation (MV)", -0.043, False),
    ("Equity (EQ)",     "Motivation (MV)",  0.025, False),
    ("Stress (EA)",     "Well-being (BP)", -0.036, False),
    ("Fatigue (FA)",    "Well-being (BP)",  0.013, False),
    ("Motivation (MV)", "Well-being (BP)", -0.035, False),
    ("Equity (EQ)",     "Well-being (BP)", -0.098, True),
]
for n1, n2, beta, sig in PATHS_STRUCT:
    x1, y1 = NODES_P[n1]; x2, y2 = NODES_P[n2]
    dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    ux   = (x2-x1)/dist; uy = (y2-y1)/dist
    xs   = x1 + ux * (CR + 0.1)
    ys   = y1 + uy * (CR + 0.1)
    xe   = x2 - ux * (CR + 0.1)
    ye   = y2 - uy * (CR + 0.1)
    c_a  = "#1A1A2E" if sig else "#AAAAAA"
    lw_a = 3.0 if sig else 1.4
    ax.annotate("", xy=(xe, ye), xytext=(xs, ys),
        arrowprops=dict(
            arrowstyle="-|>,head_width=0.4,head_length=0.25",
            color=c_a, lw=lw_a,
            connectionstyle="arc3,rad=-0.08",
        ), zorder=5)
    stars = "*" if sig else ""
    ax.text((xs+xe)/2 + 0.35, (ys+ye)/2,
            f"β={beta:.2f}{stars}",
            ha="left", va="center", fontsize=10,
            color=c_a, fontweight="bold" if sig else "normal",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.2),
            zorder=6)

# ── Nodos con radios de ítems
for esc_name, (cx, cy) in NODES_P.items():
    color  = COLORS_N[esc_name]
    loads  = item_loadings_fig.get(esc_name, [])
    labels = item_labels_fig.get(esc_name, [])
    alpha_ = alphas_fig.get(esc_name, np.nan)
    n_it   = len(loads)

    # Ángulos para los radios — apuntando hacia el lado opuesto al flujo del SEM
    is_left = esc_name in ["Stress (EA)", "Fatigue (FA)", "Equity (EQ)"]
    if is_left:
        angles_deg = np.linspace(105, 255, n_it)  # hemisferio izquierdo
    else:
        angles_deg = np.linspace(-75, 75, n_it)    # hemisferio derecho

    for ang_d, load, lbl in zip(angles_deg, loads, labels):
        ang_r   = np.deg2rad(ang_d)
        spoke_l = SMAX * load
        xs_s    = cx + CR * np.cos(ang_r)
        ys_s    = cy + CR * np.sin(ang_r)
        xe_s    = cx + (CR + spoke_l) * np.cos(ang_r)
        ye_s    = cy + (CR + spoke_l) * np.sin(ang_r)

        ax.plot([xs_s, xe_s], [ys_s, ye_s],
                color=color, lw=4.5, alpha=0.72,
                solid_capstyle="round", zorder=2)

        xl = cx + (CR + spoke_l + 0.22) * np.cos(ang_r)
        yl = cy + (CR + spoke_l + 0.22) * np.sin(ang_r)
        ax.text(xl, yl, lbl, ha="center", va="center",
                fontsize=8.5, color=color,
                fontweight="bold", zorder=5)

    # Círculo principal
    circ = plt.Circle((cx, cy), CR,
                       facecolor=color, edgecolor="white",
                       linewidth=3.5, zorder=3, alpha=0.96)
    ax.add_patch(circ)

    # Nombre del constructo
    short = esc_name.split(" (")[0]
    ax.text(cx, cy + 0.28, short,
            ha="center", va="center", fontsize=13,
            fontweight="bold", color="white", zorder=7)
    # Alpha
    ax.text(cx, cy - 0.28,
            f"α = {alpha_:.3f}" if not np.isnan(alpha_) else "",
            ha="center", va="center", fontsize=10.5,
            color="white", zorder=7)

# ── Variables exógenas
EXOG_PATHS_FIG = {
    "Workload":   [("Stress (EA)", 0.057, False), ("Fatigue (FA)", -0.017, False)],
    "Harassment": [("Stress (EA)", 0.015, False), ("Equity (EQ)",   0.069, False)],
    "Gender":     [("Fatigue (FA)", -0.074, False)],
    "Origin":     [("Fatigue (FA)", -0.107, True)],
    "Sleep":      [("Fatigue (FA)", -0.005, False)],
}
for exog, (ex, ey) in EXOG_P.items():
    ax.text(ex, ey, exog,
            ha="center", va="center", fontsize=11,
            fontweight="bold", color="white", zorder=5,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#6B7280",
                      edgecolor="white", linewidth=1.5, alpha=0.92))
    for target, beta, sig in EXOG_PATHS_FIG.get(exog, []):
        tx, ty = NODES_P[target]
        c_ex = "#1A1A2E" if sig else "#AAAAAA"
        lw_ex = 2.0 if sig else 0.9
        ax.annotate("", xy=(tx - CR * 1.1, ty), xytext=(ex + 0.6, ey),
            arrowprops=dict(
                arrowstyle="-|>,head_width=0.3,head_length=0.2",
                color=c_ex, lw=lw_ex,
                connectionstyle="arc3,rad=0.04",
            ), zorder=4)

# ── Covarianza EA ↔ FA
x_ea, y_ea = NODES_P["Stress (EA)"]
x_fa, y_fa = NODES_P["Fatigue (FA)"]
ax.annotate("",
    xy=(x_fa - CR*0.7, y_fa + CR*0.7),
    xytext=(x_ea - CR*0.7, y_ea - CR*0.7),
    arrowprops=dict(arrowstyle="<->", color="#6B7280", lw=2.2,
                    connectionstyle="arc3,rad=-0.45"), zorder=4)
ax.text(x_ea - CR*1.5, (y_ea + y_fa)/2, "cov",
        fontsize=9, color="#6B7280", ha="center")

# ── Escala de radios (leyenda visual)
ax.text(20.8, 14.3, "Item loading scale:", fontsize=10,
        color="#555555", ha="left", fontweight="bold")
for val, lbl in [(0.40, "0.40"), (0.60, "0.60"), (0.80, "0.80")]:
    yy = 13.7 - val * 1.8
    ax.plot([20.8, 20.8 + SMAX * val], [yy, yy],
            color="#555555", lw=4.5, solid_capstyle="round")
    ax.text(20.8 + SMAX * val + 0.15, yy, lbl,
            fontsize=9, va="center", color="#555555")

# ── Leyenda principal
leg_elems = [
    mpatches.Patch(facecolor=COLORES_INDICES["IDX_EA"], label="Stress (EA)"),
    mpatches.Patch(facecolor=COLORES_INDICES["IDX_FA"], label="Fatigue (FA)"),
    mpatches.Patch(facecolor=COLORES_INDICES["IDX_EQ"], label="Equity (EQ)"),
    mpatches.Patch(facecolor=COLORES_INDICES["IDX_MV"], label="Motivation (MV)"),
    mpatches.Patch(facecolor=COLORES_INDICES["IDX_BP"], label="Well-being (BP)"),
    Line2D([0],[0], color="#1A1A2E", lw=2.5, label="Significant path (p < .05)"),
    Line2D([0],[0], color="#AAAAAA", lw=1.4, ls="--", label="Non-significant path"),
    Line2D([0],[0], color="#C0392B", lw=3,   label="Positive correlation"),
    Line2D([0],[0], color="#2980B9", lw=3,   label="Negative correlation"),
]
ax.legend(handles=leg_elems, loc="lower left",
          fontsize=10, framealpha=0.93, ncol=2,
          bbox_to_anchor=(0.0, 0.0))

ax.set_title(
    "Figure 1.  Integrated Psychometric–Structural Portrait\n"
    "Item loadings (spokes), inter-construct correlations (arcs) and SEM path coefficients",
    fontsize=18, fontweight="bold", pad=16
)
ax.text(0.5, -0.025,
        f"{NOTA_MUESTRA_EN}  "
        "Spoke length ∝ corrected item-total correlation (loading proxy).  "
        "Arc width ∝ |r|;  red = positive, blue = negative.  "
        f"α = Cronbach's Alpha inside node.  {NOTA_SIG_EN}",
        transform=ax.transAxes, ha="center", fontsize=9.5,
        color="#666666", style="italic")

fig.tight_layout()
guardar_figura(fig, "fig01_integrated_portrait.png")


# =============================================================================
# FIG02 — HEATMAP DE CARGAS FACTORIALES
# =============================================================================

imprimir_seccion("FIG02 — FACTOR LOADINGS HEATMAP")

# Construir matriz de cargas
all_items    = []
all_labels   = []
all_loadings = []
group_bounds = []
group_colors = []
group_names  = []

for nombre, (cols, suf, mapa, color) in escalas.items():
    cols_num = [c + suf for c in cols]
    cols_ok  = [c for c in cols_num if c in df.columns]
    if not cols_ok:
        continue

    data_  = df[cols_ok].dropna()
    total_ = data_.sum(axis=1)
    loadings = []
    for col in cols_ok:
        resto = total_ - data_[col]
        r, _  = stats.pearsonr(data_[col], resto)
        loadings.append(r)

    all_loadings.extend(loadings)
    group_bounds.append((len(all_items), len(all_items) + len(cols_ok)))
    group_colors.append(color)
    group_names.append(nombre)

    prefijo = nombre.split("(")[1].replace(")", "")
    for i in range(len(cols_ok)):
        all_labels.append(f"{prefijo}{i+1}")

n_items  = len(all_labels)
loadings_arr = np.array(all_loadings).reshape(1, -1)

fig, axes = plt.subplots(
    2, 1, figsize=(16, 5),
    gridspec_kw={"height_ratios": [0.08, 1]},
)

# ── Barra de color por constructo
ax_bar = axes[0]
ax_bar.set_xlim(0, n_items)
ax_bar.set_ylim(0, 1)
ax_bar.axis("off")
for (start, end), color, nombre in zip(group_bounds, group_colors, group_names):
    ax_bar.barh(0.5, end - start, left=start, color=color,
                height=0.9, align="center")
    ax_bar.text((start + end) / 2, 0.5, nombre,
                ha="center", va="center", fontsize=9,
                fontweight="bold", color="white")

# ── Heatmap
ax_hm = axes[1]
im = ax_hm.imshow(
    loadings_arr, aspect="auto", cmap="RdYlGn",
    vmin=0.2, vmax=0.8
)
ax_hm.set_xticks(range(n_items))
ax_hm.set_xticklabels(all_labels, fontsize=8, rotation=45, ha="right")
ax_hm.set_yticks([])

# Valores en las celdas
for j, val in enumerate(all_loadings):
    color_txt = "white" if val > 0.65 or val < 0.35 else "black"
    ax_hm.text(j, 0, f"{val:.2f}", ha="center", va="center",
               fontsize=7.5, color=color_txt, fontweight="bold")

# Separadores entre constructos
for start, end in group_bounds[1:]:
    ax_hm.axvline(x=start - 0.5, color="white", linewidth=2)

plt.colorbar(im, ax=ax_hm, orientation="horizontal",
             pad=0.18, fraction=0.04, label="Corrected item-total correlation (λ proxy)")

ax_hm.set_title(
    "Figure 2. Factor Loadings Heatmap (Item-Total Correlations by Construct)",
    fontsize=FS_TITULO, fontweight="bold", pad=10
)
fig.text(0.5, -0.04,
         f"{NOTA_MUESTRA_EN} Values represent corrected item-total correlations as loading proxies.",
         ha="center", fontsize=8, color="#6B7280", style="italic")

fig.tight_layout()
guardar_figura(fig, "fig02_factor_loadings_heatmap.png")


# =============================================================================
# FIG03 — MATRIZ DE CORRELACIONES ENTRE CONSTRUCTOS
# =============================================================================

imprimir_seccion("FIG03 — CONSTRUCT CORRELATIONS")

labels_cortos = [NOMBRES_INDICES_CORTOS_EN[i] for i in INDICES]
n_idx = len(INDICES)

mat_r = np.zeros((n_idx, n_idx))
mat_p = np.zeros((n_idx, n_idx))

for i, idx1 in enumerate(INDICES):
    for j, idx2 in enumerate(INDICES):
        if i == j:
            mat_r[i, j] = 1.0
        else:
            r, p = stats.pearsonr(
                df[idx1].dropna(),
                df[idx2][df[idx1].notna()],
            )
            mat_r[i, j] = r
            mat_p[i, j] = p

fig, ax = plt.subplots(figsize=(9, 8))

# Heatmap triangular inferior
mask_upper = np.triu(np.ones_like(mat_r, dtype=bool), k=1)
mat_plot   = np.where(mask_upper, np.nan, mat_r)

im = ax.imshow(mat_plot, cmap="RdBu_r", vmin=-0.6, vmax=0.6,
               aspect="equal")

# Anotaciones
for i in range(n_idx):
    for j in range(n_idx):
        if i == j:
            # Diagonal: nombre del constructo
            ax.text(j, i, labels_cortos[i],
                    ha="center", va="center", fontsize=10,
                    fontweight="bold", color="white",
                    bbox=dict(
                        facecolor=LISTA_COLORES_INDICES[i],
                        edgecolor="none", boxstyle="round,pad=0.3"
                    ))
        elif i > j:
            r  = mat_r[i, j]
            p  = mat_p[i, j]
            sig = ("***" if p < 0.001 else
                   "**"  if p < 0.01  else
                   "*"   if p < 0.05  else "")
            color_txt = "white" if abs(r) > 0.35 else "black"
            ax.text(j, i, f"{r:.2f}{sig}",
                    ha="center", va="center", fontsize=10.5,
                    fontweight="bold", color=color_txt)

ax.set_xticks(range(n_idx))
ax.set_yticks(range(n_idx))
ax.set_xticklabels(labels_cortos, fontsize=FS_TICK)
ax.set_yticklabels(labels_cortos, fontsize=FS_TICK)

# Colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Pearson r", fontsize=FS_EJE)

ax.set_title(
    "Figure 3. Pearson Correlation Matrix — Latent Constructs",
    fontsize=FS_TITULO, fontweight="bold", pad=14
)
fig.text(0.5, -0.03,
         f"{NOTA_MUESTRA_EN} Lower triangle. {NOTA_SIG_EN}",
         ha="center", fontsize=8, color="#6B7280", style="italic")

fig.tight_layout()
guardar_figura(fig, "fig03_construct_correlations.png")


# =============================================================================
# FIG04 — BUTTERFLY RAINCLOUD PLOT (mujeres ← | → hombres)
# =============================================================================

imprimir_seccion("FIG04 — BUTTERFLY RAINCLOUD PLOT")

df_gen = df[df["P4. Género"].isin(["Mujer", "Hombre"])].copy()
df_gen["Gender"] = df_gen["P4. Género"].map(GENERO_ES_EN)

color_w  = COLORES_GENERO_EN["Woman"]
color_m  = COLORES_GENERO_EN["Man"]
n_w = (df_gen["Gender"] == "Woman").sum()
n_m = (df_gen["Gender"] == "Man").sum()

fig, axes = plt.subplots(1, len(INDICES), figsize=(20, 8), sharey=True)

for ax, idx, color_idx in zip(axes, INDICES, LISTA_COLORES_INDICES):
    nombre_en = NOMBRES_INDICES_CORTOS_EN[idx]
    datos_w   = df_gen[df_gen["Gender"] == "Woman"][idx].dropna().values
    datos_m   = df_gen[df_gen["Gender"] == "Man"][idx].dropna().values

    y_range  = np.linspace(0.9, 5.1, 300)
    kde_w    = stats.gaussian_kde(datos_w, bw_method=0.35)(y_range)
    kde_m    = stats.gaussian_kde(datos_m, bw_method=0.35)(y_range)

    # Normalizar por el máximo global para comparar en la misma escala
    max_kde  = max(kde_w.max(), kde_m.max())
    kde_w_n  = kde_w / max_kde * 0.42
    kde_m_n  = kde_m / max_kde * 0.42

    # ── Violines empalados: mujer a la izquierda (negativo), hombre a la derecha (positivo)
    ax.fill_betweenx(y_range, -kde_w_n, 0,
                     color=color_w, alpha=0.50, zorder=2)
    ax.fill_betweenx(y_range,  0,  kde_m_n,
                     color=color_m, alpha=0.50, zorder=2)

    # Contorno de los violines
    ax.plot(-kde_w_n, y_range, color=color_w, lw=1.5, alpha=0.85, zorder=3)
    ax.plot( kde_m_n, y_range, color=color_m, lw=1.5, alpha=0.85, zorder=3)

    # ── Boxplots empalmados (ligeramente desplazados en x para visibilidad)
    for datos, color, x_off in [(datos_w, color_w, -0.06), (datos_m, color_m, 0.06)]:
        q1, med, q3 = np.percentile(datos, [25, 50, 75])
        iqr = q3 - q1
        wlo = max(datos.min(), q1 - 1.5 * iqr)
        whi = min(datos.max(), q3 + 1.5 * iqr)

        # Whiskers
        ax.plot([x_off, x_off], [wlo, q1], color=color, lw=2.0, zorder=4)
        ax.plot([x_off, x_off], [q3, whi], color=color, lw=2.0, zorder=4)
        # Caja
        rect = plt.Rectangle((x_off - 0.055, q1), 0.11, iqr,
                               facecolor=color, alpha=0.85,
                               edgecolor="white", linewidth=2, zorder=5)
        ax.add_patch(rect)
        # Mediana
        ax.plot([x_off - 0.055, x_off + 0.055], [med, med],
                color="white", lw=2.5, zorder=6)
        # Media (diamante)
        ax.plot(x_off, datos.mean(), marker="D",
                color="white", markersize=6,
                markeredgecolor=color, markeredgewidth=2, zorder=7)

    # ── Puntos dispersos: mujeres a la izquierda, hombres a la derecha
    for datos, color, sign in [(datos_w, color_w, -1), (datos_m, color_m, 1)]:
        np.random.seed(42)
        muestra  = np.random.choice(datos, size=min(70, len(datos)), replace=False)
        x_jitter = sign * np.random.uniform(0.12, 0.44, size=len(muestra))
        ax.scatter(x_jitter, muestra,
                   color=color, alpha=0.22, s=9, zorder=2, lw=0)

    # ── Test de significancia
    _, p_sw_w = stats.shapiro(datos_w[:1000])
    _, p_sw_m = stats.shapiro(datos_m[:1000])
    if p_sw_w > 0.05 and p_sw_m > 0.05:
        stat_t, p_t = stats.ttest_ind(datos_w, datos_m)
    else:
        stat_t, p_t = stats.mannwhitneyu(datos_w, datos_m, alternative="two-sided")

    sig_label = ("***" if p_t < 0.001 else
                 "**"  if p_t < 0.01  else
                 "*"   if p_t < 0.05  else "ns")
    ax.text(0, 5.25, sig_label, ha="center", va="bottom",
            fontsize=12, color="#1A1A2E", fontweight="bold")
    ax.plot([-0.1, 0.1], [5.18, 5.18], color="#1A1A2E", lw=1.2)

    # ── Línea de punto medio
    ax.axhline(3, color="#D1D5DB", lw=1.0, ls="--", zorder=1)

    # ── Medias numéricas
    ax.text(-0.50, datos_w.mean(),
            f"M={datos_w.mean():.2f}",
            ha="right", va="center", fontsize=9,
            color=color_w, fontweight="bold")
    ax.text( 0.50, datos_m.mean(),
            f"M={datos_m.mean():.2f}",
            ha="left",  va="center", fontsize=9,
            color=color_m, fontweight="bold")

    ax.set_xlim(-0.62, 0.62)
    ax.set_ylim(0.8, 5.5)
    ax.set_xticks([])
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.tick_params(axis="y", labelsize=11)
    ax.set_title(nombre_en, fontsize=14, fontweight="bold",
                 color=color_idx, pad=10)
    ax.spines[["top","right","bottom"]].set_visible(False)

axes[0].set_ylabel("Scale score (1–5)", fontsize=13)

# Etiquetas ← / →
for ax in axes:
    ax.text(-0.31, 0.97, "← Women", transform=ax.transAxes,
            ha="left", va="top", fontsize=9.5,
            color=color_w, fontweight="bold")
    ax.text( 0.69, 0.97, "Men →",   transform=ax.transAxes,
            ha="left", va="top", fontsize=9.5,
            color=color_m, fontweight="bold")

handles_g = [
    mpatches.Patch(facecolor=color_w, label=f"Women (n = {n_w})"),
    mpatches.Patch(facecolor=color_m, label=f"Men (n = {n_m})"),
    Line2D([0],[0], marker="D", color="gray",
           markerfacecolor="white", markersize=7, label="Group mean"),
]
fig.legend(handles=handles_g, loc="lower center",
           ncol=3, fontsize=12, framealpha=0.92,
           bbox_to_anchor=(0.5, -0.05))

fig.suptitle(
    "Figure 4.  Butterfly Raincloud Plot — Index Distributions by Gender",
    fontsize=18, fontweight="bold", y=1.02
)
fig.text(0.5, -0.10,
         f"{NOTA_MUESTRA_EN}  Women's distribution ←left, Men's distribution right→.  "
         "Diamond = group mean.  Dashed line = scale midpoint (3.0).  "
         f"***p < .001, **p < .01, *p < .05, ns = not significant.",
         ha="center", fontsize=9.5, color="#666666", style="italic")

fig.tight_layout()
guardar_figura(fig, "fig04_butterfly_raincloud.png")

# =============================================================================
# FIG05 — ENHANCED DIVERGING LIKERT + GENDER COMPARISON
# =============================================================================

imprimir_seccion("FIG05 — ENHANCED DIVERGING LIKERT + GENDER")

# Descripciones abreviadas de ítems (para mostrar en el eje Y)
ITEM_DESC = {
    "EA1": "Tasks generate pressure",       "EA2": "Exams cause anxiety",
    "EA3": "Demands exceed capacity",        "EA4": "Worried about grades",
    "EA5": "Not enough time for school",     "EA6": "Overwhelmed by content",
    "EA7": "Pressure causes physical symptoms","EA8": "Cannot disconnect mentally",
    "FA1": "Emotionally exhausted at day end","FA2": "Fatigue affects concentration",
    "FA3": "Arrive home without energy",     "FA4": "Wake up tired",
    "FA5": "Batteries at limit",             "FA6": "Hard to stay focused in class",
    "FA7": "Wanted to skip class to recover","FA8": "Fatigue affects work quality",
    "MV1": "Clear why I study this",         "MV2": "Learning is useful for future",
    "MV3": "Have concrete goals",            "MV4": "Imagine finishing degree",
    "MV5": "Find reasons to keep going",     "MV6": "Interested in postgrad",
    "MV7": "Positive job expectations",      "MV8": "Committed despite stress",
    "EQ1": "Treated fairly by faculty",      "EQ2": "Grades reflect performance",
    "EQ3": "Equal participation chances",    "EQ4": "Perceived preferential treatment",
    "EQ5": "Comfortable participating",      "EQ6": "Inclusive environment",
    "EQ7": "Expected less/more for wrong reasons","EQ8": "Feel I belong here",
    "BP1": "Life close to ideal",            "BP2": "Living conditions excellent",
    "BP3": "Satisfied with life now",        "BP4": "Achieved important goals",
    "BP5": "Would not change much",
}

df_w = df[df["P4. Género"] == "Mujer"].copy()
df_h = df[df["P4. Género"] == "Hombre"].copy()

escala_info_5 = [
    ("Stress (EA)",     EA_COLS, "_num",   likert_freq,    COLORES_INDICES["IDX_EA"],
     ["Never","Rarely","Sometimes","Freq.","Always"]),
    ("Fatigue (FA)",    FA_COLS, "_num",   likert_freq,    COLORES_INDICES["IDX_FA"],
     ["Never","Rarely","Sometimes","Freq.","Always"]),
    ("Motivation (MV)", MV_COLS, "_num",   likert_acuerdo, COLORES_INDICES["IDX_MV"],
     ["Str.Disagree","Disagree","Neutral","Agree","Str.Agree"]),
    ("Equity (EQ)",     EQ_COLS, "_final", likert_acuerdo, COLORES_INDICES["IDX_EQ"],
     ["Str.Disagree","Disagree","Neutral","Agree","Str.Agree"]),
    ("Well-being (BP)", BP_COLS, "_num",   likert_acuerdo, COLORES_INDICES["IDX_BP"],
     ["Str.Disagree","Disagree","Neutral","Agree","Str.Agree"]),
]

fig, axes = plt.subplots(
    len(escala_info_5), 1, figsize=(18, 22),
    gridspec_kw={"hspace": 0.70}
)

for ax, (nombre, cols, suf, mapa, color_base, cats) in \
        zip(axes, escala_info_5):

    cols_ok = [c + suf for c in cols if c + suf in df.columns]
    if not cols_ok:
        ax.axis("off"); continue

    # Prefijo del ítem (EA, FA, etc.)
    pref = nombre.split("(")[1].replace(")", "")
    n_items  = len(cols_ok)
    y_labels = []
    pcts_neg_tot, pcts_neu_tot, pcts_pos_tot = [], [], []
    pcts_pos_w, pcts_pos_m = [], []
    mean_w_list, mean_m_list = [], []
    sig_list = []

    for i, col in enumerate(cols_ok):
        item_key = f"{pref}{i+1}"
        desc     = ITEM_DESC.get(item_key, item_key)
        y_labels.append(f"{item_key}: {desc}")

        # Total
        vals = df[col].dropna()
        n_   = len(vals)
        pcts_neg_tot.append(((vals <= 2).sum() / n_) * 100)
        pcts_neu_tot.append(((vals == 3).sum() / n_) * 100)
        pcts_pos_tot.append(((vals >= 4).sum() / n_) * 100)

        # Por género (positivos)
        vw = df_w[col].dropna(); nh_c = df_h[col].dropna()
        pcts_pos_w.append(((vw >= 4).sum() / len(vw)) * 100 if len(vw) > 0 else np.nan)
        pcts_pos_m.append(((nh_c >= 4).sum() / len(nh_c)) * 100 if len(nh_c) > 0 else np.nan)
        mean_w_list.append(vw.mean())
        mean_m_list.append(nh_c.mean())

        # Test de diferencias (Mann-Whitney)
        if len(vw) > 2 and len(nh_c) > 2:
            _, p_mw = stats.mannwhitneyu(vw, nh_c, alternative="two-sided")
        else:
            p_mw = 1.0
        sig_list.append(p_mw)

    y_pos = np.arange(n_items)

    # ── Barras divergentes — total muestra
    c_neg = "#EF4444"; c_neu = "#E5E7EB"
    ax.barh(y_pos, [-p for p in pcts_neg_tot],
            color=c_neg, alpha=0.80, height=0.70,
            label="Negative / Disagree (1–2)")
    ax.barh(y_pos, [-p/2 for p in pcts_neu_tot],
            left=[-p for p in pcts_neg_tot],
            color=c_neu, alpha=0.75, height=0.70)
    ax.barh(y_pos, [p/2 for p in pcts_neu_tot],
            color=c_neu, alpha=0.75, height=0.70,
            label="Neutral (3)")
    ax.barh(y_pos, pcts_pos_tot,
            color=color_base, alpha=0.80, height=0.70,
            label="Positive / Agree (4–5)")

    # ── Etiquetas de porcentaje en barras
    for i, (pn, pneu, pp) in enumerate(zip(pcts_neg_tot, pcts_neu_tot, pcts_pos_tot)):
        if pn > 8:
            ax.text(-pn/2, i, f"{pn:.0f}%",
                    ha="center", va="center", fontsize=8.5,
                    color="white", fontweight="bold")
        if pp > 8:
            ax.text(pp/2, i, f"{pp:.0f}%",
                    ha="center", va="center", fontsize=8.5,
                    color="white", fontweight="bold")

    # ── Puntos de % positivo por género
    for i, (pw, pm) in enumerate(zip(pcts_pos_w, pcts_pos_m)):
        if not np.isnan(pw):
            ax.plot(pw, i + 0.27, marker="o", ms=7,
                    color=COLORES_GENERO_EN["Woman"],
                    markeredgecolor="white", markeredgewidth=1, zorder=6)
        if not np.isnan(pm):
            ax.plot(pm, i - 0.27, marker="s", ms=7,
                    color=COLORES_GENERO_EN["Man"],
                    markeredgecolor="white", markeredgewidth=1, zorder=6)

    # ── Diferencia de medias con estrella de significancia
    for i, (mw, mm, p_val) in enumerate(zip(mean_w_list, mean_m_list, sig_list)):
        if p_val < 0.05:
            stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*"
            # Convertir media a escala de porcentaje equivalente
            # Solo mostrar la estrella al extremo derecho
            ax.text(82, i, stars, ha="left", va="center",
                    fontsize=10, color="#1A1A2E", fontweight="bold")

    # ── Línea de media de la escala
    mean_scale = df[[c for c in cols_ok if c in df.columns]].mean().mean()
    # Convertir a % positivo para el eje
    mean_pct_equiv = ((mean_scale - 1) / 4) * 100 - 50  # centrado en 0
    ax.axvline(mean_pct_equiv, color=color_base, lw=1.5,
               ls=":", alpha=0.7, zorder=1)

    ax.axvline(0, color="#374151", lw=2.0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=9.5)
    ax.set_xlim(-82, 92)
    ticks_x = [-75, -50, -25, 0, 25, 50, 75]
    ax.set_xticks(ticks_x)
    ax.set_xticklabels([str(abs(t)) for t in ticks_x], fontsize=10)
    ax.set_xlabel("← Negative (%)    |    Positive (%) →", fontsize=11)
    ax.set_title(f"{nombre}", fontsize=14, fontweight="bold",
                 color=color_base, pad=8)
    ax.grid(axis="x", alpha=0.25, lw=0.8)
    ax.spines[["top","right"]].set_visible(False)

    # Leyenda interna
    from matplotlib.lines import Line2D as L2D
    h_leg = [
        mpatches.Patch(facecolor=c_neg,    label="Negative / Disagree (1–2)"),
        mpatches.Patch(facecolor=c_neu,    label="Neutral (3)"),
        mpatches.Patch(facecolor=color_base, label="Positive / Agree (4–5)"),
        L2D([0],[0], marker="o", color="w",
            markerfacecolor=COLORES_GENERO_EN["Woman"], ms=8,
            label="% Positive — Women"),
        L2D([0],[0], marker="s", color="w",
            markerfacecolor=COLORES_GENERO_EN["Man"], ms=8,
            label="% Positive — Men"),
    ]
    ax.legend(handles=h_leg, loc="lower right",
              fontsize=8.5, framealpha=0.88, ncol=2)

fig.suptitle(
    "Figure 5.  Enhanced Diverging Bar Chart — Likert Distributions with Gender Comparison",
    fontsize=18, fontweight="bold", y=1.008
)
fig.text(0.5, -0.012,
         f"{NOTA_MUESTRA_EN}  "
         "Bars = total sample percentages.  "
         "Circles (●) = % positive Women;  squares (■) = % positive Men.  "
         "Dotted vertical line = scale mean score (converted to % equivalent).  "
         "* p < .05, ** p < .01, *** p < .001 (Mann-Whitney U, gender comparison).",
         ha="center", fontsize=9.5, color="#666666", style="italic")

fig.tight_layout()
guardar_figura(fig, "fig05_enhanced_likert_gender.png")

# =============================================================================
# RESUMEN FINAL
# =============================================================================

imprimir_seccion("RESUMEN — FIGURAS DE PUBLICACIÓN")

figuras = [
    "fig01_sem_path_diagram.png",
    "fig02_factor_loadings_heatmap.png",
    "fig03_construct_correlations.png",
    "fig04_raincloud_gender.png",
    "fig05_likert_diverging.png",
]
for f in figuras:
    ruta   = FIGURAS_PUB_DIR / f
    estado = "✅" if ruta.exists() else "❌"
    tam    = f"{ruta.stat().st_size / 1024:.0f} KB" if ruta.exists() else ""
    print(f"  {estado} {f} {tam}")