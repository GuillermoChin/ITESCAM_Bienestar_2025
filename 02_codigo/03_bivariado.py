"""
03_bivariado.py
Correlaciones, comparaciones por grupo y análisis bivariado.

Entrada:  03_salidas/modelos/datos_procesados.pkl
Salidas:  03_salidas/tablas/tab12_matriz_correlaciones.txt
          03_salidas/tablas/tab13_comparacion_genero.txt
          03_salidas/tablas/tab14_comparacion_carrera.txt
          03_salidas/tablas/tab15_comparacion_origen.txt
          03_salidas/tablas/tab16_comparacion_semestre.txt
          03_salidas/tablas/tab17_comparacion_carga_laboral.txt
          03_salidas/tablas/tab18_impacto_salud_mental.txt
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from scipy import stats
from itertools import combinations

sys.path.append(str(__import__("pathlib").Path(__file__).parent))
from config import (
    ARCHIVO_PROCESADO, TABLAS_DIR,
    INDICES, NOMBRES_INDICES, NOMBRES_INDICES_EN,
    NOMBRES_INDICES_CORTOS, NOMBRES_INDICES_CORTOS_EN,
    CARRERAS_ORDEN, SEMESTRE_ES_EN,
    ALPHA_SIG,
)

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def imprimir_seccion(titulo):
    print(f"\n{'='*60}")
    print(f"  {titulo}")
    print(f"{'='*60}")


def guardar_tabla(contenido: str, nombre: str):
    ruta = TABLAS_DIR / nombre
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(contenido)
    print(f"  ✅ Guardado: {nombre}")


def encabezado_bilingue(titulo_es, titulo_en, ancho=80):
    return "\n".join([
        "=" * ancho,
        f"  {titulo_es}",
        f"  {titulo_en}",
        "=" * ancho,
    ])


def nota_pie(extra_es="", extra_en=""):
    lineas = [
        "\n  Nota. n = 555. ITESCAM, Campeche, México (2026).",
        "  Note. n = 555. ITESCAM, Campeche, México (2026).",
    ]
    if extra_es:
        lineas.append(f"  {extra_es}")
    if extra_en:
        lineas.append(f"  {extra_en}")
    return "\n".join(lineas) + "\n"


def sig_stars(p: float) -> str:
    try:
        p = float(p)
        if np.isnan(p):
            return "   "
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "** "
        elif p < 0.05:
            return "*  "
        return "   "
    except Exception:
        return "   "


def cohen_d(grupo1: pd.Series, grupo2: pd.Series) -> float:
    """d de Cohen para dos grupos independientes."""
    n1, n2 = len(grupo1), len(grupo2)
    if n1 < 2 or n2 < 2:
        return np.nan
    s_pooled = np.sqrt(
        ((n1 - 1) * grupo1.std(ddof=1) ** 2 + (n2 - 1) * grupo2.std(ddof=1) ** 2)
        / (n1 + n2 - 2)
    )
    if s_pooled == 0:
        return np.nan
    return (grupo1.mean() - grupo2.mean()) / s_pooled


def eta_cuadrado(grupos: list) -> float:
    """η² (eta cuadrado) para ANOVA de un factor."""
    gran_media = np.concatenate([g.values for g in grupos]).mean()
    ss_between = sum(len(g) * (g.mean() - gran_media) ** 2 for g in grupos)
    ss_total   = sum(((g - gran_media) ** 2).sum() for g in grupos)
    return ss_between / ss_total if ss_total > 0 else np.nan


def interpretar_d(d: float) -> str:
    d = abs(d)
    if np.isnan(d):
        return "N/D"
    if d < 0.2:
        return "trivial"
    elif d < 0.5:
        return "small/pequeño"
    elif d < 0.8:
        return "medium/mediano"
    return "large/grande"


def interpretar_eta(eta: float) -> str:
    if np.isnan(eta):
        return "N/D"
    if eta < 0.01:
        return "trivial"
    elif eta < 0.06:
        return "small/pequeño"
    elif eta < 0.14:
        return "medium/mediano"
    return "large/grande"


def test_normalidad(serie: pd.Series) -> bool:
    """Shapiro-Wilk para n ≤ 5000. Retorna True si es normal (p > .05)."""
    if len(serie) < 3:
        return False
    try:
        _, p = stats.shapiro(serie.sample(min(len(serie), 1000), random_state=42))
        return p > 0.05
    except Exception:
        return False


# =============================================================================
# CARGA DE DATOS
# =============================================================================

imprimir_seccion("CARGANDO DATOS")
df = pd.read_pickle(ARCHIVO_PROCESADO)
N  = len(df)
print(f"  n = {N}")

# Verificar columnas necesarias
cols_requeridas = INDICES + ["GENERO_MUJER", "ORIGEN_RURAL", "CARGA_TOTAL",
                              "SEMESTRE_num", "P4. Género", "CARRERA_CORTA"]
faltantes = [c for c in cols_requeridas if c not in df.columns]
if faltantes:
    print(f"  ❌ Columnas faltantes: {faltantes}")
    sys.exit(1)
print(f"  ✅ Todas las columnas requeridas presentes")


# =============================================================================
# TAB12 — MATRIZ DE CORRELACIONES
# =============================================================================

imprimir_seccion("TAB12 — MATRIZ DE CORRELACIONES")

# Variables para la matriz
vars_corr = INDICES + [
    "P16a. ¿Qué tan probable: terminar a tiempo?_num",
    "P16b. ¿Qué tan probable: abandonar o interrumpir estudios?_num",
    "P16c. ¿Qué tan probable: continuar con un posgrado?_num",
    "CARGA_TOTAL",
    "P8_num",
    "SEMESTRE_num",
]

etiquetas_corr = {
    "IDX_EA":  "EA (Estrés / Stress)",
    "IDX_FA":  "FA (Fatiga / Fatigue)",
    "IDX_MV":  "MV (Motivación / Motivation)",
    "IDX_EQ":  "EQ (Equidad / Equity)",
    "IDX_BP":  "BP (Bienestar / Well-being)",
    "P16a. ¿Qué tan probable: terminar a tiempo?_num":
               "P16a (Terminar / Graduate)",
    "P16b. ¿Qué tan probable: abandonar o interrumpir estudios?_num":
               "P16b (Abandonar / Drop out)",
    "P16c. ¿Qué tan probable: continuar con un posgrado?_num":
               "P16c (Posgrado / Grad school)",
    "CARGA_TOTAL":   "Carga total / Total workload",
    "P8_num":        "Horas sueño / Sleep hours",
    "SEMESTRE_num":  "Semestre / Semester",
}

# Filtrar solo las que existen en el dataframe
vars_corr    = [v for v in vars_corr if v in df.columns]
etiquetas_ok = {k: v for k, v in etiquetas_corr.items() if k in vars_corr}

n_vars  = len(vars_corr)
datos_c = df[vars_corr].dropna()

# Calcular matriz de r y p
mat_r = np.zeros((n_vars, n_vars))
mat_p = np.zeros((n_vars, n_vars))

for i in range(n_vars):
    for j in range(n_vars):
        if i == j:
            mat_r[i, j] = 1.0
            mat_p[i, j] = 0.0
        elif i < j:
            r, p = stats.pearsonr(datos_c.iloc[:, i], datos_c.iloc[:, j])
            mat_r[i, j] = mat_r[j, i] = r
            mat_p[i, j] = mat_p[j, i] = p

print(f"  Variables en la matriz: {n_vars}")
print(f"  n para correlaciones:   {len(datos_c)}")

# Formatear tabla
lineas = []
lineas.append(encabezado_bilingue(
    "Tabla 12. Matriz de correlaciones de Pearson",
    "Table 12. Pearson Correlation Matrix"
))
lineas.append(f"\n  n = {len(datos_c)}")
lineas.append("\n  Variables:")
for i, v in enumerate(vars_corr):
    lineas.append(f"    {i+1:>2}. {etiquetas_ok.get(v, v)}")

ancho_col = 9
lineas.append(f"\n  {'Variable':<4}" + "".join(f"{i+1:>{ancho_col}}" for i in range(n_vars)))
lineas.append("  " + "─" * (4 + n_vars * ancho_col))

for i in range(n_vars):
    fila = f"  {i+1:<4}"
    for j in range(n_vars):
        if i == j:
            fila += f"{'—':>{ancho_col}}"
        elif j < i:
            r = mat_r[i, j]
            p = mat_p[i, j]
            s = sig_stars(p).strip()
            celda = f"{r:.2f}{s}"
            fila += f"{celda:>{ancho_col}}"
        else:
            fila += f"{'':>{ancho_col}}"
    lineas.append(fila)

lineas.append(nota_pie(
    "r de Pearson. Triángulo inferior. ***p < .001, **p < .01, *p < .05.",
    "Pearson's r. Lower triangle. ***p < .001, **p < .01, *p < .05.",
))
guardar_tabla("\n".join(lineas), "tab12_matriz_correlaciones.txt")


# =============================================================================
# TAB13 — COMPARACIÓN POR GÉNERO
# =============================================================================

imprimir_seccion("TAB13 — COMPARACIÓN POR GÉNERO")

df_gen = df[df["P4. Género"].isin(["Mujer", "Hombre"])].copy()
mujeres = df_gen[df_gen["P4. Género"] == "Mujer"]
hombres = df_gen[df_gen["P4. Género"] == "Hombre"]
n_m, n_h = len(mujeres), len(hombres)

print(f"  Mujer: n = {n_m} | Hombre: n = {n_h}")

lineas = []
lineas.append(encabezado_bilingue(
    "Tabla 13. Comparación de índices por género (Mujer vs. Hombre)",
    "Table 13. Index Comparison by Gender (Women vs. Men)"
))
lineas.append(
    f"\n  Mujer / Women n = {n_m} | Hombre / Men n = {n_h}"
)
lineas.append(
    f"\n  {'Índice / Index':<28} "
    f"{'M♀':>7} {'DE♀':>7} "
    f"{'M♂':>7} {'DE♂':>7} "
    f"{'Est.':>8} {'p':>7} "
    f"{'d':>7} {'Tamaño/Size':>14} {'Prueba/Test':>12}"
)
lineas.append("  " + "─" * 102)

for idx in INDICES:
    nombre_es = NOMBRES_INDICES[idx]
    nombre_en = NOMBRES_INDICES_EN[idx]
    g1 = mujeres[idx].dropna()
    g2 = hombres[idx].dropna()

    m1, s1 = g1.mean(), g1.std()
    m2, s2 = g2.mean(), g2.std()

    normal1 = test_normalidad(g1)
    normal2 = test_normalidad(g2)

    if normal1 and normal2:
        stat, p = stats.ttest_ind(g1, g2)
        prueba = "t de Student"
    else:
        stat, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
        prueba = "Mann-Whitney"

    d   = cohen_d(g1, g2)
    tam = interpretar_d(d)
    sig = sig_stars(p).strip()

    print(f"  {idx}: M♀={m1:.3f}, M♂={m2:.3f}, {prueba}, p={p:.4f}, d={d:.3f}")

    lineas.append(
        f"  {nombre_es[:26]:<28} "
        f"{m1:>7.3f} {s1:>7.3f} "
        f"{m2:>7.3f} {s2:>7.3f} "
        f"{stat:>8.3f} {p:>5.4f}{sig} "
        f"{d:>7.3f} {tam:>14} {prueba:>12}"
    )
    lineas.append(f"  {nombre_en[:26]:<28}")
    lineas.append("  " + "─" * 102)

lineas.append(nota_pie(
    "M = media; DE = desviación estándar; d = d de Cohen; "
    "t de Student para distribuciones normales, Mann-Whitney U para no normales.",
    "M = mean; SD = standard deviation; d = Cohen's d; "
    "Student's t for normal distributions, Mann-Whitney U for non-normal.",
))
guardar_tabla("\n".join(lineas), "tab13_comparacion_genero.txt")


# =============================================================================
# TAB14 — COMPARACIÓN POR CARRERA
# =============================================================================

imprimir_seccion("TAB14 — COMPARACIÓN POR CARRERA")

lineas = []
lineas.append(encabezado_bilingue(
    "Tabla 14. Comparación de índices por carrera (ANOVA / Kruskal-Wallis)",
    "Table 14. Index Comparison by Academic Program (ANOVA / Kruskal-Wallis)"
))

# ── Encabezado de columnas con siglas de carrera
header = f"\n  {'Índice':<10}" + "".join(f"{s:>8}" for s in CARRERAS_ORDEN)
header += f"{'F/H':>9} {'p':>8} {'η²':>7} {'Tamaño':>10}"
lineas.append(header)
lineas.append("  " + "─" * (10 + len(CARRERAS_ORDEN) * 8 + 36))

# ── Fila con n por carrera
fila_n = f"  {'n':<10}"
for sig in CARRERAS_ORDEN:
    n_c = (df["CARRERA_CORTA"] == sig).sum()
    fila_n += f"{n_c:>8}"
lineas.append(fila_n)
lineas.append("  " + "─" * (10 + len(CARRERAS_ORDEN) * 8 + 36))

for idx in INDICES:
    nombre_es = NOMBRES_INDICES_CORTOS[idx]
    nombre_en = NOMBRES_INDICES_CORTOS_EN[idx]

    grupos = []
    medias = []
    for sig in CARRERAS_ORDEN:
        g = df[df["CARRERA_CORTA"] == sig][idx].dropna()
        grupos.append(g)
        medias.append(g.mean())

    # ANOVA o Kruskal-Wallis
    normales = all(test_normalidad(g) for g in grupos if len(g) >= 3)
    if normales:
        stat, p = stats.f_oneway(*grupos)
        prueba  = "ANOVA"
    else:
        stat, p = stats.kruskal(*grupos)
        prueba  = "K-W"

    eta2 = eta_cuadrado(grupos)
    tam  = interpretar_eta(eta2)
    sig_s = sig_stars(p).strip()

    fila = f"  {nombre_es[:8]:<10}"
    for m in medias:
        fila += f"{m:>8.2f}"
    fila += f"{stat:>9.2f} {p:>6.4f}{sig_s} {eta2:>7.3f} {tam:>10}"
    lineas.append(fila)
    lineas.append(f"  {nombre_en[:8]:<10}")

# ── Post-hoc Tukey para el índice con diferencias significativas
lineas.append(f"\n\n  {'─'*80}")
lineas.append("  Comparaciones post-hoc (Tukey HSD) para diferencias significativas")
lineas.append("  Post-hoc comparisons (Tukey HSD) for significant differences")
lineas.append("  (Solo pares con p < .05 / Only pairs with p < .05)\n")

for idx in INDICES:
    nombre_es = NOMBRES_INDICES[idx]
    nombre_en = NOMBRES_INDICES_EN[idx]

    grupos_dict = {
        sig: df[df["CARRERA_CORTA"] == sig][idx].dropna()
        for sig in CARRERAS_ORDEN
    }

    normales = all(test_normalidad(g) for g in grupos_dict.values() if len(g) >= 3)
    _, p_global = stats.f_oneway(*grupos_dict.values()) if normales \
        else stats.kruskal(*grupos_dict.values())

    if p_global >= ALPHA_SIG:
        continue

    lineas.append(f"  {nombre_es} / {nombre_en}")
    lineas.append(
        f"  {'Par / Pair':<14} {'M(A)':>7} {'M(B)':>7} "
        f"{'Dif':>7} {'p':>8} {'d':>7}"
    )
    lineas.append("  " + "─" * 56)

    hay_sig = False
    for (s1, g1), (s2, g2) in combinations(grupos_dict.items(), 2):
        if len(g1) < 2 or len(g2) < 2:
            continue
        _, p_par = stats.ttest_ind(g1, g2)
        n_pares = len(list(combinations(CARRERAS_ORDEN, 2)))
        p_adj = min(p_par * n_pares, 1.0)  # Bonferroni simple
        if p_adj < ALPHA_SIG:
            d = cohen_d(g1, g2)
            lineas.append(
                f"  {s1}–{s2:<11} {g1.mean():>7.3f} {g2.mean():>7.3f} "
                f"{g1.mean()-g2.mean():>7.3f} {p_adj:>8.4f} {d:>7.3f}"
            )
            hay_sig = True

    if not hay_sig:
        lineas.append("  Sin diferencias significativas tras corrección Bonferroni.")
        lineas.append("  No significant differences after Bonferroni correction.")
    lineas.append("")

lineas.append(nota_pie(
    "M = media; η² = eta cuadrado; K-W = Kruskal-Wallis. "
    "Post-hoc con corrección Bonferroni.",
    "M = mean; η² = eta squared; K-W = Kruskal-Wallis. "
    "Post-hoc with Bonferroni correction.",
))
guardar_tabla("\n".join(lineas), "tab14_comparacion_carrera.txt")


# =============================================================================
# TAB15 — COMPARACIÓN POR ORIGEN (RURAL VS URBANO)
# =============================================================================

imprimir_seccion("TAB15 — COMPARACIÓN POR ORIGEN")

df_ori = df[df["ORIGEN_RURAL"].notna()].copy()
rural  = df_ori[df_ori["ORIGEN_RURAL"] == 1]
urbano = df_ori[df_ori["ORIGEN_RURAL"] == 0]
n_r, n_u = len(rural), len(urbano)

print(f"  Rural: n = {n_r} | Urbano/semi-urbano: n = {n_u}")

lineas = []
lineas.append(encabezado_bilingue(
    "Tabla 15. Comparación de índices por lugar de origen (Rural vs. Urbano)",
    "Table 15. Index Comparison by Place of Origin (Rural vs. Urban)"
))
lineas.append(
    f"\n  Rural n = {n_r} | Urbano / Urban n = {n_u}"
)
lineas.append(
    f"\n  {'Índice / Index':<28} "
    f"{'M_R':>7} {'DE_R':>7} "
    f"{'M_U':>7} {'DE_U':>7} "
    f"{'Est.':>8} {'p':>7} "
    f"{'d':>7} {'Tamaño/Size':>14}"
)
lineas.append("  " + "─" * 94)

for idx in INDICES:
    nombre_es = NOMBRES_INDICES[idx]
    nombre_en = NOMBRES_INDICES_EN[idx]
    g1 = rural[idx].dropna()
    g2 = urbano[idx].dropna()

    m1, s1 = g1.mean(), g1.std()
    m2, s2 = g2.mean(), g2.std()

    if test_normalidad(g1) and test_normalidad(g2):
        stat, p = stats.ttest_ind(g1, g2)
    else:
        stat, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")

    d   = cohen_d(g1, g2)
    tam = interpretar_d(d)
    sig = sig_stars(p).strip()

    lineas.append(
        f"  {nombre_es[:26]:<28} "
        f"{m1:>7.3f} {s1:>7.3f} "
        f"{m2:>7.3f} {s2:>7.3f} "
        f"{stat:>8.3f} {p:>5.4f}{sig} "
        f"{d:>7.3f} {tam:>14}"
    )
    lineas.append(f"  {nombre_en[:26]:<28}")
    lineas.append("  " + "─" * 94)

lineas.append(nota_pie(
    "Rural incluye: pueblo o localidad rural y comunidad indígena/muy pequeña. "
    "Urbano incluye: cabecera municipal y ciudad grande.",
    "Rural includes: rural town and indigenous/very small community. "
    "Urban includes: small city and large city.",
))
guardar_tabla("\n".join(lineas), "tab15_comparacion_origen.txt")


# =============================================================================
# TAB16 — COMPARACIÓN POR SEMESTRE
# =============================================================================

imprimir_seccion("TAB16 — COMPARACIÓN POR SEMESTRE")

semestres_ord = [2, 4, 6, 8]
sem_labels    = {2: "2°", 4: "4°", 6: "6°", 8: "8°"}
sem_labels_en = {2: "2nd", 4: "4th", 6: "6th", 8: "8th"}

lineas = []
lineas.append(encabezado_bilingue(
    "Tabla 16. Comparación de índices por semestre",
    "Table 16. Index Comparison by Semester"
))

header = f"\n  {'Índice':<10}" + "".join(
    f"{sem_labels[s]+'/'+sem_labels_en[s]:>12}" for s in semestres_ord
)
header += f"{'F/H':>9} {'p':>8} {'η²':>7}"
lineas.append(header)
lineas.append("  " + "─" * (10 + len(semestres_ord) * 12 + 26))

fila_n = f"  {'n':<10}"
for s in semestres_ord:
    n_s = (df["SEMESTRE_num"] == s).sum()
    fila_n += f"{n_s:>12}"
lineas.append(fila_n)
lineas.append("  " + "─" * (10 + len(semestres_ord) * 12 + 26))

for idx in INDICES:
    nombre_es = NOMBRES_INDICES_CORTOS[idx]
    nombre_en = NOMBRES_INDICES_CORTOS_EN[idx]

    grupos = [df[df["SEMESTRE_num"] == s][idx].dropna() for s in semestres_ord]
    medias = [g.mean() for g in grupos]

    normales = all(test_normalidad(g) for g in grupos if len(g) >= 3)
    if normales:
        stat, p = stats.f_oneway(*grupos)
    else:
        stat, p = stats.kruskal(*grupos)

    eta2 = eta_cuadrado(grupos)
    sig_s = sig_stars(p).strip()

    fila = f"  {nombre_es[:8]:<10}"
    for m in medias:
        fila += f"{m:>12.3f}"
    fila += f"{stat:>9.2f} {p:>6.4f}{sig_s} {eta2:>7.3f}"
    lineas.append(fila)
    lineas.append(f"  {nombre_en[:8]:<10}")

# Tendencia lineal por semestre
lineas.append(f"\n\n  {'─'*70}")
lineas.append("  Correlación de Pearson con semestre (tendencia lineal)")
lineas.append("  Pearson correlation with semester (linear trend)")
lineas.append(
    f"\n  {'Índice / Index':<36} {'r':>8} {'p':>8} {'Dirección/Direction':>20}"
)
lineas.append("  " + "─" * 74)

df_sem = df[df["SEMESTRE_num"].notna()].copy()
for idx in INDICES:
    nombre_es = NOMBRES_INDICES[idx]
    nombre_en = NOMBRES_INDICES_EN[idx]
    datos_   = df_sem[[idx, "SEMESTRE_num"]].dropna()
    r, p     = stats.pearsonr(datos_[idx], datos_["SEMESTRE_num"])
    sig_s    = sig_stars(p).strip()
    direccion = "↑ aumenta/increases" if r > 0 else "↓ disminuye/decreases"

    lineas.append(
        f"  {nombre_es[:34]:<36} "
        f"{r:>8.3f} {p:>6.4f}{sig_s} {direccion:>20}"
    )

lineas.append(nota_pie(
    "Tendencia lineal estimada con r de Pearson entre el índice y el número de semestre.",
    "Linear trend estimated with Pearson's r between index and semester number.",
))
guardar_tabla("\n".join(lineas), "tab16_comparacion_semestre.txt")


# =============================================================================
# TAB17 — COMPARACIÓN POR CARGA LABORAL
# =============================================================================

imprimir_seccion("TAB17 — COMPARACIÓN POR CARGA LABORAL")

niveles_trabajo = {
    0: "Sin trabajo / No work",
    1: "< 5h/sem",
    2: "5–10h/sem",
    3: "10–20h/sem",
    4: "20–30h/sem",
    5: "> 30h/sem",
}

lineas = []
lineas.append(encabezado_bilingue(
    "Tabla 17. Índices de bienestar según carga laboral (P11)",
    "Table 17. Well-being Indices by Paid Work Load (P11)"
))

header = f"\n  {'Nivel / Level':<22}"
for nivel, etiqueta in niveles_trabajo.items():
    lineas_et = etiqueta.split("/")
    col_header = lineas_et[0].strip()[:8]
    header += f"{col_header:>10}"
header += f"{'F/H':>9} {'p':>8} {'η²':>7}"
lineas.append(header)

sub_header = f"  {'':<22}"
for _, etiqueta in niveles_trabajo.items():
    partes = etiqueta.split("/")
    col_sub = partes[1].strip()[:8] if len(partes) > 1 else ""
    sub_header += f"{col_sub:>10}"
lineas.append(sub_header)
lineas.append("  " + "─" * (22 + len(niveles_trabajo) * 10 + 26))

# n por nivel
fila_n = f"  {'n':<22}"
for nivel in niveles_trabajo:
    n_niv = (df["P11_num"] == nivel).sum()
    fila_n += f"{n_niv:>10}"
lineas.append(fila_n)
lineas.append("  " + "─" * (22 + len(niveles_trabajo) * 10 + 26))

for idx in INDICES:
    nombre_es = NOMBRES_INDICES_CORTOS[idx]
    nombre_en = NOMBRES_INDICES_CORTOS_EN[idx]

    grupos = [
        df[df["P11_num"] == nivel][idx].dropna()
        for nivel in niveles_trabajo
    ]
    medias = [g.mean() if len(g) > 0 else np.nan for g in grupos]

    grupos_validos = [g for g in grupos if len(g) >= 2]
    if len(grupos_validos) >= 2:
        normales = all(test_normalidad(g) for g in grupos_validos if len(g) >= 3)
        if normales:
            stat, p = stats.f_oneway(*grupos_validos)
        else:
            stat, p = stats.kruskal(*grupos_validos)
        eta2 = eta_cuadrado(grupos_validos)
        sig_s = sig_stars(p).strip()
    else:
        stat, p, eta2, sig_s = np.nan, np.nan, np.nan, ""

    fila = f"  {nombre_es[:20]:<22}"
    for m in medias:
        if np.isnan(m):
            fila += f"{'—':>10}"
        else:
            fila += f"{m:>10.3f}"
    fila += f"{stat:>9.2f} {p:>6.4f}{sig_s} {eta2:>7.3f}"
    lineas.append(fila)
    lineas.append(f"  {nombre_en[:20]:<22}")

# Correlación carga_total con índices
lineas.append(f"\n\n  {'─'*70}")
lineas.append("  Correlación de Pearson: carga total vs. índices")
lineas.append("  Pearson correlation: total workload vs. indices")
lineas.append(
    f"\n  {'Índice / Index':<36} {'r':>8} {'p':>8}"
)
lineas.append("  " + "─" * 54)
for idx in INDICES:
    nombre_es = NOMBRES_INDICES[idx]
    datos_ = df[["CARGA_TOTAL", idx]].dropna()
    r, p   = stats.pearsonr(datos_["CARGA_TOTAL"], datos_[idx])
    sig_s  = sig_stars(p).strip()
    lineas.append(f"  {nombre_es[:34]:<36} {r:>8.3f} {p:>6.4f}{sig_s}")

lineas.append(nota_pie(
    "Niveles de trabajo: 0 = no trabaja, 5 = tiempo completo (>30h). η² = eta cuadrado.",
    "Work levels: 0 = no work, 5 = full-time (>30h). η² = eta squared.",
))
guardar_tabla("\n".join(lineas), "tab17_comparacion_carga_laboral.txt")


# =============================================================================
# TAB18 — IMPACTO EN SALUD MENTAL: REGRESIÓN LOGÍSTICA ORDINAL
# =============================================================================

imprimir_seccion("TAB18 — PREDICTORES DE IMPACTO EN SALUD MENTAL")

try:
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    df_log = df[
        ["IMPACTO_MENTAL"] + INDICES +
        ["GENERO_MUJER", "ORIGEN_RURAL", "CARGA_TOTAL",
         "P8_num", "SEMESTRE_num", "ACOSO_EXP"]
    ].dropna()

    y = df_log["IMPACTO_MENTAL"].astype(int)
    X = df_log[INDICES + ["GENERO_MUJER", "ORIGEN_RURAL",
                           "CARGA_TOTAL", "P8_num",
                           "SEMESTRE_num", "ACOSO_EXP"]]

    modelo_ord = OrderedModel(y, X, distr="logit")
    resultado  = modelo_ord.fit(method="bfgs", disp=False)

    lineas = []
    lineas.append(encabezado_bilingue(
        "Tabla 18. Predictores del impacto en salud mental — Regresión logística ordinal",
        "Table 18. Predictors of Mental Health Impact — Ordinal Logistic Regression"
    ))
    lineas.append(f"\n  n = {len(df_log)} | Variable dependiente: P20 (1–4)")
    lineas.append(f"  Dependent variable: P20 mental health impact (1 = none, 4 = severe)\n")
    lineas.append(
        f"  {'Variable':<36} {'Coef.':>8} {'SE':>7} {'z':>7} {'p':>8} {'OR':>8}"
    )
    lineas.append("  " + "─" * 78)

    etiquetas_pred = {
        "IDX_EA":        "Estrés Académico / Academic Stress",
        "IDX_FA":        "Fatiga / Fatigue",
        "IDX_MV":        "Motivación / Motivation",
        "IDX_EQ":        "Equidad / Equity",
        "IDX_BP":        "Bienestar / Well-being",
        "GENERO_MUJER":  "Género (Mujer=1) / Gender (Woman=1)",
        "ORIGEN_RURAL":  "Origen Rural / Rural origin",
        "CARGA_TOTAL":   "Carga total / Total workload",
        "P8_num":        "Horas de sueño / Sleep hours",
        "SEMESTRE_num":  "Semestre",
        "ACOSO_EXP":     "Experiencia de acoso / Harassment exp.",
    }

# Extraer parámetros directamente (compatible con todas las versiones)
    coefs  = resultado.params
    ses    = resultado.bse
    zvals  = resultado.tvalues
    pvals  = resultado.pvalues

    for var in X.columns:
        if var not in coefs.index:
            continue
        coef  = float(coefs[var])
        se    = float(ses[var])
        z     = float(zvals[var])
        p     = float(pvals[var])
        OR    = np.exp(coef)
        sig_s = sig_stars(p).strip()
        etiq  = etiquetas_pred.get(var, var)[:34]
        lineas.append(
            f"  {etiq:<36} {coef:>8.3f} {se:>7.3f} "
            f"{z:>7.3f} {p:>6.4f}{sig_s} {OR:>8.3f}"
        )

    # Umbrales del modelo ordinal (thresholds)
    lineas.append(f"\n  {'─'*78}")
    lineas.append("  Umbrales / Thresholds")
    for var in coefs.index:
        if var not in X.columns:
            coef  = float(coefs[var])
            se    = float(ses[var])
            z     = float(zvals[var])
            p     = float(pvals[var])
            sig_s = sig_stars(p).strip()
            lineas.append(
                f"  {str(var):<36} {coef:>8.3f} {se:>7.3f} "
                f"{z:>7.3f} {p:>6.4f}{sig_s}"
            )

    # Bondad de ajuste
    try:
        pseudo_r2 = resultado.prsquared
        lineas.append(f"\n  Pseudo R² (McFadden): {pseudo_r2:.4f}")
    except AttributeError:
        lineas.append("\n  Pseudo R²: no disponible / not available")

    try:
        lineas.append(f"  Log-likelihood: {resultado.llf:.2f}")
        lineas.append(f"  AIC: {resultado.aic:.2f}")
    except AttributeError:
        pass

    lineas.append(nota_pie(
        "OR = Odds Ratio (exp(coef)). Coeficientes positivos indican mayor impacto en salud mental.",
        "OR = Odds Ratio (exp(coef)). Positive coefficients indicate higher mental health impact.",
    ))
    guardar_tabla("\n".join(lineas), "tab18_impacto_salud_mental.txt")
    print(f"  Pseudo R² (McFadden): {resultado.prsquared:.4f}")

except ImportError:
    print("  ⚠️  statsmodels no disponible — instalar con: pip install statsmodels")
    print("  Se guardará tabla con nota de omisión.")
    lineas = [
        encabezado_bilingue(
            "Tabla 18. Predictores del impacto en salud mental",
            "Table 18. Predictors of Mental Health Impact"
        ),
        "\n  Nota: Análisis omitido. Instalar statsmodels para generar esta tabla.",
        "  Note: Analysis omitted. Install statsmodels to generate this table.",
    ]
    guardar_tabla("\n".join(lineas), "tab18_impacto_salud_mental.txt")

except Exception as e:
    print(f"  ⚠️  Error en regresión logística ordinal: {e}")
    lineas = [
        encabezado_bilingue(
            "Tabla 18. Predictores del impacto en salud mental",
            "Table 18. Predictors of Mental Health Impact"
        ),
        f"\n  Error al ajustar el modelo: {e}",
    ]
    guardar_tabla("\n".join(lineas), "tab18_impacto_salud_mental.txt")


# =============================================================================
# RESUMEN FINAL
# =============================================================================

imprimir_seccion("RESUMEN — BIVARIADO COMPLETADO")
tablas = [
    "tab12_matriz_correlaciones.txt",
    "tab13_comparacion_genero.txt",
    "tab14_comparacion_carrera.txt",
    "tab15_comparacion_origen.txt",
    "tab16_comparacion_semestre.txt",
    "tab17_comparacion_carga_laboral.txt",
    "tab18_impacto_salud_mental.txt",
]
for t in tablas:
    ruta   = TABLAS_DIR / t
    estado = "✅" if ruta.exists() else "❌"
    print(f"  {estado} {t}")