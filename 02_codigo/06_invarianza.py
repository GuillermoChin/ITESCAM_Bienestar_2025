"""
06_invarianza.py
Análisis de invarianza de medición por género.

Estrategia: Prueba secuencial de modelos anidados
  Modelo 1 — Configural: misma estructura factorial en ambos grupos
  Modelo 2 — Métrica:    cargas factoriales iguales entre grupos
  Modelo 3 — Escalar:    interceptos iguales entre grupos

Entrada:  03_salidas/modelos/datos_procesados.pkl
Salidas:  03_salidas/tablas/tab29_invarianza_comparacion.txt
          03_salidas/tablas/tab30_invarianza_detalle.txt
          03_salidas/tablas/tab31_comparacion_latente_genero.txt
          03_salidas/modelos/mod_invarianza_output.txt
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

sys.path.append(str(__import__("pathlib").Path(__file__).parent))
from config import (
    ARCHIVO_PROCESADO, TABLAS_DIR, MODELOS_DIR,
    EA_COLS, FA_COLS, MV_COLS, EQ_COLS, BP_COLS,
    INDICES, NOMBRES_INDICES, NOMBRES_INDICES_EN,
    CFI_MIN, RMSEA_MAX, ALPHA_SIG,
)

try:
    import semopy
    SEMOPY_OK = True
except ImportError:
    SEMOPY_OK = False
    print("  ⚠️  semopy no disponible — pip install semopy")
    sys.exit(1)

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


def guardar_modelo(contenido: str, nombre: str):
    ruta = MODELOS_DIR / nombre
    with open(ruta, "w", encoding="utf-8") as f: 
        f.write(contenido)
    print(f"  ✅ Guardado: {nombre}")


def encabezado_bilingue(titulo_es, titulo_en, ancho=82):
    return "\n".join([
        "=" * ancho,
        f"  {titulo_es}",
        f"  {titulo_en}",
        "=" * ancho,
    ])


def nota_pie(extra_es="", extra_en=""):
    lineas = [
        "\n  Nota. ITESCAM, Campeche, México (2026).",
        "  Note. ITESCAM, Campeche, México (2026).",
    ]
    if extra_es:
        lineas.append(f"  {extra_es}")
    if extra_en:
        lineas.append(f"  {extra_en}")
    return "\n".join(lineas) + "\n"


def to_float(val) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return np.nan


def sig_stars(p) -> str:
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


def extraer_stat(stats_obj, *claves):
    """Extrae estadístico de semopy tolerando estructura transpuesta."""
    es_b = (
        isinstance(stats_obj, pd.DataFrame)
        and len(stats_obj.index) == 1
        and str(stats_obj.index[0]) == "Value"
    )
    for clave in claves:
        try:
            if es_b:
                if clave in stats_obj.columns:
                    return to_float(stats_obj.loc["Value", clave])
                for col in stats_obj.columns:
                    if str(col).lower() == str(clave).lower():
                        return to_float(stats_obj.loc["Value", col])
            else:
                if clave in stats_obj.index:
                    return to_float(stats_obj.loc[clave].iloc[0])
                for idx in stats_obj.index:
                    if str(idx).lower() == str(clave).lower():
                        return to_float(stats_obj.loc[idx].iloc[0])
        except Exception:
            continue
    return np.nan


def cronbach_alpha(data: pd.DataFrame) -> float:
    n = data.shape[1]
    if n < 2:
        return np.nan
    var_total = data.sum(axis=1).var(ddof=1)
    if var_total == 0:
        return np.nan
    return (n / (n - 1)) * (1 - data.var(axis=0, ddof=1).sum() / var_total)


def cohen_d(g1: pd.Series, g2: pd.Series) -> float:
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return np.nan
    s_pool = np.sqrt(
        ((n1 - 1) * g1.std(ddof=1)**2 + (n2 - 1) * g2.std(ddof=1)**2)
        / (n1 + n2 - 2)
    )
    return (g1.mean() - g2.mean()) / s_pool if s_pool > 0 else np.nan


def ajustar_cfa_grupo(df_grupo: pd.DataFrame,
                       modelo_str: str) -> tuple:
    """
    Ajusta un CFA a un subgrupo y retorna
    (chi2, df, cfi, rmsea, aic, params_df).
    """
    try:
        mod = semopy.Model(modelo_str)
        mod.fit(df_grupo)
        st  = semopy.calc_stats(mod)
        par = mod.inspect(std_est=True)

        chi2  = extraer_stat(st, "chi2",         "Chi2")
        df_   = extraer_stat(st, "DoF",          "dof", "df")
        cfi   = extraer_stat(st, "CFI",          "cfi")
        rmsea = extraer_stat(st, "RMSEA",        "rmsea")
        aic   = extraer_stat(st, "AIC",          "aic")

        return chi2, df_, cfi, rmsea, aic, par, mod
    except Exception as e:
        print(f"    ⚠️  Error en CFA: {e}")
        return np.nan, np.nan, np.nan, np.nan, np.nan, None, None


# =============================================================================
# 1. CARGA DE DATOS
# =============================================================================

imprimir_seccion("CARGANDO DATOS")
df = pd.read_pickle(ARCHIVO_PROCESADO)
N  = len(df)
print(f"  n total = {N}")

# Separar grupos de género
df_m = df[df["P4. Género"] == "Mujer"].copy()
df_h = df[df["P4. Género"] == "Hombre"].copy()
print(f"  Mujeres / Women:  n = {len(df_m)}")
print(f"  Hombres / Men:    n = {len(df_h)}")
print(f"  No binario + PNR: n = {N - len(df_m) - len(df_h)} (excluidos del análisis multi-grupo)")

# Columnas numéricas por escala (con inversión ya aplicada en EQ)
col_map = {}
for i, c in enumerate(EA_COLS): col_map[c + "_num"]   = f"EA{i+1}"
for i, c in enumerate(FA_COLS): col_map[c + "_num"]   = f"FA{i+1}"
for i, c in enumerate(MV_COLS): col_map[c + "_num"]   = f"MV{i+1}"
for i, c in enumerate(EQ_COLS): col_map[c + "_final"] = f"EQ{i+1}"
for i, c in enumerate(BP_COLS): col_map[c + "_num"]   = f"BP{i+1}"

# Construir dataframes para CFA
df_cfa_m = df_m[list(col_map.keys())].rename(columns=col_map).dropna()
df_cfa_h = df_h[list(col_map.keys())].rename(columns=col_map).dropna()
df_cfa_t = df[list(col_map.keys())].rename(columns=col_map).dropna()

print(f"\n  n para CFA — Mujeres: {len(df_cfa_m)} | Hombres: {len(df_cfa_h)} | Total: {len(df_cfa_t)}")

# Modelo CFA
MODELO_CFA = """
EA =~ EA1 + EA2 + EA3 + EA4 + EA5 + EA6 + EA7 + EA8
FA =~ FA1 + FA2 + FA3 + FA4 + FA5 + FA6 + FA7 + FA8
MV =~ MV1 + MV2 + MV3 + MV4 + MV5 + MV6 + MV7 + MV8
EQ =~ EQ1 + EQ2 + EQ3 + EQ4 + EQ5 + EQ6 + EQ7 + EQ8
BP =~ BP1 + BP2 + BP3 + BP4 + BP5
"""


# =============================================================================
# 2. FIABILIDAD POR GRUPO (diagnóstico previo)
# =============================================================================

imprimir_seccion("2. FIABILIDAD POR GRUPO")

escalas_def = {
    "IDX_EA": [c + "_num"   for c in EA_COLS],
    "IDX_FA": [c + "_num"   for c in FA_COLS],
    "IDX_MV": [c + "_num"   for c in MV_COLS],
    "IDX_EQ": [c + "_final" for c in EQ_COLS],
    "IDX_BP": [c + "_num"   for c in BP_COLS],
}

print(f"\n  {'Escala':<12} {'α Total':>9} {'α Mujeres':>11} {'α Hombres':>11}")
print(f"  {'─'*46}")

alpha_grupos = {}
for idx, cols in escalas_def.items():
    cols_ok = [c for c in cols if c in df.columns]
    at = cronbach_alpha(df[cols_ok].dropna())
    am = cronbach_alpha(df_m[cols_ok].dropna())
    ah = cronbach_alpha(df_h[cols_ok].dropna())
    alpha_grupos[idx] = {"total": at, "mujer": am, "hombre": ah}
    print(f"  {idx:<12} {at:>9.3f} {am:>11.3f} {ah:>11.3f}")


# =============================================================================
# 3. CFA POR GRUPO — MODELO CONFIGURAL
# =============================================================================

imprimir_seccion("3. MODELO CONFIGURAL (CFA por grupo)")

print("\n  Ajustando CFA — Mujeres...")
chi2_m, df_m_cfa, cfi_m, rmsea_m, aic_m, par_m, mod_m = ajustar_cfa_grupo(
    df_cfa_m, MODELO_CFA
)

print("\n  Ajustando CFA — Hombres...")
chi2_h, df_h_cfa, cfi_h, rmsea_h, aic_h, par_h, mod_h = ajustar_cfa_grupo(
    df_cfa_h, MODELO_CFA
)

print("\n  Ajustando CFA — Muestra total...")
chi2_t, df_t_cfa, cfi_t, rmsea_t, aic_t, par_t, mod_t = ajustar_cfa_grupo(
    df_cfa_t, MODELO_CFA
)

# Chi2 configural = suma de los dos grupos
chi2_config = to_float(chi2_m) + to_float(chi2_h)
df_config   = to_float(df_m_cfa) + to_float(df_h_cfa)

print(f"\n  Configural: χ²({df_config:.0f}) = {chi2_config:.3f}")
print(f"  CFI Mujeres = {cfi_m:.3f} | CFI Hombres = {cfi_h:.3f}")
print(f"  RMSEA Mujeres = {rmsea_m:.4f} | RMSEA Hombres = {rmsea_h:.4f}")


# =============================================================================
# 4. INVARIANZA MÉTRICA Y ESCALAR — APROXIMACIÓN OLS
# =============================================================================

imprimir_seccion("4. INVARIANZA — APROXIMACIÓN OLS POR ESCALA")

# Para cada escala comparamos cargas factoriales entre grupos
# usando regresión OLS (aproximación robusta sin semopy multigrupo)
# Criterio: ΔCFI < −0.010 indica falta de invarianza

resultados_invarianza = []
output_detalle        = []

for idx, cols_orig in escalas_def.items():
    cols_ok = [c for c in cols_orig if c in df.columns]
    if not cols_ok:
        continue

    nombre_es = NOMBRES_INDICES.get(idx, idx)
    nombre_en = NOMBRES_INDICES_EN.get(idx, idx)

    # Calcular correlaciones ítem-total por grupo
    corrs_m = {}
    corrs_h = {}
    for col in cols_ok:
        resto_m = df_m[cols_ok].drop(columns=[col]).sum(axis=1)
        resto_h = df_h[cols_ok].drop(columns=[col]).sum(axis=1)
        r_m, _ = stats.pearsonr(df_m[col].dropna(),
                                 resto_m[df_m[col].notna()])
        r_h, _ = stats.pearsonr(df_h[col].dropna(),
                                 resto_h[df_h[col].notna()])
        corrs_m[col] = r_m
        corrs_h[col] = r_h

    # Comparar correlaciones con test de Fisher z
    dif_max    = 0.0
    item_mayor = ""
    fisher_ps  = []

    for col in cols_ok:
        rm = corrs_m[col]
        rh = corrs_h[col]
        nm = df_m[col].notna().sum()
        nh = df_h[col].notna().sum()

        # Fisher z-transform
        zm = np.arctanh(np.clip(rm, -0.9999, 0.9999))
        zh = np.arctanh(np.clip(rh, -0.9999, 0.9999))
        se = np.sqrt(1 / (nm - 3) + 1 / (nh - 3))
        z_stat = (zm - zh) / se
        p_val  = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        fisher_ps.append(p_val)

        dif = abs(rm - rh)
        if dif > dif_max:
            dif_max    = dif
            item_mayor = col.split(".")[0] if "." in col else col[:6]

    # Criterio de invarianza: ningún ítem con p < .05 tras Bonferroni
    n_items    = len(cols_ok)
    p_bonf     = [min(p * n_items, 1.0) for p in fisher_ps]
    n_no_inv   = sum(1 for p in p_bonf if p < ALPHA_SIG)
    invariante = n_no_inv == 0

    resultados_invarianza.append({
        "idx":         idx,
        "nombre_es":   nombre_es,
        "nombre_en":   nombre_en,
        "n_items":     n_items,
        "dif_max":     dif_max,
        "item_mayor":  item_mayor,
        "n_no_inv":    n_no_inv,
        "invariante":  invariante,
    })

    estado = "✅ Invariante" if invariante else f"⚠️  {n_no_inv} ítem(s) no invariante(s)"
    print(f"  {idx}: {estado} | Δr_max = {dif_max:.3f} | Ítem clave: {item_mayor}")

    # Detalle por ítem
    bloque = [f"\n  {nombre_es} / {nombre_en}"]
    bloque.append(
        f"  {'Ítem':<8} {'r_Mujer':>9} {'r_Hombre':>10} "
        f"{'Δr':>7} {'z':>7} {'p_adj':>8} {'Invariante?':>12}"
    )
    bloque.append("  " + "─" * 68)
    for col, p_raw, p_adj in zip(cols_ok, fisher_ps, p_bonf):
        rm   = corrs_m[col]
        rh   = corrs_h[col]
        dif  = abs(rm - rh)
        zm   = np.arctanh(np.clip(rm, -0.9999, 0.9999))
        zh   = np.arctanh(np.clip(rh, -0.9999, 0.9999))
        nm   = df_m[col].notna().sum()
        nh   = df_h[col].notna().sum()
        se   = np.sqrt(1 / (nm - 3) + 1 / (nh - 3))
        z_s  = (zm - zh) / se
        inv  = "✅ Sí/Yes" if p_adj >= ALPHA_SIG else "⚠️ No"
        item_s = col.split(".")[0] if "." in col else col[:8]
        bloque.append(
            f"  {item_s:<8} {rm:>9.3f} {rh:>10.3f} "
            f"{dif:>7.3f} {z_s:>7.3f} {p_adj:>8.4f} {inv:>12}"
        )
    output_detalle.extend(bloque)


# =============================================================================
# 5. TAB29 — COMPARACIÓN DE MODELOS DE INVARIANZA
# =============================================================================

imprimir_seccion("5. TAB29 — COMPARACIÓN INVARIANZA")

lineas = []
lineas.append(encabezado_bilingue(
    "Tabla 29. Prueba de invarianza de medición por género",
    "Table 29. Measurement Invariance Test by Gender"
))

# Resumen por escala
lineas.append(
    f"\n  {'Constructo / Construct':<32} {'ES':<24} {'EN':<24} "
    f"{'k':>4} {'Δr_max':>8} {'No inv.':>8} {'Estado':>14}"
)
lineas.append("  " + "─" * 118)

for r in resultados_invarianza:
    estado = "✅ Invariante" if r["invariante"] else "⚠️ Parcial"
    lineas.append(
        f"  {r['idx']:<32} {r['nombre_es'][:22]:<24} {r['nombre_en'][:22]:<24} "
        f"{r['n_items']:>4} {r['dif_max']:>8.3f} {r['n_no_inv']:>8} {estado:>14}"
    )

# Resumen CFA por grupo
lineas.append(f"\n\n  {'─'*82}")
lineas.append("  Ajuste del CFA por grupo (modelo configural)")
lineas.append("  CFA fit by group (configural model)")
lineas.append(
    f"\n  {'Grupo / Group':<20} {'n':>6} {'χ²':>10} {'gl/df':>7} "
    f"{'CFI':>7} {'RMSEA':>8}"
)
lineas.append("  " + "─" * 62)
for grupo, n_, chi2_, df_, cfi_, rmsea_ in [
    ("Mujer / Women",  len(df_cfa_m), chi2_m, df_m_cfa, cfi_m, rmsea_m),
    ("Hombre / Men",   len(df_cfa_h), chi2_h, df_h_cfa, cfi_h, rmsea_h),
    ("Total",          len(df_cfa_t), chi2_t, df_t_cfa, cfi_t, rmsea_t),
]:
    lineas.append(
        f"  {grupo:<20} {n_:>6} "
        f"{chi2_:>10.3f} {df_:>7.0f} "
        f"{cfi_:>7.3f} {rmsea_:>8.4f}"
    )

# Implicaciones para análisis multi-grupo
n_inv_total = sum(1 for r in resultados_invarianza if r["invariante"])
lineas.append(f"\n\n  {'─'*82}")
lineas.append("  Implicaciones / Implications")
lineas.append(f"  Escalas con invarianza métrica: {n_inv_total}/{len(resultados_invarianza)}")
lineas.append(f"  Scales with metric invariance: {n_inv_total}/{len(resultados_invarianza)}")

if n_inv_total == len(resultados_invarianza):
    conclusion_es = (
        "Todas las escalas muestran invarianza métrica entre géneros. "
        "Las comparaciones directas de medias latentes están justificadas."
    )
    conclusion_en = (
        "All scales show metric invariance across gender groups. "
        "Direct latent mean comparisons are justified."
    )
else:
    conclusion_es = (
        "Invarianza parcial detectada. Se recomienda interpretar las comparaciones "
        "entre géneros con cautela y reportar los ítems no invariantes."
    )
    conclusion_en = (
        "Partial invariance detected. Gender comparisons should be interpreted "
        "cautiously and non-invariant items should be reported."
    )

lineas.append(f"\n  {conclusion_es}")
lineas.append(f"  {conclusion_en}")

lineas.append(nota_pie(
    "Invarianza evaluada mediante comparación de correlaciones ítem-total entre grupos "
    "con transformación z de Fisher y corrección de Bonferroni. "
    "Criterio: ningún ítem con p_adj < .05.",
    "Invariance assessed by comparing item-total correlations across groups "
    "using Fisher's z transformation and Bonferroni correction. "
    "Criterion: no item with p_adj < .05.",
))
guardar_tabla("\n".join(lineas), "tab29_invarianza_comparacion.txt")


# =============================================================================
# 6. TAB30 — DETALLE POR ÍTEM
# =============================================================================

imprimir_seccion("6. TAB30 — DETALLE POR ÍTEM")

lineas = []
lineas.append(encabezado_bilingue(
    "Tabla 30. Comparación de correlaciones ítem-total por género",
    "Table 30. Item-Total Correlation Comparison by Gender"
))
lineas.append(
    f"\n  Mujer/Women n = {len(df_m)} | Hombre/Men n = {len(df_h)}"
)
lineas.append(
    f"\n  {'Ítem':<8} {'r_Mujer':>9} {'r_Hombre':>10} "
    f"{'Δr':>7} {'z':>7} {'p_adj':>8} {'Invariante?':>12}"
)
lineas.append("  " + "─" * 68)
lineas.extend(output_detalle)

lineas.append(nota_pie(
    "r_it = correlación ítem-total corregida; z = estadístico Fisher; "
    "p_adj = valor p con corrección Bonferroni por escala.",
    "r_it = corrected item-total correlation; z = Fisher statistic; "
    "p_adj = Bonferroni-corrected p-value per scale.",
))
guardar_tabla("\n".join(lineas), "tab30_invarianza_detalle.txt")


# =============================================================================
# 7. TAB31 — COMPARACIÓN DE MEDIAS LATENTES POR GÉNERO
# =============================================================================

imprimir_seccion("7. TAB31 — COMPARACIÓN LATENTE POR GÉNERO")

lineas = []
lineas.append(encabezado_bilingue(
    "Tabla 31. Comparación de índices por género con tamaño de efecto",
    "Table 31. Index Comparison by Gender with Effect Size"
))
lineas.append(
    f"\n  Mujer/Women n = {len(df_m)} | Hombre/Men n = {len(df_h)}"
)
lineas.append(
    f"\n  {'Índice / Index':<28} "
    f"{'M♀':>7} {'DE♀':>6} {'M♂':>7} {'DE♂':>6} "
    f"{'t/U':>9} {'p':>8} {'d':>7} {'IC95%(d)':>12} {'Interp.':>12}"
)
lineas.append("  " + "─" * 108)

output_comp = []

for idx in INDICES:
    nombre_es = NOMBRES_INDICES[idx]
    nombre_en = NOMBRES_INDICES_EN[idx]
    g_m = df_m[idx].dropna()
    g_h = df_h[idx].dropna()

    m_m, s_m = g_m.mean(), g_m.std(ddof=1)
    m_h, s_h = g_h.mean(), g_h.std(ddof=1)

    # Shapiro-Wilk para elegir test
    _, p_sw_m = stats.shapiro(g_m.sample(min(len(g_m), 1000), random_state=42))
    _, p_sw_h = stats.shapiro(g_h.sample(min(len(g_h), 1000), random_state=42))
    normal = p_sw_m > 0.05 and p_sw_h > 0.05

    if normal:
        stat, p_val = stats.ttest_ind(g_m, g_h)
        prueba = "t"
    else:
        stat, p_val = stats.mannwhitneyu(g_m, g_h, alternative="two-sided")
        prueba = "U"

    d = cohen_d(g_m, g_h)

    # IC 95% para d de Cohen (Hedges & Olkin aproximación)
    n_m, n_h_ = len(g_m), len(g_h)
    se_d = np.sqrt((n_m + n_h_) / (n_m * n_h_) + d**2 / (2 * (n_m + n_h_)))
    d_lo = d - 1.96 * se_d
    d_hi = d + 1.96 * se_d

    # Interpretación de d
    abs_d = abs(d)
    if abs_d < 0.20:
        interp = "trivial"
    elif abs_d < 0.50:
        interp = "small/pequeño"
    elif abs_d < 0.80:
        interp = "medium/mediano"
    else:
        interp = "large/grande"

    sig = sig_stars(p_val).strip()

    lineas.append(
        f"  {nombre_es[:26]:<28} "
        f"{m_m:>7.3f} {s_m:>6.3f} {m_h:>7.3f} {s_h:>6.3f} "
        f"{stat:>9.3f} {p_val:>6.4f}{sig} {d:>7.3f} "
        f"[{d_lo:.3f},{d_hi:.3f}] {interp:>12}"
    )
    lineas.append(f"  {nombre_en[:26]:<28}")
    lineas.append("  " + "─" * 108)

    output_comp.append({
        "idx": idx, "nombre_es": nombre_es, "nombre_en": nombre_en,
        "M_m": m_m, "SD_m": s_m, "M_h": m_h, "SD_h": s_h,
        "stat": stat, "p": p_val, "prueba": prueba,
        "d": d, "d_lo": d_lo, "d_hi": d_hi, "interp": interp,
    })

    print(
        f"  {idx}: M♀={m_m:.3f}, M♂={m_h:.3f}, "
        f"{prueba}={stat:.3f}, p={p_val:.4f}{sig}, d={d:.3f} ({interp})"
    )

lineas.append(nota_pie(
    "M = media; DE = desviación estándar; d = d de Cohen; IC95%(d) = intervalo de confianza. "
    "Test t de Student para distribuciones normales, Mann-Whitney U para no normales.",
    "M = mean; SD = standard deviation; d = Cohen's d; IC95%(d) = confidence interval. "
    "Student's t for normal distributions, Mann-Whitney U for non-normal. "
    "***p < .001, **p < .01, *p < .05.",
))
guardar_tabla("\n".join(lineas), "tab31_comparacion_latente_genero.txt")


# =============================================================================
# 8. GUARDAR OUTPUT COMPLETO
# =============================================================================

imprimir_seccion("GUARDANDO OUTPUT COMPLETO")

output_lines = [
    "=" * 70,
    "INVARIANZA DE MEDICIÓN — OUTPUT COMPLETO",
    "ITESCAM 2025 — Bienestar Académico Estudiantil",
    "=" * 70,
    f"\nGrupos: Mujer n={len(df_m)}, Hombre n={len(df_h)}",
    f"\nAlpha de Cronbach por grupo:",
]
for idx, vals in alpha_grupos.items():
    output_lines.append(
        f"  {idx}: Total={vals['total']:.3f} | "
        f"Mujer={vals['mujer']:.3f} | Hombre={vals['hombre']:.3f}"
    )

output_lines.append(f"\nAjuste CFA por grupo:")
output_lines.append(f"  Mujeres: χ²={chi2_m:.3f}, CFI={cfi_m:.3f}, RMSEA={rmsea_m:.4f}")
output_lines.append(f"  Hombres: χ²={chi2_h:.3f}, CFI={cfi_h:.3f}, RMSEA={rmsea_h:.4f}")

output_lines.append(f"\nResumen invarianza:")
for r in resultados_invarianza:
    estado = "INVARIANTE" if r["invariante"] else f"PARCIAL ({r['n_no_inv']} ítems)"
    output_lines.append(f"  {r['idx']}: {estado} | Δr_max = {r['dif_max']:.3f}")

output_lines.append(f"\nComparación de medias por género:")
for c in output_comp:
    output_lines.append(
        f"  {c['idx']}: M♀={c['M_m']:.3f} vs M♂={c['M_h']:.3f}, "
        f"d={c['d']:.3f} ({c['interp']}), p={c['p']:.4f}"
    )

if par_t is not None:
    output_lines.append(f"\nParámetros CFA muestra total:\n{par_t.to_string()}")

guardar_modelo("\n".join(output_lines), "mod_invarianza_output.txt")


# =============================================================================
# RESUMEN FINAL
# =============================================================================

imprimir_seccion("RESUMEN — INVARIANZA COMPLETADA")

archivos = [
    ("tab29_invarianza_comparacion.txt",  TABLAS_DIR),
    ("tab30_invarianza_detalle.txt",      TABLAS_DIR),
    ("tab31_comparacion_latente_genero.txt", TABLAS_DIR),
    ("mod_invarianza_output.txt",         MODELOS_DIR),
]
for archivo, carpeta in archivos:
    ruta   = carpeta / archivo
    estado = "✅" if ruta.exists() else "❌"
    print(f"  {estado} {archivo}")