"""
05_sem.py
Modelado de Ecuaciones Estructurales — Análisis de rutas con índices observados.

Entrada:  03_salidas/modelos/datos_procesados.pkl
Salidas:  03_salidas/tablas/tab25_sem_ajuste.txt
          03_salidas/tablas/tab26_sem_efectos_directos.txt
          03_salidas/tablas/tab27_sem_efectos_indirectos.txt
          03_salidas/tablas/tab28_sem_efectos_totales.txt
          03_salidas/modelos/mod_sem_especificacion.txt
          03_salidas/modelos/mod_sem_output_completo.txt
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
    INDICES, NOMBRES_INDICES, NOMBRES_INDICES_EN,
    CFI_MIN, TLI_MIN, RMSEA_MAX, SRMR_MAX,
    N_BOOTSTRAP, ALPHA_SIG,
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
        "\n  Nota. n = 555. ITESCAM, Campeche, México (2026).",
        "  Note. n = 555. ITESCAM, Campeche, México (2026).",
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


def interpretar_r2(r2: float) -> str:
    if np.isnan(r2):
        return "N/D"
    if r2 < 0.02:
        return "trivial"
    elif r2 < 0.13:
        return "small/pequeño"
    elif r2 < 0.26:
        return "medium/mediano"
    return "large/grande"


def eval_idx(val, umbral, modo="menor"):
    if np.isnan(val):
        return "N/D"
    if modo == "menor":
        return "✅ Good" if val <= umbral else "⚠️ Marginal"
    return "✅ Good" if val >= umbral else "⚠️ Marginal"


# =============================================================================
# 1. CARGA DE DATOS
# =============================================================================

imprimir_seccion("CARGANDO DATOS")
df = pd.read_pickle(ARCHIVO_PROCESADO)
N  = len(df)
print(f"  n = {N}")

VARS_EXOG = [
    "CARGA_TOTAL", "GENERO_MUJER", "ORIGEN_RURAL",
    "ACOSO_EXP", "P8_num", "SEMESTRE_num",
]
VARS_ENDOG = ["IDX_EA", "IDX_FA", "IDX_EQ", "IDX_MV", "IDX_BP"]
orden_endog = VARS_ENDOG

df_sem = df[VARS_EXOG + VARS_ENDOG].dropna().reset_index(drop=True)
n_sem  = len(df_sem)
print(f"  n para SEM (sin NA): {n_sem}")

ETIQ = {
    "CARGA_TOTAL":   "Carga total / Total workload",
    "GENERO_MUJER":  "Género: Mujer / Gender: Woman",
    "ORIGEN_RURAL":  "Origen rural / Rural origin",
    "ACOSO_EXP":     "Exp. acoso / Harassment exp.",
    "P8_num":        "Horas sueño / Sleep hours",
    "SEMESTRE_num":  "Semestre",
    "IDX_EA":        "Estrés Académico / Academic Stress",
    "IDX_FA":        "Fatiga / Fatigue",
    "IDX_EQ":        "Equidad / Equity",
    "IDX_MV":        "Motivación / Motivation",
    "IDX_BP":        "Bienestar / Well-being",
}

# ── Ecuaciones del modelo (definidas aquí, usadas en todo el script)
ecuaciones = {
    "IDX_EA": ["CARGA_TOTAL", "GENERO_MUJER", "ORIGEN_RURAL", "ACOSO_EXP", "P8_num"],
    "IDX_FA": ["CARGA_TOTAL", "GENERO_MUJER", "ORIGEN_RURAL", "ACOSO_EXP", "P8_num"],
    "IDX_EQ": ["GENERO_MUJER", "ORIGEN_RURAL", "ACOSO_EXP"],
    "IDX_MV": ["IDX_EA", "IDX_FA", "IDX_EQ", "GENERO_MUJER", "SEMESTRE_num"],
    "IDX_BP": ["IDX_EA", "IDX_FA", "IDX_MV", "IDX_EQ"],
}

# ── Calcular R² y coeficientes OLS (usados en TAB25, TAB26 y bootstrap)
r2_por_eq = {}
ols_coefs = {}

for endog, predictores in ecuaciones.items():
    datos_ = df_sem[[endog] + predictores].dropna()
    y_     = datos_[endog]
    X_     = sm.add_constant(datos_[predictores])
    mod_   = sm.OLS(y_, X_).fit()
    r2_por_eq[endog] = mod_.rsquared
    sy = y_.std(ddof=1)
    for pred in predictores:
        sx = datos_[pred].std(ddof=1)
        std_b = float(mod_.params[pred]) * sx / sy \
                if sy > 0 and sx > 0 else np.nan
        ols_coefs[(endog, pred)] = {
            "B":    float(mod_.params[pred]),
            "SE":   float(mod_.bse[pred]),
            "t":    float(mod_.tvalues[pred]),
            "p":    float(mod_.pvalues[pred]),
            "beta": std_b,
        }

print(f"\n  R² por ecuación (OLS):")
for k, v in r2_por_eq.items():
    print(f"    {k}: {v:.4f}")


# =============================================================================
# 2. ESPECIFICACIÓN DEL MODELO
# =============================================================================

imprimir_seccion("ESPECIFICACIÓN DEL MODELO SEM")

MODELO_SEM = """
IDX_EA ~ CARGA_TOTAL + GENERO_MUJER + ORIGEN_RURAL + ACOSO_EXP + P8_num
IDX_FA ~ CARGA_TOTAL + GENERO_MUJER + ORIGEN_RURAL + ACOSO_EXP + P8_num
IDX_EQ ~ GENERO_MUJER + ORIGEN_RURAL + ACOSO_EXP
IDX_MV ~ IDX_EA + IDX_FA + IDX_EQ + GENERO_MUJER + SEMESTRE_num
IDX_BP ~ IDX_EA + IDX_FA + IDX_MV + IDX_EQ
IDX_EA ~~ IDX_FA
"""

print("  Ecuaciones del modelo:")
for linea in MODELO_SEM.strip().split("\n"):
    print(f"    {linea}")

guardar_modelo(
    f"ESPECIFICACIÓN DEL MODELO SEM\n{'='*60}\n{MODELO_SEM}\n\n"
    f"Estrategia: Dos pasos (Anderson & Gerbing, 1988)\n"
    f"Paso 1: CFA validado en 02_psicometria.py\n"
    f"Paso 2: Análisis de rutas con índices observados\n"
    f"n = {n_sem}\n",
    "mod_sem_especificacion.txt"
)


# =============================================================================
# 3. AJUSTE DEL MODELO
# =============================================================================

imprimir_seccion("AJUSTE DEL MODELO SEM")

modelo      = semopy.Model(MODELO_SEM)
res_fit     = modelo.fit(df_sem)
params_raw  = modelo.inspect(std_est=True)
stats_sem   = semopy.calc_stats(modelo)

print(f"  Resultado: {res_fit.message if hasattr(res_fit,'message') else res_fit}")
print(f"  Columnas calc_stats: {list(stats_sem.columns)}")

chi2    = extraer_stat(stats_sem, "chi2",         "Chi2")
df_chi2 = extraer_stat(stats_sem, "DoF",          "dof", "df")
p_chi2  = extraer_stat(stats_sem, "chi2 p-value", "p-value")
cfi     = extraer_stat(stats_sem, "CFI",          "cfi")
tli     = extraer_stat(stats_sem, "TLI",          "tli", "NNFI")
rmsea   = extraer_stat(stats_sem, "RMSEA",        "rmsea")
aic     = extraer_stat(stats_sem, "AIC",          "aic")
bic     = extraer_stat(stats_sem, "BIC",          "bic")

# ── SRMR vía residuos OLS
try:
    residuos_matrix = pd.DataFrame(index=df_sem.index)
    for endog, predictores in ecuaciones.items():
        datos_  = df_sem[[endog] + predictores].dropna()
        y_      = datos_[endog]
        X_      = sm.add_constant(datos_[predictores])
        mod_    = sm.OLS(y_, X_).fit()
        resid_  = pd.Series(mod_.resid, index=datos_.index)
        residuos_matrix[endog] = resid_

    residuos_matrix = residuos_matrix.dropna()
    S_obs = df_sem[VARS_ENDOG].loc[residuos_matrix.index].cov().values
    S_res = residuos_matrix.cov().values
    p_    = S_obs.shape[0]
    d_obs = np.sqrt(np.diag(S_obs)); d_obs[d_obs == 0] = 1e-10
    d_res = np.sqrt(np.diag(S_res)); d_res[d_res == 0] = 1e-10
    R_obs = S_obs / np.outer(d_obs, d_obs)
    R_res = S_res / np.outer(d_res, d_res)
    res_  = (R_obs - R_res)[np.tril_indices(p_, k=-1)]
    srmr  = float(np.sqrt(np.mean(res_ ** 2)))
    print(f"  SRMR (vía residuos OLS): {srmr:.4f}")
except Exception as e:
    srmr = np.nan
    print(f"  ⚠️  SRMR no calculado: {e}")

# ── RMSEA IC 90%
try:
    gl_      = int(df_chi2) if not np.isnan(df_chi2) else 1
    rmsea_lo = max(0.0, np.sqrt(max(0, chi2 - gl_) / (gl_ * (n_sem - 1))))
    rmsea_hi = np.sqrt(max(0, chi2 + 1.645 * np.sqrt(2 * gl_)) / (gl_ * (n_sem - 1)))
except Exception:
    rmsea_lo = rmsea_hi = np.nan

# ── Validar rango CFI/TLI
if not (np.isnan(cfi)) and not (0.0 <= cfi <= 1.0):
    print(f"  ⚠️  CFI = {cfi:.3f} fuera de rango — path model casi saturado.")
    print(f"       Se reportará R² y AIC/BIC como criterios principales.")
    cfi = tli = np.nan
else:
    print(f"  CFI = {cfi:.3f}  TLI = {tli:.3f}")

print(f"  RMSEA = {rmsea:.4f} [{rmsea_lo:.4f}, {rmsea_hi:.4f}]")
print(f"  SRMR  = {srmr:.4f}")


# =============================================================================
# 4. TAB25 — ÍNDICES DE AJUSTE
# =============================================================================

imprimir_seccion("TAB25 — ÍNDICES DE AJUSTE")

nombres_eq = {
    "IDX_EA": "Estrés Académico / Academic Stress",
    "IDX_FA": "Fatiga / Fatigue",
    "IDX_EQ": "Equidad / Equity",
    "IDX_MV": "Motivación / Motivation",
    "IDX_BP": "Bienestar / Well-being",
}

lineas = []
lineas.append(encabezado_bilingue(
    "Tabla 25. Índices de ajuste del modelo SEM (análisis de rutas)",
    "Table 25. SEM Path Model Fit Indices"
))
lineas.append(f"\n  n = {n_sem} | Variables endógenas observadas: {len(VARS_ENDOG)}")
lineas.append(
    f"\n  {'Índice / Index':<28} {'Valor / Value':>18} "
    f"{'Umbral':>18} {'Evaluación':>14}"
)
lineas.append("  " + "─" * 82)

for nombre, valor, umbral, evaluacion in [
    ("χ²",           f"{chi2:.3f}"  if not np.isnan(chi2)    else "N/D", "—",       "—"),
    ("gl / df",      f"{int(df_chi2)}" if not np.isnan(df_chi2) else "N/D", "—",    "—"),
    ("p (χ²)",       f"{p_chi2:.4f}" if not np.isnan(p_chi2)  else "N/D", "> .05",  ""),
    ("CFI",          f"{cfi:.3f}"   if not np.isnan(cfi)      else "N/A*", "≥ .95",
     eval_idx(cfi, CFI_MIN, "mayor") if not np.isnan(cfi) else "Ver nota*"),
    ("TLI",          f"{tli:.3f}"   if not np.isnan(tli)      else "N/A*", "≥ .95",
     eval_idx(tli, TLI_MIN, "mayor") if not np.isnan(tli) else "Ver nota*"),
    ("RMSEA",        f"{rmsea:.4f}" if not np.isnan(rmsea)    else "N/D",  "< .08",
     eval_idx(rmsea, RMSEA_MAX, "menor")),
    ("RMSEA IC 90%", f"[{rmsea_lo:.4f}, {rmsea_hi:.4f}]"
     if not np.isnan(rmsea_lo) else "N/D",                                 "LI<.08", ""),
    ("SRMR",         f"{srmr:.4f}"  if not np.isnan(srmr)     else "N/D",  "< .08",
     eval_idx(srmr, SRMR_MAX, "menor")),
    ("AIC",          f"{aic:.2f}"   if not np.isnan(aic)       else "N/D", "—",      "—"),
    ("BIC",          f"{bic:.2f}"   if not np.isnan(bic)       else "N/D", "—",      "—"),
]:
    lineas.append(f"  {nombre:<28} {valor:>18} {umbral:>18} {evaluacion:>14}")

lineas.append(f"\n\n  {'─'*82}")
lineas.append(
    "  R² por ecuación / R² per equation "
    "(criterio principal / main fit criterion)"
)
lineas.append(
    f"\n  {'Ecuación / Equation':<38} {'R²':>8} {'Interpretación / Interpretation':>20}"
)
lineas.append("  " + "─" * 70)
for endog in orden_endog:
    r2v  = r2_por_eq.get(endog, np.nan)
    nom  = nombres_eq.get(endog, endog)
    interp = interpretar_r2(r2v)
    lineas.append(f"  {nom:<38} {r2v:>8.4f} {interp:>20}")

lineas.append(nota_pie(
    "Estrategia de dos pasos: CFA (script 02) + análisis de rutas. "
    "Anderson & Gerbing (1988). "
    "*CFI/TLI reportados como N/A: en modelos de rutas con variables observadas "
    "casi saturados estos índices pueden superar 1 y no son interpretables "
    "(Kline, 2016). Se reporta R² por ecuación como criterio principal.",
    "Two-step strategy: CFA (script 02) + path analysis. "
    "Anderson & Gerbing (1988). "
    "*CFI/TLI reported as N/A: in nearly saturated path models with observed "
    "variables these indices may exceed 1 and are not interpretable "
    "(Kline, 2016). R² per equation is reported as the main fit criterion.",
))
guardar_tabla("\n".join(lineas), "tab25_sem_ajuste.txt")


# =============================================================================
# 5. TAB26 — EFECTOS DIRECTOS
# =============================================================================

imprimir_seccion("EFECTOS DIRECTOS")

lineas = []
lineas.append(encabezado_bilingue(
    "Tabla 26. Efectos directos estandarizados del modelo SEM",
    "Table 26. Standardized Direct Effects from SEM Path Model"
))
lineas.append(
    f"\n  {'Ecuación':<14} {'Predictor':<34} "
    f"{'β':>7} {'SE':>7} {'t':>7} {'p':>8} {'R²':>7}"
)
lineas.append("  " + "─" * 88)

for endog in orden_endog:
    predictores = ecuaciones[endog]
    r2          = r2_por_eq.get(endog, np.nan)
    nom_es      = NOMBRES_INDICES.get(endog, endog)
    nom_en      = NOMBRES_INDICES_EN.get(endog, endog)
    lineas.append(f"\n  {endog} — {nom_es} / {nom_en}  (R² = {r2:.3f})")
    lineas.append("  " + "─" * 88)

    for pred in predictores:
        c     = ols_coefs.get((endog, pred), {})
        beta  = c.get("beta", np.nan)
        se    = c.get("SE",   np.nan)
        t_val = c.get("t",    np.nan)
        p_val = c.get("p",    np.nan)
        sig   = sig_stars(p_val).strip()
        etiq  = ETIQ.get(pred, pred)[:32]
        lineas.append(
            f"  {endog:<14} {etiq:<34} "
            f"{beta:>7.3f} {se:>7.3f} {t_val:>7.3f} "
            f"{p_val:>6.4f}{sig} {r2:>7.3f}"
        )
        print(f"  {endog} ~ {pred}: β={beta:.3f}, p={p_val:.4f}{sig.strip()}")

lineas.append(nota_pie(
    "β = coeficiente estandarizado; SE = error estándar; R² por ecuación.",
    "β = standardized coefficient; SE = standard error; R² per equation. "
    "***p < .001, **p < .01, *p < .05.",
))
guardar_tabla("\n".join(lineas), "tab26_sem_efectos_directos.txt")


# =============================================================================
# 6. TAB27 — EFECTOS INDIRECTOS (BOOTSTRAP)
# =============================================================================

imprimir_seccion(f"EFECTOS INDIRECTOS — BOOTSTRAP (n={N_BOOTSTRAP})")

CONTROLES_M = {
    "IDX_EA": ["CARGA_TOTAL", "GENERO_MUJER", "ORIGEN_RURAL", "ACOSO_EXP", "P8_num"],
    "IDX_FA": ["CARGA_TOTAL", "GENERO_MUJER", "ORIGEN_RURAL", "ACOSO_EXP", "P8_num"],
    "IDX_EQ": ["GENERO_MUJER", "ORIGEN_RURAL", "ACOSO_EXP"],
    "IDX_MV": ["IDX_EA", "IDX_FA", "IDX_EQ", "GENERO_MUJER", "SEMESTRE_num"],
    "IDX_BP": ["IDX_EA", "IDX_FA", "IDX_MV", "IDX_EQ"],
}

MEDIACIONES = [
    ("CARGA_TOTAL",  "IDX_EA", "IDX_BP", "Carga → Estrés → Bienestar",     "Workload → Stress → Well-being"),
    ("CARGA_TOTAL",  "IDX_FA", "IDX_BP", "Carga → Fatiga → Bienestar",      "Workload → Fatigue → Well-being"),
    ("CARGA_TOTAL",  "IDX_EA", "IDX_MV", "Carga → Estrés → Motivación",     "Workload → Stress → Motivation"),
    ("CARGA_TOTAL",  "IDX_FA", "IDX_MV", "Carga → Fatiga → Motivación",     "Workload → Fatigue → Motivation"),
    ("ACOSO_EXP",    "IDX_EA", "IDX_BP", "Acoso → Estrés → Bienestar",      "Harassment → Stress → Well-being"),
    ("ACOSO_EXP",    "IDX_FA", "IDX_BP", "Acoso → Fatiga → Bienestar",      "Harassment → Fatigue → Well-being"),
    ("ACOSO_EXP",    "IDX_EQ", "IDX_BP", "Acoso → Equidad → Bienestar",     "Harassment → Equity → Well-being"),
    ("GENERO_MUJER", "IDX_EA", "IDX_BP", "Género → Estrés → Bienestar",     "Gender → Stress → Well-being"),
    ("GENERO_MUJER", "IDX_FA", "IDX_BP", "Género → Fatiga → Bienestar",     "Gender → Fatigue → Well-being"),
    ("IDX_EA",       "IDX_MV", "IDX_BP", "Estrés → Motivación → Bienestar", "Stress → Motivation → Well-being"),
    ("IDX_FA",       "IDX_MV", "IDX_BP", "Fatiga → Motivación → Bienestar", "Fatigue → Motivation → Well-being"),
]


def calcular_ie(data, x, m, y, controles_m, controles_y):
    preds_m = list(dict.fromkeys([x] + controles_m))
    preds_m = [p for p in preds_m if p in data.columns]
    datos_m = data[[m] + preds_m].dropna()
    sy_m = datos_m[m].std(ddof=1)
    sx   = datos_m[x].std(ddof=1) if x in datos_m.columns else 1.0
    if sy_m == 0 or sx == 0:
        return np.nan
    a = float(sm.OLS(datos_m[m], sm.add_constant(datos_m[preds_m])).fit().params[x]) * sx / sy_m

    preds_y = list(dict.fromkeys([m] + controles_y))
    preds_y = [p for p in preds_y if p in data.columns]
    datos_y = data[[y] + preds_y].dropna()
    sy_y = datos_y[y].std(ddof=1)
    sm_  = datos_y[m].std(ddof=1) if m in datos_y.columns else 1.0
    if sy_y == 0 or sm_ == 0:
        return np.nan
    b = float(sm.OLS(datos_y[y], sm.add_constant(datos_y[preds_y])).fit().params[m]) * sm_ / sy_y

    return a * b


resultados_boot = []
np.random.seed(42)

for x, m, y, desc_es, desc_en in MEDIACIONES:
    controles_m = [c for c in CONTROLES_M.get(m, []) if c != x]
    controles_y = [c for c in CONTROLES_M.get(y, []) if c != m]

    ie_obs   = calcular_ie(df_sem, x, m, y, controles_m, controles_y)
    boot_vals = []
    for _ in range(N_BOOTSTRAP):
        muestra = df_sem.sample(n=n_sem, replace=True)
        ie_b    = calcular_ie(muestra, x, m, y, controles_m, controles_y)
        if not np.isnan(ie_b):
            boot_vals.append(ie_b)

    if len(boot_vals) >= 10:
        ic_lo = np.percentile(boot_vals, 2.5)
        ic_hi = np.percentile(boot_vals, 97.5)
        se_b  = np.std(boot_vals, ddof=1)
        sig   = "✅ Yes" if (ic_lo > 0 or ic_hi < 0) else "⚠️ No"
    else:
        ic_lo = ic_hi = se_b = np.nan
        sig = "N/D"

    resultados_boot.append({
        "x": x, "m": m, "y": y,
        "desc_es": desc_es, "desc_en": desc_en,
        "ie": ie_obs, "se": se_b,
        "ic_lo": ic_lo, "ic_hi": ic_hi,
        "sig": sig,
    })
    print(f"  {x}→{m}→{y}: IE={ie_obs:.3f} IC95%[{ic_lo:.3f},{ic_hi:.3f}] {sig}")

lineas = []
lineas.append(encabezado_bilingue(
    "Tabla 27. Efectos indirectos estandarizados con IC bootstrap",
    "Table 27. Standardized Indirect Effects with Bootstrap CIs"
))
lineas.append(f"\n  Bootstrap: {N_BOOTSTRAP} muestras | IC 95% percentil | Semilla: 42")
lineas.append(
    f"\n  {'Ruta / Path':<44} {'IE':>7} {'SE':>7} "
    f"{'IC lo':>8} {'IC hi':>8} {'Sig.':>8}"
)
lineas.append("  " + "─" * 88)
for r in resultados_boot:
    lineas.append(
        f"  {r['desc_es'][:42]:<44} "
        f"{r['ie']:>7.3f} {r['se']:>7.3f} "
        f"{r['ic_lo']:>8.3f} {r['ic_hi']:>8.3f} {r['sig']:>8}"
    )
    lineas.append(f"  {r['desc_en'][:42]:<44}")
lineas.append(nota_pie(
    f"IE = efecto indirecto estandarizado (a×b). IC bootstrap 95%, {N_BOOTSTRAP} muestras.",
    f"IE = standardized indirect effect (a×b). Bootstrap 95% CI, {N_BOOTSTRAP} samples.",
))
guardar_tabla("\n".join(lineas), "tab27_sem_efectos_indirectos.txt")


# =============================================================================
# 7. TAB28 — EFECTOS TOTALES
# =============================================================================

imprimir_seccion("EFECTOS TOTALES")

pares_totales = [
    ("CARGA_TOTAL",  "IDX_BP", "Carga → Bienestar (total)",    "Workload → Well-being (total)"),
    ("CARGA_TOTAL",  "IDX_MV", "Carga → Motivación (total)",   "Workload → Motivation (total)"),
    ("ACOSO_EXP",    "IDX_BP", "Acoso → Bienestar (total)",    "Harassment → Well-being (total)"),
    ("ACOSO_EXP",    "IDX_MV", "Acoso → Motivación (total)",   "Harassment → Motivation (total)"),
    ("GENERO_MUJER", "IDX_BP", "Género → Bienestar (total)",   "Gender → Well-being (total)"),
    ("IDX_EA",       "IDX_BP", "Estrés → Bienestar (total)",   "Stress → Well-being (total)"),
    ("IDX_FA",       "IDX_BP", "Fatiga → Bienestar (total)",   "Fatigue → Well-being (total)"),
]

lineas = []
lineas.append(encabezado_bilingue(
    "Tabla 28. Efectos totales estandarizados (directos + indirectos)",
    "Table 28. Standardized Total Effects (Direct + Indirect)"
))
lineas.append(
    f"\n  {'Ruta / Path':<46} {'Directo':>9} {'Indirecto':>11} "
    f"{'Total':>8} {'% Med.':>8}"
)
lineas.append("  " + "─" * 86)

for x, y, desc_es, desc_en in pares_totales:
    directo = ols_coefs.get((y, x), {}).get("beta", 0.0)
    if np.isnan(directo):
        directo = 0.0
    ind_sum = sum(
        r["ie"] for r in resultados_boot
        if r["x"] == x and r["y"] == y and not np.isnan(r["ie"])
    )
    total  = directo + ind_sum
    pct_in = abs(ind_sum) / abs(total) * 100 if total != 0 else np.nan

    lineas.append(
        f"  {desc_es[:44]:<46} {directo:>9.3f} {ind_sum:>11.3f} "
        f"{total:>8.3f} "
        + (f"{pct_in:>7.1f}%" if not np.isnan(pct_in) else f"{'N/D':>8}")
    )
    lineas.append(f"  {desc_en[:44]:<46}")
    print(f"  {x}→{y}: dir={directo:.3f}, ind={ind_sum:.3f}, total={total:.3f}")

lineas.append(nota_pie(
    "Efectos totales = directos + suma de indirectos vía mediadores. "
    "% Med. = porcentaje del efecto total explicado por mediación.",
    "Total effects = direct + sum of indirect through mediators. "
    "% Med. = percentage of total effect explained by mediation.",
))
guardar_tabla("\n".join(lineas), "tab28_sem_efectos_totales.txt")


# =============================================================================
# 8. OUTPUT COMPLETO
# =============================================================================

imprimir_seccion("GUARDANDO OUTPUT COMPLETO")

output = [
    "=" * 70,
    "SEM PATH MODEL — OUTPUT COMPLETO / FULL OUTPUT",
    "ITESCAM 2025 — Bienestar Académico Estudiantil",
    "=" * 70,
    f"\nModelo:\n{MODELO_SEM}",
    f"\nn = {n_sem}",
    f"\nÍndices de ajuste:",
    f"  CFI  = {cfi:.3f}" if not np.isnan(cfi) else "  CFI  = N/A (path model casi saturado)",
    f"  TLI  = {tli:.3f}" if not np.isnan(tli) else "  TLI  = N/A (path model casi saturado)",
    f"  RMSEA = {rmsea:.4f} [{rmsea_lo:.4f}, {rmsea_hi:.4f}]",
    f"  SRMR  = {srmr:.4f}" if not np.isnan(srmr) else "  SRMR  = N/D",
    f"  AIC   = {aic:.2f}  BIC = {bic:.2f}",
    f"\nR² por ecuación:",
]
for k, v in r2_por_eq.items():
    output.append(f"  {k}: {v:.4f}")
output.append(f"\nParámetros estimados:\n{params_raw.to_string()}")
output.append(f"\nEfectos indirectos (bootstrap):")
for r in resultados_boot:
    output.append(
        f"  {r['x']}→{r['m']}→{r['y']}: "
        f"IE={r['ie']:.3f}, IC95%[{r['ic_lo']:.3f},{r['ic_hi']:.3f}], {r['sig']}"
    )

guardar_modelo("\n".join(output), "mod_sem_output_completo.txt")


# =============================================================================
# RESUMEN FINAL
# =============================================================================

imprimir_seccion("RESUMEN — SEM COMPLETADO")

archivos = [
    ("tab25_sem_ajuste.txt",             TABLAS_DIR),
    ("tab26_sem_efectos_directos.txt",   TABLAS_DIR),
    ("tab27_sem_efectos_indirectos.txt", TABLAS_DIR),
    ("tab28_sem_efectos_totales.txt",    TABLAS_DIR),
    ("mod_sem_especificacion.txt",       MODELOS_DIR),
    ("mod_sem_output_completo.txt",      MODELOS_DIR),
]
for archivo, carpeta in archivos:
    ruta   = carpeta / archivo
    estado = "✅" if ruta.exists() else "❌"
    print(f"  {estado} {archivo}")

print(f"\n  Índices de ajuste finales:")
print(f"    CFI   = {cfi:.3f}"   if not np.isnan(cfi)   else "    CFI   = N/A (path model)")
print(f"    TLI   = {tli:.3f}"   if not np.isnan(tli)   else "    TLI   = N/A (path model)")
print(f"    RMSEA = {rmsea:.4f}  (umbral < .08)")
print(f"    SRMR  = {srmr:.4f}"  if not np.isnan(srmr)  else "    SRMR  = N/D")
print(f"    AIC   = {aic:.2f}"   if not np.isnan(aic)   else "    AIC   = N/D")
print(f"\n  R² por ecuación:")
for k, v in r2_por_eq.items():
    print(f"    {k}: {v:.4f}  ({interpretar_r2(v)})")