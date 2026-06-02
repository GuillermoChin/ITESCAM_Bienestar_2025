"""
05_sem.py
Modelado de Ecuaciones Estructurales — Análisis de rutas con índices observados.

Estrategia de dos pasos (Anderson & Gerbing, 1988):
  Paso 1 — CFA validó el modelo de medición (script 02)
  Paso 2 — Análisis de rutas con índices como variables observadas (este script)

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


def beta_std_ols(y: pd.Series, X: pd.DataFrame) -> dict:
    """Coeficientes beta estandarizados vía OLS."""
    resultado = sm.OLS(y, sm.add_constant(X)).fit()
    betas = {}
    sy = y.std(ddof=1)
    for col in X.columns:
        sx = X[col].std(ddof=1)
        if sx > 0 and sy > 0:
            betas[col] = float(resultado.params[col]) * sx / sy
        else:
            betas[col] = np.nan
    return betas, resultado


# =============================================================================
# CARGA DE DATOS
# =============================================================================

imprimir_seccion("CARGANDO DATOS")
df = pd.read_pickle(ARCHIVO_PROCESADO)
N  = len(df)
print(f"  n = {N}")

# Variables observadas para el modelo de rutas
VARS_EXOG = [
    "CARGA_TOTAL",
    "GENERO_MUJER",
    "ORIGEN_RURAL",
    "ACOSO_EXP",
    "P8_num",
    "SEMESTRE_num",
]

VARS_ENDOG = ["IDX_EA", "IDX_FA", "IDX_EQ", "IDX_MV", "IDX_BP"]

todas_vars = VARS_EXOG + VARS_ENDOG
df_sem = df[todas_vars].dropna().reset_index(drop=True)
n_sem  = len(df_sem)
print(f"  n para SEM (sin NA): {n_sem}")

# Etiquetas bilingües
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

# =============================================================================
# ESPECIFICACIÓN DEL MODELO
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

# Guardar especificación
guardar_modelo(
    f"ESPECIFICACIÓN DEL MODELO SEM\n{'='*60}\n{MODELO_SEM}\n\n"
    f"Estrategia: Dos pasos (Anderson & Gerbing, 1988)\n"
    f"Paso 1: CFA validado en 02_psicometria.py\n"
    f"Paso 2: Análisis de rutas con índices observados\n"
    f"n = {n_sem}\n",
    "mod_sem_especificacion.txt"
)

# =============================================================================
# AJUSTE DEL MODELO
# =============================================================================

imprimir_seccion("AJUSTE DEL MODELO SEM")

modelo = semopy.Model(MODELO_SEM)
resultado_fit = modelo.fit(df_sem)
params_raw    = modelo.inspect(std_est=True)
stats_sem     = semopy.calc_stats(modelo)

print(f"  Resultado: {resultado_fit.message if hasattr(resultado_fit, 'message') else resultado_fit}")
print(f"  Columnas calc_stats: {list(stats_sem.columns) if hasattr(stats_sem, 'columns') else list(stats_sem.index)}")

# Extraer índices de ajuste
chi2    = extraer_stat(stats_sem, "chi2",         "Chi2",        "Chi-square")
df_chi2 = extraer_stat(stats_sem, "DoF",          "dof",         "df",         "Df")
p_chi2  = extraer_stat(stats_sem, "chi2 p-value", "p-value",     "pvalue")
cfi     = extraer_stat(stats_sem, "CFI",          "cfi")
tli     = extraer_stat(stats_sem, "TLI",          "tli",         "NNFI")
rmsea   = extraer_stat(stats_sem, "RMSEA",        "rmsea")
aic     = extraer_stat(stats_sem, "AIC",          "aic")
bic     = extraer_stat(stats_sem, "BIC",          "bic")

# SRMR manual
try:
    S_obs = df_sem[VARS_ENDOG].cov().values
    pred_ = modelo.predict(df_sem)
    if hasattr(pred_, "values"):
        pred_ = pred_.values
    S_mod = np.cov(pred_.T)
    p_    = S_obs.shape[0]
    d_obs = np.sqrt(np.diag(S_obs))
    d_mod = np.sqrt(np.diag(S_mod))
    d_obs[d_obs == 0] = 1e-10
    d_mod[d_mod == 0] = 1e-10
    R_obs = S_obs / np.outer(d_obs, d_obs)
    R_mod = S_mod / np.outer(d_mod, d_mod)
    res_  = (R_obs - R_mod)[np.tril_indices(p_, k=-1)]
    srmr  = float(np.sqrt(np.mean(res_ ** 2)))
except Exception as e:
    srmr = np.nan
    print(f"  ⚠️  SRMR no calculado: {e}")

# RMSEA IC 90%
try:
    gl_   = int(df_chi2) if not np.isnan(df_chi2) else 1
    rmsea_lo = max(0.0, np.sqrt(max(0, chi2 - gl_) / (gl_ * (n_sem - 1))))
    rmsea_hi = np.sqrt(max(0, chi2 + 1.645 * np.sqrt(2 * gl_)) / (gl_ * (n_sem - 1)))
except Exception:
    rmsea_lo = rmsea_hi = np.nan

print(f"\n  CFI = {cfi:.3f}  TLI = {tli:.3f}")
print(f"  RMSEA = {rmsea:.4f} [{rmsea_lo:.4f}, {rmsea_hi:.4f}]")
print(f"  SRMR = {srmr:.4f}")

def eval_idx(val, umbral, modo="menor"):
    if np.isnan(val):
        return "N/D"
    if modo == "menor":
        return "✅ Acceptable" if val <= umbral else "⚠️ Marginal"
    return "✅ Acceptable" if val >= umbral else "⚠️ Marginal"

# TAB25
lineas = []
lineas.append(encabezado_bilingue(
    "Tabla 25. Índices de ajuste del modelo SEM (análisis de rutas)",
    "Table 25. SEM Path Model Fit Indices"
))
lineas.append(f"\n  Modelo: análisis de rutas con 5 variables endógenas observadas")
lineas.append(f"  Model: path analysis with 5 observed endogenous variables")
lineas.append(f"  n = {n_sem}")
lineas.append(
    f"\n  {'Índice / Index':<28} {'Valor / Value':>16} "
    f"{'Umbral / Threshold':>20} {'Evaluación':>14}"
)
lineas.append("  " + "─" * 82)

indices_tabla = [
    ("χ² / Chi-square",
     f"{chi2:.3f}" if not np.isnan(chi2) else "N/D", "—", "—"),
    ("gl / df",
     f"{int(df_chi2)}" if not np.isnan(df_chi2) else "N/D", "—", "—"),
    ("p (χ²)",
     f"{p_chi2:.4f}" if not np.isnan(p_chi2) else "N/D", "> .05", ""),
    ("CFI",
     f"{cfi:.3f}" if not np.isnan(cfi) else "N/D",
     "≥ .95", eval_idx(cfi, CFI_MIN, "mayor")),
    ("TLI",
     f"{tli:.3f}" if not np.isnan(tli) else "N/D",
     "≥ .95", eval_idx(tli, TLI_MIN, "mayor")),
    ("RMSEA",
     f"{rmsea:.4f}" if not np.isnan(rmsea) else "N/D",
     "< .08", eval_idx(rmsea, RMSEA_MAX, "menor")),
    ("RMSEA IC 90% / 90% CI",
     f"[{rmsea_lo:.4f}, {rmsea_hi:.4f}]" if not np.isnan(rmsea_lo) else "N/D",
     "LS < .08", ""),
    ("SRMR",
     f"{srmr:.4f}" if not np.isnan(srmr) else "N/D",
     "< .08", eval_idx(srmr, SRMR_MAX, "menor")),
    ("AIC",
     f"{aic:.2f}" if not np.isnan(aic) else "N/D", "—", "—"),
    ("BIC",
     f"{bic:.2f}" if not np.isnan(bic) else "N/D", "—", "—"),
]

for nombre, valor, umbral, evaluacion in indices_tabla:
    lineas.append(
        f"  {nombre:<28} {valor:>16} {umbral:>20} {evaluacion:>14}"
    )

lineas.append(nota_pie(
    "Estrategia de dos pasos: CFA (script 02) + análisis de rutas (este script). "
    "Anderson & Gerbing (1988).",
    "Two-step strategy: CFA (script 02) + path analysis (this script). "
    "Anderson & Gerbing (1988).",
))
guardar_tabla("\n".join(lineas), "tab25_sem_ajuste.txt")


# =============================================================================
# EFECTOS DIRECTOS (TAB26)
# =============================================================================

imprimir_seccion("EFECTOS DIRECTOS")

# Filtrar rutas estructurales de params_raw
# Semopy: lval = variable dependiente, op = ~, rval = predictor
rutas = params_raw[
    (params_raw["op"] == "~") &
    (~params_raw["lval"].str.startswith("IDX_EA~~", na=True))
].copy()

# Calcular R² por ecuación vía OLS para complementar
r2_por_eq = {}
ecuaciones = {
    "IDX_EA": ["CARGA_TOTAL", "GENERO_MUJER", "ORIGEN_RURAL", "ACOSO_EXP", "P8_num"],
    "IDX_FA": ["CARGA_TOTAL", "GENERO_MUJER", "ORIGEN_RURAL", "ACOSO_EXP", "P8_num"],
    "IDX_EQ": ["GENERO_MUJER", "ORIGEN_RURAL", "ACOSO_EXP"],
    "IDX_MV": ["IDX_EA", "IDX_FA", "IDX_EQ", "GENERO_MUJER", "SEMESTRE_num"],
    "IDX_BP": ["IDX_EA", "IDX_FA", "IDX_MV", "IDX_EQ"],
}

ols_coefs = {}  # guardamos coeficientes OLS para bootstrap

for endog, predictores in ecuaciones.items():
    datos_ = df_sem[[endog] + predictores].dropna()
    y_     = datos_[endog]
    X_     = sm.add_constant(datos_[predictores])
    mod_   = sm.OLS(y_, X_).fit()
    r2_por_eq[endog] = mod_.rsquared

    # Guardar coefs estandarizados
    sy = y_.std(ddof=1)
    for pred in predictores:
        sx = datos_[pred].std(ddof=1)
        std_b = float(mod_.params[pred]) * sx / sy if sy > 0 and sx > 0 else np.nan
        ols_coefs[(endog, pred)] = {
            "B":    float(mod_.params[pred]),
            "SE":   float(mod_.bse[pred]),
            "t":    float(mod_.tvalues[pred]),
            "p":    float(mod_.pvalues[pred]),
            "beta": std_b,
        }

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

orden_endog = ["IDX_EA", "IDX_FA", "IDX_EQ", "IDX_MV", "IDX_BP"]

for endog in orden_endog:
    predictores = ecuaciones[endog]
    r2 = r2_por_eq.get(endog, np.nan)
    nom_es = NOMBRES_INDICES.get(endog, endog)
    nom_en = NOMBRES_INDICES_EN.get(endog, endog)
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
    "β = coeficiente estandarizado; SE = error estándar del coeficiente no estandarizado; "
    "R² reportado por ecuación.",
    "β = standardized coefficient; SE = standard error of unstandardized coefficient; "
    "R² reported per equation. ***p < .001, **p < .01, *p < .05.",
))
guardar_tabla("\n".join(lineas), "tab26_sem_efectos_directos.txt")


# =============================================================================
# EFECTOS INDIRECTOS CON BOOTSTRAP (TAB27)
# =============================================================================

imprimir_seccion(f"EFECTOS INDIRECTOS — BOOTSTRAP (n={N_BOOTSTRAP})")

# Mediaciones a probar
MEDIACIONES = [
    # (X, M, Y, descripción ES, descripción EN)
    ("CARGA_TOTAL", "IDX_EA", "IDX_BP",
     "Carga → Estrés → Bienestar",
     "Workload → Stress → Well-being"),
    ("CARGA_TOTAL", "IDX_FA", "IDX_BP",
     "Carga → Fatiga → Bienestar",
     "Workload → Fatigue → Well-being"),
    ("CARGA_TOTAL", "IDX_EA", "IDX_MV",
     "Carga → Estrés → Motivación",
     "Workload → Stress → Motivation"),
    ("CARGA_TOTAL", "IDX_FA", "IDX_MV",
     "Carga → Fatiga → Motivación",
     "Workload → Fatigue → Motivation"),
    ("ACOSO_EXP",   "IDX_EA", "IDX_BP",
     "Acoso → Estrés → Bienestar",
     "Harassment → Stress → Well-being"),
    ("ACOSO_EXP",   "IDX_FA", "IDX_BP",
     "Acoso → Fatiga → Bienestar",
     "Harassment → Fatigue → Well-being"),
    ("ACOSO_EXP",   "IDX_EQ", "IDX_BP",
     "Acoso → Equidad → Bienestar",
     "Harassment → Equity → Well-being"),
    ("GENERO_MUJER","IDX_EA", "IDX_BP",
     "Género → Estrés → Bienestar",
     "Gender → Stress → Well-being"),
    ("GENERO_MUJER","IDX_FA", "IDX_BP",
     "Género → Fatiga → Bienestar",
     "Gender → Fatigue → Well-being"),
    ("IDX_EA",      "IDX_MV", "IDX_BP",
     "Estrés → Motivación → Bienestar",
     "Stress → Motivation → Well-being"),
    ("IDX_FA",      "IDX_MV", "IDX_BP",
     "Fatiga → Motivación → Bienestar",
     "Fatigue → Motivation → Well-being"),
]


def calcular_efecto_indirecto(data: pd.DataFrame, x: str, m: str, y: str,
                               controles_m: list, controles_y: list) -> float:
    """
    Calcula el efecto indirecto estandarizado X → M → Y.
    controles_m: covariables en la ecuación de M
    controles_y: covariables en la ecuación de Y
    """
    # Ruta a: X → M (controlando covariables de M)
    preds_m = list(dict.fromkeys([x] + controles_m))
    preds_m = [p for p in preds_m if p in data.columns]
    datos_m = data[[m] + preds_m].dropna()
    sy_m    = datos_m[m].std(ddof=1)
    sx      = datos_m[x].std(ddof=1) if x in datos_m.columns else 1.0
    if sy_m == 0 or sx == 0:
        return np.nan
    mod_m = sm.OLS(datos_m[m], sm.add_constant(datos_m[preds_m])).fit()
    a_std = float(mod_m.params[x]) * sx / sy_m

    # Ruta b: M → Y (controlando covariables de Y)
    preds_y = list(dict.fromkeys([m] + controles_y))
    preds_y = [p for p in preds_y if p in data.columns]
    datos_y = data[[y] + preds_y].dropna()
    sy_y    = datos_y[y].std(ddof=1)
    sm_val  = datos_y[m].std(ddof=1) if m in datos_y.columns else 1.0
    if sy_y == 0 or sm_val == 0:
        return np.nan
    mod_y = sm.OLS(datos_y[y], sm.add_constant(datos_y[preds_y])).fit()
    b_std = float(mod_y.params[m]) * sm_val / sy_y

    return a_std * b_std


# Definir covariables para cada ecuación
CONTROLES_M = {
    "IDX_EA": ["CARGA_TOTAL", "GENERO_MUJER", "ORIGEN_RURAL", "ACOSO_EXP", "P8_num"],
    "IDX_FA": ["CARGA_TOTAL", "GENERO_MUJER", "ORIGEN_RURAL", "ACOSO_EXP", "P8_num"],
    "IDX_EQ": ["GENERO_MUJER", "ORIGEN_RURAL", "ACOSO_EXP"],
    "IDX_MV": ["IDX_EA", "IDX_FA", "IDX_EQ", "GENERO_MUJER", "SEMESTRE_num"],
    "IDX_BP": ["IDX_EA", "IDX_FA", "IDX_MV", "IDX_EQ"],
}

resultados_boot = []
np.random.seed(42)

for x, m, y, desc_es, desc_en in MEDIACIONES:
    controles_m = [c for c in CONTROLES_M.get(m, []) if c != x]
    controles_y = [c for c in CONTROLES_M.get(y, []) if c != m]

    # Efecto puntual
    ie_obs = calcular_efecto_indirecto(df_sem, x, m, y, controles_m, controles_y)

    # Bootstrap
    boot_vals = []
    for _ in range(N_BOOTSTRAP):
        muestra = df_sem.sample(n=n_sem, replace=True)
        ie_b    = calcular_efecto_indirecto(muestra, x, m, y, controles_m, controles_y)
        if not np.isnan(ie_b):
            boot_vals.append(ie_b)

    if len(boot_vals) >= 10:
        ic_lo = np.percentile(boot_vals, 2.5)
        ic_hi = np.percentile(boot_vals, 97.5)
        se_b  = np.std(boot_vals, ddof=1)
        sig   = "✅ Sí/Yes" if (ic_lo > 0 or ic_hi < 0) else "⚠️ No"
    else:
        ic_lo = ic_hi = se_b = np.nan
        sig = "N/D"

    resultados_boot.append({
        "x": x, "m": m, "y": y,
        "desc_es": desc_es, "desc_en": desc_en,
        "ie": ie_obs, "se": se_b,
        "ic_lo": ic_lo, "ic_hi": ic_hi,
        "significativo": sig,
    })

    print(
        f"  {x}→{m}→{y}: IE={ie_obs:.3f} "
        f"IC95%[{ic_lo:.3f},{ic_hi:.3f}] {sig}"
    )

lineas = []
lineas.append(encabezado_bilingue(
    "Tabla 27. Efectos indirectos estandarizados con intervalos de confianza bootstrap",
    "Table 27. Standardized Indirect Effects with Bootstrap Confidence Intervals"
))
lineas.append(
    f"\n  Bootstrap: {N_BOOTSTRAP} muestras | Semilla: 42 | IC: 95% percentil"
)
lineas.append(
    f"\n  {'Ruta / Path':<42} {'IE':>7} {'SE':>7} "
    f"{'IC lo':>8} {'IC hi':>8} {'Sig.':>10}"
)
lineas.append("  " + "─" * 88)

for r in resultados_boot:
    lineas.append(
        f"  {r['desc_es'][:40]:<42} "
        f"{r['ie']:>7.3f} {r['se']:>7.3f} "
        f"{r['ic_lo']:>8.3f} {r['ic_hi']:>8.3f} {r['significativo']:>10}"
    )
    lineas.append(f"  {r['desc_en'][:40]:<42}")

lineas.append(nota_pie(
    "IE = efecto indirecto estandarizado (producto de rutas a × b). "
    f"IC bootstrap al 95% con {N_BOOTSTRAP} muestras. "
    "Significativo si el IC no incluye el cero.",
    "IE = standardized indirect effect (product of paths a × b). "
    f"Bootstrap 95% CI with {N_BOOTSTRAP} samples. "
    "Significant if CI does not include zero.",
))
guardar_tabla("\n".join(lineas), "tab27_sem_efectos_indirectos.txt")


# =============================================================================
# EFECTOS TOTALES (TAB28)
# =============================================================================

imprimir_seccion("EFECTOS TOTALES")

# Efecto total = efecto directo + suma de efectos indirectos
# Para cada par X → Y relevante para la teoría

pares_totales = [
    # (X, Y, descripción ES, descripción EN)
    ("CARGA_TOTAL", "IDX_BP",
     "Carga → Bienestar (total)",
     "Workload → Well-being (total)"),
    ("CARGA_TOTAL", "IDX_MV",
     "Carga → Motivación (total)",
     "Workload → Motivation (total)"),
    ("ACOSO_EXP",   "IDX_BP",
     "Acoso → Bienestar (total)",
     "Harassment → Well-being (total)"),
    ("ACOSO_EXP",   "IDX_MV",
     "Acoso → Motivación (total)",
     "Harassment → Motivation (total)"),
    ("GENERO_MUJER","IDX_BP",
     "Género → Bienestar (total)",
     "Gender → Well-being (total)"),
    ("IDX_EA",      "IDX_BP",
     "Estrés → Bienestar (total)",
     "Stress → Well-being (total)"),
    ("IDX_FA",      "IDX_BP",
     "Fatiga → Bienestar (total)",
     "Fatigue → Well-being (total)"),
]

lineas = []
lineas.append(encabezado_bilingue(
    "Tabla 28. Efectos totales estandarizados (directos + indirectos)",
    "Table 28. Standardized Total Effects (Direct + Indirect)"
))
lineas.append(
    f"\n  {'Ruta / Path':<44} {'Directo':>9} {'Indirecto':>11} "
    f"{'Total':>8} {'% Ind.':>8}"
)
lineas.append("  " + "─" * 84)

for x, y, desc_es, desc_en in pares_totales:
    # Efecto directo
    directo = ols_coefs.get((y, x), {}).get("beta", 0.0)
    if np.isnan(directo):
        directo = 0.0

    # Suma de efectos indirectos
    ind_sum = sum(
        r["ie"] for r in resultados_boot
        if r["x"] == x and r["y"] == y and not np.isnan(r["ie"])
    )

    total  = directo + ind_sum
    pct_in = (abs(ind_sum) / abs(total) * 100) if total != 0 else np.nan

    lineas.append(
        f"  {desc_es[:42]:<44} {directo:>9.3f} {ind_sum:>11.3f} "
        f"{total:>8.3f} "
        f"{pct_in:>7.1f}%" if not np.isnan(pct_in) else
        f"  {desc_es[:42]:<44} {directo:>9.3f} {ind_sum:>11.3f} "
        f"{total:>8.3f} {'N/D':>8}"
    )
    lineas.append(f"  {desc_en[:42]:<44}")
    print(
        f"  {x}→{y}: dir={directo:.3f}, "
        f"ind={ind_sum:.3f}, total={total:.3f}"
    )

lineas.append(nota_pie(
    "Efectos totales = efectos directos + suma de efectos indirectos vía mediadores. "
    "% Ind. = porcentaje del efecto total explicado por la mediación.",
    "Total effects = direct effects + sum of indirect effects through mediators. "
    "% Ind. = percentage of total effect explained by mediation.",
))
guardar_tabla("\n".join(lineas), "tab28_sem_efectos_totales.txt")


# =============================================================================
# GUARDAR OUTPUT COMPLETO
# =============================================================================

imprimir_seccion("GUARDANDO OUTPUT COMPLETO")

output = []
output.append("=" * 70)
output.append("SEM PATH MODEL — OUTPUT COMPLETO / FULL OUTPUT")
output.append("ITESCAM 2025 — Bienestar Académico Estudiantil")
output.append("=" * 70)
output.append(f"\nModelo / Model:\n{MODELO_SEM}")
output.append(f"\nn = {n_sem}")
output.append(f"\nResultado de ajuste:\n{resultado_fit}")
output.append(f"\nÍndices de ajuste / Fit indices:")
output.append(f"  CFI = {cfi:.3f} | TLI = {tli:.3f}")
output.append(f"  RMSEA = {rmsea:.4f} [{rmsea_lo:.4f}, {rmsea_hi:.4f}]")
output.append(f"  SRMR = {srmr:.4f}")
output.append(f"  AIC = {aic:.2f} | BIC = {bic:.2f}")
output.append(f"\nParámetros estimados / Parameters:\n{params_raw.to_string()}")
output.append(f"\nR² por ecuación / R² per equation:")
for k, v in r2_por_eq.items():
    output.append(f"  {k}: R² = {v:.4f}")
output.append(f"\nEfectos indirectos / Indirect effects:")
for r in resultados_boot:
    output.append(
        f"  {r['x']}→{r['m']}→{r['y']}: "
        f"IE={r['ie']:.3f}, IC95%[{r['ic_lo']:.3f},{r['ic_hi']:.3f}]"
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
print(f"    CFI   = {cfi:.3f}   (umbral ≥ .95)")
print(f"    TLI   = {tli:.3f}   (umbral ≥ .95)")
print(f"    RMSEA = {rmsea:.4f}  (umbral < .08)")
print(f"    SRMR  = {srmr:.4f}  (umbral < .08)")