"""
02_psicometria.py
Análisis de fiabilidad, correlaciones ítem-total y AFC.

Entrada:  03_salidas/modelos/datos_procesados.pkl
Salidas:  03_salidas/tablas/tab07_fiabilidad.txt
          03_salidas/tablas/tab08_correlaciones_item_total.txt
          03_salidas/tablas/tab09_cfa_ajuste.txt
          03_salidas/tablas/tab10_cargas_factoriales.txt
          03_salidas/tablas/tab11_validez_convergente.txt
          03_salidas/modelos/mod_cfa_output_completo.txt
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from scipy import stats

sys.path.append(str(__import__("pathlib").Path(__file__).parent))
from config import (
    ARCHIVO_PROCESADO, TABLAS_DIR, MODELOS_DIR,
    EA_COLS, FA_COLS, MV_COLS, EQ_COLS, BP_COLS,
    INDICES, NOMBRES_INDICES, NOMBRES_INDICES_EN,
    ALPHA_CRONBACH_MIN, ALPHA_CRONBACH_OK,
    CFI_MIN, TLI_MIN, RMSEA_MAX, SRMR_MAX, AVE_MIN,
    NOTA_MUESTRA, NOTA_MUESTRA_EN,
)

# Verificar semopy
try:
    import semopy
    SEMOPY_OK = True
    print("  ✅ semopy disponible")
except ImportError:
    SEMOPY_OK = False
    print("  ⚠️  semopy no encontrado — instalar con: pip install semopy")
    print("       Las secciones de CFA se omitirán")

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


def encabezado_bilingue(titulo_es, titulo_en, ancho=76):
    return "\n".join([
        "=" * ancho,
        f"  {titulo_es}",
        f"  {titulo_en}",
        "=" * ancho,
    ])


def nota_pie(extra_es="", extra_en=""):
    n_str = "n = 555 estudiantes. ITESCAM, Campeche, México (2026)."
    lineas = [
        f"\n  Nota. {n_str}",
        f"  Note. {n_str.replace('estudiantes', 'students')}",
    ]
    if extra_es:
        lineas.append(f"  {extra_es}")
    if extra_en:
        lineas.append(f"  {extra_en}")
    return "\n".join(lineas) + "\n"


def cronbach_alpha(data: pd.DataFrame) -> float:
    """Alpha de Cronbach estándar."""
    n = data.shape[1]
    if n < 2:
        return np.nan
    var_total = data.sum(axis=1).var(ddof=1)
    if var_total == 0:
        return np.nan
    var_items = data.var(axis=0, ddof=1).sum()
    return (n / (n - 1)) * (1 - var_items / var_total)


def alpha_si_elimina(data: pd.DataFrame) -> dict:
    """Alpha si se elimina cada ítem."""
    resultados = {}
    for col in data.columns:
        sub = data.drop(columns=[col])
        resultados[col] = cronbach_alpha(sub)
    return resultados


def r_it_corregida(data: pd.DataFrame, col: str) -> float:
    """Correlación ítem-total corregida (excluye el ítem del total)."""
    resto = data.drop(columns=[col]).sum(axis=1)
    r, _ = stats.pearsonr(data[col], resto)
    return r


def calcular_ave(lambdas: np.ndarray) -> float:
    """Average Variance Extracted."""
    return float(np.mean(lambdas ** 2))


def calcular_cr(lambdas: np.ndarray) -> float:
    """Composite Reliability (fiabilidad compuesta)."""
    suma_l = np.sum(lambdas)
    suma_err = np.sum(1 - lambdas ** 2)
    return float(suma_l ** 2 / (suma_l ** 2 + suma_err))


def calcular_srmr(S_obs: np.ndarray, S_mod: np.ndarray) -> float:
    """Standardized Root Mean Square Residual."""
    p = S_obs.shape[0]
    d_obs = np.sqrt(np.diag(S_obs))
    d_mod = np.sqrt(np.diag(S_mod))
    # Evitar división por cero
    d_obs[d_obs == 0] = 1e-10
    d_mod[d_mod == 0] = 1e-10
    R_obs = S_obs / np.outer(d_obs, d_obs)
    R_mod = S_mod / np.outer(d_mod, d_mod)
    residuos = (R_obs - R_mod)[np.tril_indices(p, k=-1)]
    return float(np.sqrt(np.mean(residuos ** 2)))


def to_float(val) -> float:
    """Convierte cualquier valor a float de forma segura."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return np.nan


def sig_stars(p) -> str:
    """Devuelve asteriscos de significancia, tolerando tipos no estándar."""
    try:
        p = float(p)
        if np.isnan(p):
            return ""
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        return ""
    except (TypeError, ValueError):
        return ""


# =============================================================================
# CARGA DE DATOS
# =============================================================================

imprimir_seccion("CARGANDO DATOS")
df = pd.read_pickle(ARCHIVO_PROCESADO)
N = len(df)
print(f"  n = {N}")

# Definición de columnas numéricas por escala
ESCALAS = {
    "IDX_EA": {"cols_num": [c + "_num"   for c in EA_COLS],
               "cols_raw": EA_COLS, "n_items": 8, "escala": "frecuencia"},
    "IDX_FA": {"cols_num": [c + "_num"   for c in FA_COLS],
               "cols_raw": FA_COLS, "n_items": 8, "escala": "frecuencia"},
    "IDX_MV": {"cols_num": [c + "_num"   for c in MV_COLS],
               "cols_raw": MV_COLS, "n_items": 8, "escala": "acuerdo"},
    "IDX_EQ": {"cols_num": [c + "_final" for c in EQ_COLS],
               "cols_raw": EQ_COLS, "n_items": 8, "escala": "acuerdo"},
    "IDX_BP": {"cols_num": [c + "_num"   for c in BP_COLS],
               "cols_raw": BP_COLS, "n_items": 5, "escala": "acuerdo"},
}

# Etiquetas cortas de ítems para tablas
ITEM_CORTO = {}
for idx, info in ESCALAS.items():
    prefijo = idx.replace("IDX_", "")
    for i, col in enumerate(info["cols_raw"]):
        ITEM_CORTO[col] = f"{prefijo}{i+1}"


# =============================================================================
# SECCIÓN 1 — FIABILIDAD (TAB07)
# =============================================================================

imprimir_seccion("1. FIABILIDAD — ALPHA DE CRONBACH")

resultados_alpha = {}

for idx, info in ESCALAS.items():
    cols = info["cols_num"]
    data = df[cols].dropna()
    alpha = cronbach_alpha(data)
    alpha_elim = alpha_si_elimina(data)
    resultados_alpha[idx] = {
        "alpha": alpha,
        "n_items": info["n_items"],
        "alpha_si_elimina": alpha_elim,
        "cols_num": cols,
        "cols_raw": info["cols_raw"],
    }

    estado = "✅" if alpha >= ALPHA_CRONBACH_MIN else "🔴"
    print(f"  {estado} {idx}: α = {alpha:.3f}  ({info['n_items']} ítems)")

# Guardar TAB07
lineas = []
lineas.append(encabezado_bilingue(
    "Tabla 7. Fiabilidad de las escalas (Alpha de Cronbach)",
    "Table 7. Scale Reliability (Cronbach's Alpha)"
))

lineas.append(
    f"\n  {'Escala / Scale':<32} {'ES':<26} {'EN':<26} "
    f"{'k':>3} {'α':>7} {'Estado/Status':>12}"
)
lineas.append("  " + "─" * 90)

interpretacion = {
    "IDX_EA": ("Estrés Académico",    "Academic Stress"),
    "IDX_FA": ("Fatiga/Agotamiento",  "Fatigue/Burnout"),
    "IDX_MV": ("Motivación",          "Motivation"),
    "IDX_EQ": ("Percepción Equidad",  "Perceived Equity"),
    "IDX_BP": ("Bienestar Percibido", "Perceived Well-being"),
}

for idx, datos in resultados_alpha.items():
    alpha = datos["alpha"]
    k     = datos["n_items"]
    es, en = interpretacion[idx]
    if alpha >= ALPHA_CRONBACH_OK:
        status = "Bueno / Good"
    elif alpha >= ALPHA_CRONBACH_MIN:
        status = "Aceptable"
    else:
        status = "Bajo / Low"
    lineas.append(
        f"  {idx:<32} {es:<26} {en:<26} "
        f"{k:>3} {alpha:>7.3f} {status:>12}"
    )

lineas.append(f"\n  {'─'*90}")
lineas.append("  Alpha si se elimina el ítem / Alpha if item deleted")
lineas.append(
    f"\n  {'Ítem':<8} {'Descripción (ES)':<46} {'α sin ítem':>10} {'Δα':>8}"
)
lineas.append("  " + "─" * 76)

for idx, datos in resultados_alpha.items():
    alpha_total = datos["alpha"]
    es, en = interpretacion[idx]
    lineas.append(f"\n  {NOMBRES_INDICES[idx]} / {NOMBRES_INDICES_EN[idx]}")
    for col_num, col_raw in zip(datos["cols_num"], datos["cols_raw"]):
        a_sin = datos["alpha_si_elimina"][col_num]
        delta = a_sin - alpha_total
        signo = "▲" if delta > 0.01 else ("▽" if delta < -0.01 else " ")
        item_corto = ITEM_CORTO[col_raw]
        desc = col_raw.split(". ", 1)[-1][:44] if ". " in col_raw else col_raw[:44]
        lineas.append(
            f"  {item_corto:<8} {desc:<46} "
            f"{a_sin:>10.3f} {signo}{abs(delta):>7.3f}"
        )

lineas.append(nota_pie(
    "α = Alpha de Cronbach; k = número de ítems; Δα = cambio en alpha si se elimina el ítem.",
    "α = Cronbach's Alpha; k = number of items; Δα = change in alpha if item deleted.",
))
guardar_tabla("\n".join(lineas), "tab07_fiabilidad.txt")


# =============================================================================
# SECCIÓN 2 — CORRELACIONES ÍTEM-TOTAL (TAB08)
# =============================================================================

imprimir_seccion("2. CORRELACIONES ÍTEM-TOTAL")

lineas = []
lineas.append(encabezado_bilingue(
    "Tabla 8. Estadísticos por ítem y correlaciones ítem-total corregidas",
    "Table 8. Item Statistics and Corrected Item-Total Correlations"
))

lineas.append(
    f"\n  {'Ítem':<8} {'Descripción (ES/EN)':<48} "
    f"{'M':>6} {'DE':>6} {'r_it':>7} {'α sin':>7}"
)
lineas.append("  " + "─" * 86)

for idx, datos in resultados_alpha.items():
    es, en = interpretacion[idx]
    lineas.append(f"\n  {NOMBRES_INDICES[idx]} / {NOMBRES_INDICES_EN[idx]}")
    lineas.append("  " + "─" * 86)

    data_escala = df[datos["cols_num"]].dropna()

    for col_num, col_raw in zip(datos["cols_num"], datos["cols_raw"]):
        r_it = r_it_corregida(data_escala, col_num)
        m    = data_escala[col_num].mean()
        s    = data_escala[col_num].std()
        a_sin = datos["alpha_si_elimina"][col_num]
        item_corto = ITEM_CORTO[col_raw]
        desc_es = col_raw.split(". ", 1)[-1][:46] if ". " in col_raw else col_raw[:46]
        lineas.append(
            f"  {item_corto:<8} {desc_es:<48} "
            f"{m:>6.3f} {s:>6.3f} {r_it:>7.3f} {a_sin:>7.3f}"
        )

lineas.append(nota_pie(
    "M = media; DE = desviación estándar; r_it = correlación ítem-total corregida; "
    "α sin = alpha si se elimina el ítem.",
    "M = mean; SD = standard deviation; r_it = corrected item-total correlation; "
    "α if del. = alpha if item deleted.",
))
guardar_tabla("\n".join(lineas), "tab08_correlaciones_item_total.txt")


# =============================================================================
# SECCIÓN 3 — AFC / CFA (requiere semopy)
# =============================================================================

if not SEMOPY_OK:
    imprimir_seccion("3. AFC — OMITIDO (semopy no disponible)")
    print("  Instalar con: pip install semopy")
    print("  Luego volver a ejecutar este script.")
    sys.exit(0)

imprimir_seccion("3. PREPARACIÓN DEL AFC")

# Construir dataframe con nombres cortos para semopy
col_map = {}  # nombre_largo → nombre_corto

for i, col in enumerate(EA_COLS):
    col_map[col + "_num"]   = f"EA{i+1}"
for i, col in enumerate(FA_COLS):
    col_map[col + "_num"]   = f"FA{i+1}"
for i, col in enumerate(MV_COLS):
    col_map[col + "_num"]   = f"MV{i+1}"
for i, col in enumerate(EQ_COLS):
    col_map[col + "_final"] = f"EQ{i+1}"  # ya tiene inversión aplicada
for i, col in enumerate(BP_COLS):
    col_map[col + "_num"]   = f"BP{i+1}"

df_cfa = df[list(col_map.keys())].copy().rename(columns=col_map).dropna()
print(f"  Muestra para AFC: n = {len(df_cfa)}")
print(f"  Variables: {len(df_cfa.columns)}")

# Especificación del modelo CFA (5 factores independientes)
modelo_cfa = """
EA =~ EA1 + EA2 + EA3 + EA4 + EA5 + EA6 + EA7 + EA8
FA =~ FA1 + FA2 + FA3 + FA4 + FA5 + FA6 + FA7 + FA8
MV =~ MV1 + MV2 + MV3 + MV4 + MV5 + MV6 + MV7 + MV8
EQ =~ EQ1 + EQ2 + EQ3 + EQ4 + EQ5 + EQ6 + EQ7 + EQ8
BP =~ BP1 + BP2 + BP3 + BP4 + BP5
"""

print("\n  Especificación del modelo:")
for linea in modelo_cfa.strip().split("\n"):
    print(f"    {linea}")


# =============================================================================
# SECCIÓN 4 — AJUSTE DEL MODELO (TAB09) + SALIDA COMPLETA
# =============================================================================

imprimir_seccion("4. AJUSTE DEL MODELO CFA")

modelo = semopy.Model(modelo_cfa)
resultado_fit = modelo.fit(df_cfa)
params = modelo.inspect(std_est=True)
stats_cfa = semopy.calc_stats(modelo)

print(f"\n  Resultado de la optimización: {resultado_fit}")

# Diagnóstico: mostrar estructura real de calc_stats
print(f"\n  Índice de calc_stats:  {list(stats_cfa.index)}")
print(f"  Columnas de calc_stats: {list(stats_cfa.columns)}")

def extraer_stat(stats_obj, *claves_posibles):
    """
    Extrae un valor de semopy.calc_stats() tolerando dos estructuras:
      A) stats como filas: stats_obj.loc['CFI'].iloc[0]      <- versiones antiguas
      B) stats como columnas: stats_obj.loc['Value', 'CFI']  <- versiones recientes
    """
    # Detectar estructura: si 'Value' es el único índice → estructura B
    es_estructura_b = (
        isinstance(stats_obj, pd.DataFrame)
        and len(stats_obj.index) == 1
        and str(stats_obj.index[0]) == "Value"
    )

    for clave in claves_posibles:
        try:
            if es_estructura_b:
                # Buscar en columnas (exacto)
                if clave in stats_obj.columns:
                    return to_float(stats_obj.loc["Value", clave])
                # Buscar insensible a mayúsculas
                for col in stats_obj.columns:
                    if str(col).lower() == str(clave).lower():
                        return to_float(stats_obj.loc["Value", col])
            else:
                # Buscar en índice (exacto)
                if clave in stats_obj.index:
                    return to_float(stats_obj.loc[clave].iloc[0])
                # Buscar insensible a mayúsculas
                for idx in stats_obj.index:
                    if str(idx).lower() == str(clave).lower():
                        return to_float(stats_obj.loc[idx].iloc[0])
        except Exception:
            continue
    return np.nan

chi2    = extraer_stat(stats_cfa, "chi2",         "Chi2",        "Chi-square",  "χ²")
df_chi2 = extraer_stat(stats_cfa, "DoF",          "dof",         "df",          "Df")
p_chi2  = extraer_stat(stats_cfa, "chi2 p-value", "p-value",     "pvalue",      "Pr(>chi2)")
cfi     = extraer_stat(stats_cfa, "CFI",          "cfi")
tli     = extraer_stat(stats_cfa, "TLI",          "tli",         "NNFI")
rmsea   = extraer_stat(stats_cfa, "RMSEA",        "rmsea")
aic     = extraer_stat(stats_cfa, "AIC",          "aic")
bic     = extraer_stat(stats_cfa, "BIC",          "bic")

# Calcular SRMR manualmente
try:
    S_obs = df_cfa.cov().values
    S_mod = modelo.predict(df_cfa).values
    srmr  = calcular_srmr(S_obs, S_mod)
except Exception:
    srmr = np.nan

# RMSEA IC 90% (aproximación)
try:
    n_obs = len(df_cfa)
    gl    = int(df_chi2)
    if gl > 0 and n_obs > 0:
        rmsea_lo = max(0, np.sqrt(max(0, chi2 - gl) / (gl * (n_obs - 1))))
        rmsea_hi = np.sqrt(max(0, chi2 + 1.645 * np.sqrt(2 * gl)) / (gl * (n_obs - 1)))
    else:
        rmsea_lo = rmsea_hi = np.nan
except Exception:
    rmsea_lo = rmsea_hi = np.nan

print(f"\n  χ²({int(df_chi2) if not np.isnan(df_chi2) else '?'}) = {chi2:.3f}, p = {p_chi2:.4f}")
print(f"  CFI = {cfi:.3f}  TLI = {tli:.3f}")
print(f"  RMSEA = {rmsea:.4f} [{rmsea_lo:.4f}, {rmsea_hi:.4f}]")
print(f"  SRMR = {srmr:.4f}")
print(f"  AIC = {aic:.2f}  BIC = {bic:.2f}")

def eval_ajuste(val, umbral, mejor="menor"):
    if np.isnan(val):
        return "N/D"
    if mejor == "menor":
        return "✅ Bueno" if val <= umbral else "⚠️ Marginal"
    else:
        return "✅ Bueno" if val >= umbral else "⚠️ Marginal"

lineas = []
lineas.append(encabezado_bilingue(
    "Tabla 9. Índices de ajuste del Análisis Factorial Confirmatorio",
    "Table 9. Confirmatory Factor Analysis Fit Indices"
))

lineas.append(f"\n  Modelo: 5 factores correlacionados, {len(df_cfa.columns)} ítems observados")
lineas.append(f"  Model: 5 correlated factors, {len(df_cfa.columns)} observed items")
lineas.append(f"  n = {len(df_cfa)}")
lineas.append("\n  " + "─" * 72)
lineas.append(
    f"  {'Índice / Index':<28} {'Valor / Value':>14} "
    f"{'Umbral / Threshold':>20} {'Evaluación':>12}"
)
lineas.append("  " + "─" * 72)

indices_ajuste = [
    ("χ² / Chi-square",              f"{chi2:.3f}",              "—",         "—"),
    ("gl / df",                       f"{int(df_chi2) if not np.isnan(df_chi2) else 'N/D'}",
                                                                  "—",         "—"),
    ("p",                             f"{p_chi2:.4f}",            "> .05",     ""),
    ("CFI",                           f"{cfi:.3f}",               "≥ .95",     eval_ajuste(cfi, CFI_MIN, "mayor")),
    ("TLI",                           f"{tli:.3f}",               "≥ .95",     eval_ajuste(tli, TLI_MIN, "mayor")),
    ("RMSEA",                         f"{rmsea:.4f}",             "< .08",     eval_ajuste(rmsea, RMSEA_MAX, "menor")),
    ("RMSEA IC 90% / 90% CI",
     f"[{rmsea_lo:.4f}, {rmsea_hi:.4f}]", "LI < .08",           ""),
    ("SRMR",                          f"{srmr:.4f}",              "< .08",     eval_ajuste(srmr, SRMR_MAX, "menor")),
    ("AIC",                           f"{aic:.2f}",               "—",         "—"),
    ("BIC",                           f"{bic:.2f}",               "—",         "—"),
]

for nombre, valor, umbral, evaluacion in indices_ajuste:
    lineas.append(
        f"  {nombre:<28} {valor:>14} {umbral:>20} {evaluacion:>12}"
    )

lineas.append(nota_pie(
    "CFI = Índice de Ajuste Comparativo; TLI = Tucker-Lewis Index; "
    "RMSEA = Error Cuadrático Medio de Aproximación; SRMR = Raíz del Residuo Cuadrático Medio Estandarizado.",
    "CFI = Comparative Fit Index; TLI = Tucker-Lewis Index; "
    "RMSEA = Root Mean Square Error of Approximation; SRMR = Standardized Root Mean Square Residual.",
))
guardar_tabla("\n".join(lineas), "tab09_cfa_ajuste.txt")

# Guardar salida completa del modelo
output_completo = []
output_completo.append("=" * 72)
output_completo.append("ANÁLISIS FACTORIAL CONFIRMATORIO — SALIDA COMPLETA")
output_completo.append("CONFIRMATORY FACTOR ANALYSIS — FULL OUTPUT")
output_completo.append("=" * 72)
output_completo.append(f"\nModelo / Model:\n{modelo_cfa}")
output_completo.append(f"\nn = {len(df_cfa)}")
output_completo.append(f"\nResultado optimización / Optimization result:\n{resultado_fit}")
output_completo.append(f"\nEstadísticos de ajuste / Fit statistics:\n{stats_cfa.to_string()}")
output_completo.append(f"\nParámetros estimados / Parameter estimates:\n{params.to_string()}")
guardar_modelo("\n".join(output_completo), "mod_cfa_output_completo.txt")


# =============================================================================
# SECCIÓN 5 — CARGAS FACTORIALES (TAB10)
# =============================================================================

imprimir_seccion("5. CARGAS FACTORIALES")

# Filtrar solo relaciones factor → ítem (operador =~)
cargas = params[params["op"] == "~"].copy()
# En semopy la dirección es ítem ~ factor
cargas = params[params["op"].isin(["~", "=~"])].copy()

# Identificar las filas de cargas factoriales
# semopy usa "lval ~ rval" donde rval es el factor latente
cargas_factoriales = params[
    params["rval"].str.match(r"^(EA|FA|MV|EQ|BP)$", na=False)
].copy()

if len(cargas_factoriales) == 0:
    # Intentar dirección invertida
    cargas_factoriales = params[
        params["lval"].str.match(r"^(EA|FA|MV|EQ|BP)$", na=False)
    ].copy()
    cargas_factoriales = cargas_factoriales.rename(
        columns={"lval": "factor", "rval": "item"}
    )
else:
    cargas_factoriales = cargas_factoriales.rename(
        columns={"rval": "factor", "lval": "item"}
    )

print(f"  Cargas factoriales encontradas: {len(cargas_factoriales)}")
print(f"  Columnas de inspect(): {list(params.columns)}")
print(f"  Primeras filas:\n{cargas_factoriales.head(3).to_string()}")

lineas = []
lineas.append(encabezado_bilingue(
    "Tabla 10. Cargas factoriales estandarizadas del AFC",
    "Table 10. Standardized Factor Loadings from CFA"
))

lineas.append(
    f"\n  {'Factor':<8} {'Ítem':<8} {'Descripción (ES)':<44} "
    f"{'λ std':>7} {'SE':>7} {'z':>7} {'p':>7} {'R²':>7}"
)
lineas.append("  " + "─" * 96)

factores_orden = ["EA", "FA", "MV", "EQ", "BP"]
lambdas_por_factor = {}

for factor in factores_orden:
    sub = cargas_factoriales[cargas_factoriales["factor"] == factor]
    if len(sub) == 0:
        continue

    nombre_es = NOMBRES_INDICES.get(f"IDX_{factor}", factor)
    nombre_en = NOMBRES_INDICES_EN.get(f"IDX_{factor}", factor)
    lineas.append(f"\n  {nombre_es} / {nombre_en}")
    lineas.append("  " + "─" * 96)

    lambdas = []
    for _, fila in sub.iterrows():
        item   = str(fila.get("item", ""))
        std_l  = to_float(fila.get("Std. Estimate", np.nan))
        se     = to_float(fila.get("Std. Err",       np.nan))
        z_val  = to_float(fila.get("z-value",        np.nan))
        p_val  = to_float(fila.get("p-value",        np.nan))
        r2     = std_l ** 2 if not np.isnan(std_l) else np.nan
        stars  = sig_stars(p_val)

        # Buscar descripción larga del ítem
        item_num = item  # ej: "EA1"
        desc_es = item_num  # fallback
        prefijo = item_num[:2] if len(item_num) >= 2 else ""
        try:
            num = int(item_num[2:]) - 1
            cols_map = {
                "EA": EA_COLS, "FA": FA_COLS, "MV": MV_COLS,
                "EQ": EQ_COLS, "BP": BP_COLS,
            }
            if prefijo in cols_map and num < len(cols_map[prefijo]):
                col_raw = cols_map[prefijo][num]
                desc_es = col_raw.split(". ", 1)[-1][:42] if ". " in col_raw else col_raw[:42]
        except Exception:
            pass

        lineas.append(
            f"  {factor:<8} {item_num:<8} {desc_es:<44} "
            f"{std_l:>7.3f} {se:>7.3f} {z_val:>7.3f} "
            f"{p_val:>5.3f}{stars:<2} {r2:>7.3f}"
        )

        if not np.isnan(std_l):
            lambdas.append(abs(std_l))

    lambdas_por_factor[factor] = np.array(lambdas)

lineas.append(nota_pie(
    "λ std = carga factorial estandarizada; SE = error estándar; R² = varianza explicada del ítem.",
    "λ std = standardized factor loading; SE = standard error; R² = item variance explained. "
    "***p < .001, **p < .01, *p < .05.",
))
guardar_tabla("\n".join(lineas), "tab10_cargas_factoriales.txt")


# =============================================================================
# SECCIÓN 6 — VALIDEZ CONVERGENTE (TAB11)
# =============================================================================

imprimir_seccion("6. VALIDEZ CONVERGENTE — AVE y CR")

lineas = []
lineas.append(encabezado_bilingue(
    "Tabla 11. Validez convergente: VME, FC y Alpha de Cronbach",
    "Table 11. Convergent Validity: AVE, CR, and Cronbach's Alpha"
))

lineas.append(
    f"\n  {'Constructo / Construct':<34} {'ES':<22} {'EN':<22} "
    f"{'k':>3} {'α':>7} {'VME/AVE':>8} {'FC/CR':>7} {'Eval.':>10}"
)
lineas.append("  " + "─" * 110)

resultados_val = {}

for factor in factores_orden:
    idx    = f"IDX_{factor}"
    lambdas = lambdas_por_factor.get(factor, np.array([]))
    alpha   = resultados_alpha[idx]["alpha"]
    k       = resultados_alpha[idx]["n_items"]
    es, en  = interpretacion[idx]

    if len(lambdas) > 0:
        ave = calcular_ave(lambdas)
        cr  = calcular_cr(lambdas)
    else:
        ave = cr = np.nan

    eval_ave = "✅" if not np.isnan(ave) and ave >= AVE_MIN else "⚠️"
    eval_cr  = "✅" if not np.isnan(cr)  and cr  >= 0.70    else "⚠️"

    resultados_val[factor] = {"ave": ave, "cr": cr, "alpha": alpha}

    lineas.append(
        f"  {idx:<34} {es:<22} {en:<22} "
        f"{k:>3} {alpha:>7.3f} "
        f"{ave:>8.3f} {cr:>7.3f} {eval_ave}{eval_cr}"
    )

# Tabla de discriminación (AVE vs r²)
lineas.append(f"\n\n  {'─'*110}")
lineas.append("  Validez discriminante: comparación VME vs correlaciones al cuadrado")
lineas.append("  Discriminant validity: AVE vs squared inter-construct correlations")
lineas.append(
    "\n  Para validez discriminante se requiere que VME(A) > r²(A,B) para todo par de constructos."
)
lineas.append(
    "  Discriminant validity requires AVE(A) > r²(A,B) for all construct pairs.\n"
)

# Calcular correlaciones entre índices
par_correlaciones = []
for i, f1 in enumerate(factores_orden):
    for f2 in factores_orden[i+1:]:
        r, p = stats.pearsonr(df[f"IDX_{f1}"].dropna(), df[f"IDX_{f2}"].dropna())
        r2   = r ** 2
        ave1 = resultados_val[f1]["ave"]
        ave2 = resultados_val[f2]["ave"]
        ok   = "✅" if (ave1 >= r2 and ave2 >= r2) else "⚠️"
        par_correlaciones.append((f1, f2, r, r2, ave1, ave2, ok))

lineas.append(
    f"  {'Par / Pair':<12} {'r':>7} {'r²':>7} "
    f"{'VME(A)':>8} {'VME(B)':>8} {'Discriminante?':>15}"
)
lineas.append("  " + "─" * 60)
for f1, f2, r, r2, ave1, ave2, ok in par_correlaciones:
    lineas.append(
        f"  {f1}–{f2:<9} {r:>7.3f} {r2:>7.3f} "
        f"{ave1:>8.3f} {ave2:>8.3f} {ok:>15}"
    )

lineas.append(nota_pie(
    "VME = Varianza Media Extraída (umbral ≥ .50); FC = Fiabilidad Compuesta (umbral ≥ .70); "
    "α = Alpha de Cronbach; k = número de ítems.",
    "AVE = Average Variance Extracted (threshold ≥ .50); CR = Composite Reliability (threshold ≥ .70); "
    "α = Cronbach's Alpha; k = number of items.",
))
guardar_tabla("\n".join(lineas), "tab11_validez_convergente.txt")


# =============================================================================
# RESUMEN FINAL
# =============================================================================

imprimir_seccion("RESUMEN — PSICOMETRÍA COMPLETADA")

archivos = [
    ("tab07_fiabilidad.txt",            TABLAS_DIR),
    ("tab08_correlaciones_item_total.txt", TABLAS_DIR),
    ("tab09_cfa_ajuste.txt",            TABLAS_DIR),
    ("tab10_cargas_factoriales.txt",    TABLAS_DIR),
    ("tab11_validez_convergente.txt",   TABLAS_DIR),
    ("mod_cfa_output_completo.txt",     MODELOS_DIR),
]
for archivo, carpeta in archivos:
    ruta   = carpeta / archivo
    estado = "✅" if ruta.exists() else "❌"
    print(f"  {estado} {archivo}")