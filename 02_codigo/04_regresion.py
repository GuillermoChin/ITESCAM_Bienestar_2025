"""
04_regresion.py
Regresiones múltiples OLS para cada outcome principal.

Entrada:  03_salidas/modelos/datos_procesados.pkl
Salidas:  03_salidas/tablas/tab19_reg_estres.txt
          03_salidas/tablas/tab20_reg_fatiga.txt
          03_salidas/tablas/tab21_reg_motivacion.txt
          03_salidas/tablas/tab22_reg_bienestar.txt
          03_salidas/tablas/tab23_reg_equidad.txt
          03_salidas/tablas/tab24_reg_abandono.txt
          03_salidas/modelos/mod_regresiones_completas.txt
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
    INDICES, NOMBRES_INDICES, NOMBRES_INDICES_EN,
    ALPHA_SIG,
)

try:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.stats.stattools import durbin_watson
    SM_OK = True
except ImportError:
    SM_OK = False
    print("  ⚠️  statsmodels no disponible — pip install statsmodels")
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


def calcular_vif(X_df: pd.DataFrame) -> pd.Series:
    """Calcula el VIF para cada predictor."""
    vif_data = {}
    X_arr = X_df.values
    for i, col in enumerate(X_df.columns):
        try:
            vif_data[col] = variance_inflation_factor(X_arr, i)
        except Exception:
            vif_data[col] = np.nan
    return pd.Series(vif_data)


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


def beta_estandarizado(modelo, X_df, y) -> dict:
    """Calcula coeficientes beta estandarizados."""
    betas = {}
    sy = y.std(ddof=1)
    if sy == 0:
        return {col: np.nan for col in X_df.columns}
    for col in X_df.columns:
        sx = X_df[col].std(ddof=1)
        if sx == 0 or col == "const":
            continue
        coef = modelo.params.get(col, np.nan)
        betas[col] = float(coef) * sx / sy
    return betas


def ajustar_ols(y_col: str, predictores: list,
                df_datos: pd.DataFrame,
                etiquetas: dict) -> tuple:
    """
    Ajusta un modelo OLS y retorna (resultado, texto_tabla).
    """
    datos = df_datos[[y_col] + predictores].dropna()
    y = datos[y_col]
    X = sm.add_constant(datos[predictores])

    modelo = sm.OLS(y, X).fit()

    betas = beta_estandarizado(modelo, datos[predictores], y)
    vifs  = calcular_vif(X.drop(columns=["const"]))
    dw    = durbin_watson(modelo.resid)

    # ── Tabla de resultados
    lineas = []
    lineas.append(
        f"\n  n = {len(datos)} | "
        f"R² = {modelo.rsquared:.4f} | "
        f"R² ajust. / adj. = {modelo.rsquared_adj:.4f} | "
        f"F({modelo.df_model:.0f}, {modelo.df_resid:.0f}) = "
        f"{modelo.fvalue:.3f}, p = {modelo.f_pvalue:.4f} | "
        f"Durbin-Watson = {dw:.3f}"
    )
    lineas.append(f"  Tamaño del efecto / Effect size: {interpretar_r2(modelo.rsquared)}")
    lineas.append(
        f"\n  {'Predictor':<36} {'B':>8} {'SE':>7} {'β std':>8} "
        f"{'t':>8} {'p':>8} {'VIF':>6}"
    )
    lineas.append("  " + "─" * 84)

    for pred in predictores:
        if pred not in modelo.params.index:
            continue
        B     = float(modelo.params[pred])
        se    = float(modelo.bse[pred])
        t     = float(modelo.tvalues[pred])
        p     = float(modelo.pvalues[pred])
        beta  = betas.get(pred, np.nan)
        vif   = vifs.get(pred, np.nan)
        sig   = sig_stars(p).strip()
        etiq  = etiquetas.get(pred, pred)[:34]
        lineas.append(
            f"  {etiq:<36} {B:>8.3f} {se:>7.3f} {beta:>8.3f} "
            f"{t:>8.3f} {p:>6.4f}{sig} {vif:>6.2f}"
        )

    # Constante
    if "const" in modelo.params:
        B_c  = float(modelo.params["const"])
        se_c = float(modelo.bse["const"])
        t_c  = float(modelo.tvalues["const"])
        p_c  = float(modelo.pvalues["const"])
        sig_c = sig_stars(p_c).strip()
        lineas.append(
            f"  {'Constante / Intercept':<36} {B_c:>8.3f} {se_c:>7.3f} "
            f"{'—':>8} {t_c:>8.3f} {p_c:>6.4f}{sig_c} {'—':>6}"
        )

    texto = "\n".join(lineas)

    print(
        f"  R² = {modelo.rsquared:.4f} | "
        f"R²adj = {modelo.rsquared_adj:.4f} | "
        f"F = {modelo.fvalue:.3f}, p = {modelo.f_pvalue:.4f}"
    )

    return modelo, texto


# =============================================================================
# CARGA DE DATOS
# =============================================================================

imprimir_seccion("CARGANDO DATOS")
df = pd.read_pickle(ARCHIVO_PROCESADO)
N  = len(df)
print(f"  n = {N}")

# =============================================================================
# DEFINICIÓN DE PREDICTORES
# =============================================================================

# Predictores socioestructurales (bloque 1)
PRED_STRUCT = [
    "GENERO_MUJER",
    "ORIGEN_RURAL",
    "SEMESTRE_num",
    "P8_num",          # horas de sueño
]

# Predictores de carga (bloque 2)
PRED_CARGA = [
    "P11_num",         # trabajo remunerado
    "P12_num",         # labores del hogar
    "P13_num",         # cuidado de familiares
    "CARGA_TOTAL",
]

# Predictor de acoso
PRED_ACOSO = ["ACOSO_EXP"]

# Todos los predictores exógenos juntos
PRED_EXOGENOS = PRED_STRUCT + ["P11_num", "P12_num", "P13_num"] + PRED_ACOSO

# Etiquetas bilingües de predictores
ETIQUETAS_PRED = {
    "GENERO_MUJER":  "Género: Mujer / Gender: Woman",
    "ORIGEN_RURAL":  "Origen rural / Rural origin",
    "SEMESTRE_num":  "Semestre",
    "P8_num":        "Horas de sueño / Sleep hours",
    "P11_num":       "Horas de trabajo / Work hours",
    "P12_num":       "Horas de hogar / Household hours",
    "P13_num":       "Horas de cuidado / Caregiving hours",
    "CARGA_TOTAL":   "Carga total / Total workload",
    "ACOSO_EXP":     "Experiencia acoso / Harassment exp.",
    "IDX_EA":        "Estrés Académico / Academic Stress",
    "IDX_FA":        "Fatiga / Fatigue",
    "IDX_MV":        "Motivación / Motivation",
    "IDX_EQ":        "Equidad / Equity",
    "IDX_BP":        "Bienestar / Well-being",
}

# Acumular salidas para el archivo de modelos completo
modelos_completos = []

# =============================================================================
# TAB19 — REGRESIÓN: ESTRÉS ACADÉMICO (IDX_EA)
# =============================================================================

imprimir_seccion("TAB19 — REGRESIÓN: ESTRÉS ACADÉMICO")

lineas = []
lineas.append(encabezado_bilingue(
    "Tabla 19. Regresión múltiple OLS: Predictores del Estrés Académico (IDX_EA)",
    "Table 19. OLS Multiple Regression: Predictors of Academic Stress (IDX_EA)"
))
lineas.append(
    "\n  Variable dependiente / Dependent variable: IDX_EA (Estrés Académico / Academic Stress)"
)

predictores_ea = PRED_STRUCT + ["P11_num", "P12_num", "P13_num", "ACOSO_EXP"]
modelo_ea, texto_ea = ajustar_ols("IDX_EA", predictores_ea, df, ETIQUETAS_PRED)
lineas.append(texto_ea)
lineas.append(nota_pie(
    "B = coeficiente no estandarizado; SE = error estándar; β std = beta estandarizado; "
    "VIF = Factor de Inflación de la Varianza (>10 indica multicolinealidad).",
    "B = unstandardized coefficient; SE = standard error; β std = standardized beta; "
    "VIF = Variance Inflation Factor (>10 indicates multicollinearity). "
    "***p < .001, **p < .01, *p < .05.",
))
guardar_tabla("\n".join(lineas), "tab19_reg_estres.txt")
modelos_completos.append(f"{'='*70}\nMODELO IDX_EA\n{'='*70}\n{modelo_ea.summary()}\n")


# =============================================================================
# TAB20 — REGRESIÓN: FATIGA (IDX_FA)
# =============================================================================

imprimir_seccion("TAB20 — REGRESIÓN: FATIGA")

lineas = []
lineas.append(encabezado_bilingue(
    "Tabla 20. Regresión múltiple OLS: Predictores de la Fatiga/Agotamiento (IDX_FA)",
    "Table 20. OLS Multiple Regression: Predictors of Fatigue/Burnout (IDX_FA)"
))
lineas.append(
    "\n  Variable dependiente / Dependent variable: IDX_FA (Fatiga / Fatigue)"
)

predictores_fa = PRED_STRUCT + ["P11_num", "P12_num", "P13_num", "ACOSO_EXP"]
modelo_fa, texto_fa = ajustar_ols("IDX_FA", predictores_fa, df, ETIQUETAS_PRED)
lineas.append(texto_fa)
lineas.append(nota_pie(
    "B = coeficiente no estandarizado; SE = error estándar; β std = beta estandarizado.",
    "B = unstandardized coefficient; SE = standard error; β std = standardized beta. "
    "***p < .001, **p < .01, *p < .05.",
))
guardar_tabla("\n".join(lineas), "tab20_reg_fatiga.txt")
modelos_completos.append(f"{'='*70}\nMODELO IDX_FA\n{'='*70}\n{modelo_fa.summary()}\n")


# =============================================================================
# TAB21 — REGRESIÓN: MOTIVACIÓN (IDX_MV)
# =============================================================================

imprimir_seccion("TAB21 — REGRESIÓN: MOTIVACIÓN")

lineas = []
lineas.append(encabezado_bilingue(
    "Tabla 21. Regresión múltiple OLS: Predictores de la Motivación (IDX_MV)",
    "Table 21. OLS Multiple Regression: Predictors of Motivation (IDX_MV)"
))
lineas.append(
    "\n  Variable dependiente / Dependent variable: IDX_MV (Motivación / Motivation)"
)

# Para motivación incluimos EA y FA como predictores (mediadores)
predictores_mv = (
    PRED_STRUCT
    + ["P11_num", "P12_num", "P13_num", "ACOSO_EXP"]
    + ["IDX_EA", "IDX_FA", "IDX_EQ"]
)
modelo_mv, texto_mv = ajustar_ols("IDX_MV", predictores_mv, df, ETIQUETAS_PRED)
lineas.append(texto_mv)
lineas.append(nota_pie(
    "Modelo jerárquico: bloque 1 = predictores exógenos; "
    "bloque 2 = estrés, fatiga y equidad como predictores proximales.",
    "Hierarchical model: block 1 = exogenous predictors; "
    "block 2 = stress, fatigue, and equity as proximal predictors. "
    "***p < .001, **p < .01, *p < .05.",
))
guardar_tabla("\n".join(lineas), "tab21_reg_motivacion.txt")
modelos_completos.append(f"{'='*70}\nMODELO IDX_MV\n{'='*70}\n{modelo_mv.summary()}\n")


# =============================================================================
# TAB22 — REGRESIÓN: BIENESTAR PERCIBIDO (IDX_BP)
# =============================================================================

imprimir_seccion("TAB22 — REGRESIÓN: BIENESTAR")

lineas = []
lineas.append(encabezado_bilingue(
    "Tabla 22. Regresión múltiple OLS: Predictores del Bienestar Percibido (IDX_BP)",
    "Table 22. OLS Multiple Regression: Predictors of Perceived Well-being (IDX_BP)"
))
lineas.append(
    "\n  Variable dependiente / Dependent variable: IDX_BP (Bienestar / Well-being)"
)

# Para bienestar incluimos todos los índices como predictores
predictores_bp = (
    PRED_STRUCT
    + ["P11_num", "P12_num", "P13_num", "ACOSO_EXP"]
    + ["IDX_EA", "IDX_FA", "IDX_MV", "IDX_EQ"]
)
modelo_bp, texto_bp = ajustar_ols("IDX_BP", predictores_bp, df, ETIQUETAS_PRED)
lineas.append(texto_bp)
lineas.append(nota_pie(
    "Modelo completo incluyendo índices psicológicos como predictores proximales.",
    "Full model including psychological indices as proximal predictors. "
    "***p < .001, **p < .01, *p < .05.",
))
guardar_tabla("\n".join(lineas), "tab22_reg_bienestar.txt")
modelos_completos.append(f"{'='*70}\nMODELO IDX_BP\n{'='*70}\n{modelo_bp.summary()}\n")


# =============================================================================
# TAB23 — REGRESIÓN: PERCEPCIÓN DE EQUIDAD (IDX_EQ)
# =============================================================================

imprimir_seccion("TAB23 — REGRESIÓN: EQUIDAD")

lineas = []
lineas.append(encabezado_bilingue(
    "Tabla 23. Regresión múltiple OLS: Predictores de la Percepción de Equidad (IDX_EQ)",
    "Table 23. OLS Multiple Regression: Predictors of Perceived Equity (IDX_EQ)"
))
lineas.append(
    "\n  Variable dependiente / Dependent variable: IDX_EQ (Equidad / Equity)"
)

predictores_eq = PRED_STRUCT + ["ACOSO_EXP", "P11_num", "P12_num", "P13_num"]
modelo_eq, texto_eq = ajustar_ols("IDX_EQ", predictores_eq, df, ETIQUETAS_PRED)
lineas.append(texto_eq)
lineas.append(nota_pie(
    "La percepción de equidad refleja el clima institucional percibido.",
    "Perceived equity reflects the perceived institutional climate. "
    "***p < .001, **p < .01, *p < .05.",
))
guardar_tabla("\n".join(lineas), "tab23_reg_equidad.txt")
modelos_completos.append(f"{'='*70}\nMODELO IDX_EQ\n{'='*70}\n{modelo_eq.summary()}\n")


# =============================================================================
# TAB24 — REGRESIÓN: INTENCIÓN DE ABANDONO (P16b)
# =============================================================================

imprimir_seccion("TAB24 — REGRESIÓN: INTENCIÓN DE ABANDONO")

col_p16b = "P16b. ¿Qué tan probable: abandonar o interrumpir estudios?_num"

lineas = []
lineas.append(encabezado_bilingue(
    "Tabla 24. Regresión múltiple OLS: Predictores de la Intención de Abandono (P16b)",
    "Table 24. OLS Multiple Regression: Predictors of Dropout Intention (P16b)"
))
lineas.append(
    "\n  Variable dependiente / Dependent variable: "
    "P16b (Intención de abandono / Dropout intention, 1–5)"
)

if col_p16b in df.columns:
    predictores_p16b = (
        PRED_STRUCT
        + ["P11_num", "P12_num", "P13_num", "ACOSO_EXP"]
        + ["IDX_EA", "IDX_FA", "IDX_MV", "IDX_EQ", "IDX_BP"]
    )
    modelo_p16b, texto_p16b = ajustar_ols(col_p16b, predictores_p16b, df, ETIQUETAS_PRED)
    lineas.append(texto_p16b)
    modelos_completos.append(
        f"{'='*70}\nMODELO P16b (ABANDONO)\n{'='*70}\n{modelo_p16b.summary()}\n"
    )
else:
    lineas.append("\n  ⚠️  Columna P16b_num no encontrada en el dataset procesado.")
    print("  ⚠️  P16b_num no encontrada")

lineas.append(nota_pie(
    "P16b recodificada: 1 = Muy improbable, 5 = Muy probable (abandono). "
    "Scores altos indican mayor riesgo de deserción.",
    "P16b recoded: 1 = Very unlikely, 5 = Very likely (dropout). "
    "Higher scores indicate greater dropout risk. "
    "***p < .001, **p < .01, *p < .05.",
))
guardar_tabla("\n".join(lineas), "tab24_reg_abandono.txt")


# =============================================================================
# TABLA COMPARATIVA RESUMEN DE TODOS LOS MODELOS
# =============================================================================

imprimir_seccion("TABLA COMPARATIVA DE MODELOS")

resumen_modelos = [
    ("IDX_EA", "Estrés Académico",   "Academic Stress",   modelo_ea),
    ("IDX_FA", "Fatiga",             "Fatigue",           modelo_fa),
    ("IDX_MV", "Motivación",         "Motivation",        modelo_mv),
    ("IDX_BP", "Bienestar",          "Well-being",        modelo_bp),
    ("IDX_EQ", "Equidad",            "Equity",            modelo_eq),
]

lineas_res = []
lineas_res.append(encabezado_bilingue(
    "Tabla Resumen. Comparación de modelos de regresión",
    "Summary Table. Regression Model Comparison"
))
lineas_res.append(
    f"\n  {'Modelo / Model':<36} {'R²':>8} {'R²adj':>8} "
    f"{'F':>9} {'p(F)':>8} {'AIC':>10} {'Tamaño/Size':>14}"
)
lineas_res.append("  " + "─" * 96)

for idx, nombre_es, nombre_en, modelo in resumen_modelos:
    try:
        aic_val = modelo.aic
    except Exception:
        aic_val = np.nan
    lineas_res.append(
        f"  {nombre_es:<18} {nombre_en:<18} "
        f"{modelo.rsquared:>8.4f} {modelo.rsquared_adj:>8.4f} "
        f"{modelo.fvalue:>9.3f} {modelo.f_pvalue:>8.4f} "
        f"{aic_val:>10.2f} "
        f"{interpretar_r2(modelo.rsquared):>14}"
    )

lineas_res.append(nota_pie(
    "R² = coeficiente de determinación; R²adj = R² ajustado; "
    "F = estadístico F del modelo; AIC = Criterio de Información de Akaike.",
    "R² = coefficient of determination; R²adj = adjusted R²; "
    "F = model F-statistic; AIC = Akaike Information Criterion.",
))

# Añadir al final de cada tabla individual y guardar por separado
guardar_tabla("\n".join(lineas_res), "tab_resumen_regresiones.txt")


# =============================================================================
# GUARDAR SALIDAS COMPLETAS DE MODELOS
# =============================================================================

guardar_modelo(
    "\n\n".join(modelos_completos),
    "mod_regresiones_completas.txt"
)


# =============================================================================
# RESUMEN FINAL
# =============================================================================

imprimir_seccion("RESUMEN — REGRESIONES COMPLETADAS")

archivos = [
    ("tab19_reg_estres.txt",              TABLAS_DIR),
    ("tab20_reg_fatiga.txt",              TABLAS_DIR),
    ("tab21_reg_motivacion.txt",          TABLAS_DIR),
    ("tab22_reg_bienestar.txt",           TABLAS_DIR),
    ("tab23_reg_equidad.txt",             TABLAS_DIR),
    ("tab24_reg_abandono.txt",            TABLAS_DIR),
    ("tab_resumen_regresiones.txt",       TABLAS_DIR),
    ("mod_regresiones_completas.txt",     MODELOS_DIR),
]
for archivo, carpeta in archivos:
    ruta   = carpeta / archivo
    estado = "✅" if ruta.exists() else "❌"
    print(f"  {estado} {archivo}")