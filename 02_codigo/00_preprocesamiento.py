"""
00_preprocesamiento.py
Limpieza, codificación numérica, inversión de ítems,
cómputo de índices y variables derivadas.

Entrada:  01_datos/Dataset_Final_ITESCAM_2025.xlsx
Salida:   03_salidas/modelos/datos_procesados.pkl
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

# Asegurar que Python encuentre config.py en la misma carpeta
sys.path.append(str(__import__("pathlib").Path(__file__).parent))
from config import (
    ARCHIVO_DATOS, ARCHIVO_PROCESADO,
    EA_COLS, FA_COLS, MV_COLS, EQ_COLS, BP_COLS,
    EQ_ITEMS_INVERSOS, ALL_LIKERT_COLS, P16_COLS,
    LIKERT_FRECUENCIA, LIKERT_ACUERDO, LIKERT_PROBABILIDAD,
    HORAS_TRABAJO, HORAS_HOGAR, HORAS_CUIDADO, HORAS_SUENO,
    SEMESTRE_ORD, CARRERAS_CORTAS, CARRERAS_ORDEN,
)

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def imprimir_seccion(titulo):
    print(f"\n{'='*60}")
    print(f"  {titulo}")
    print(f"{'='*60}")


def reporte_valores_perdidos(df, etapa=""):
    total_na = df.isnull().sum().sum()
    if total_na > 0:
        cols_con_na = df.isnull().sum()
        cols_con_na = cols_con_na[cols_con_na > 0]
        print(f"  ⚠️  Valores perdidos ({etapa}): {total_na} en {len(cols_con_na)} columnas")
        for col, n in cols_con_na.items():
            print(f"     · {col[:60]}: {n}")
    else:
        print(f"  ✅ Sin valores perdidos ({etapa})")


# =============================================================================
# 1. CARGA DEL DATASET
# =============================================================================

imprimir_seccion("1. CARGA DEL DATASET")

df = pd.read_excel(ARCHIVO_DATOS)
print(f"  Filas: {len(df)} | Columnas: {len(df.columns)}")
reporte_valores_perdidos(df, "crudo")

# Verificar que todos los ítems Likert están en el dataframe
cols_faltantes = [c for c in ALL_LIKERT_COLS if c not in df.columns]
if cols_faltantes:
    print(f"  ❌ Columnas no encontradas en el dataset:")
    for c in cols_faltantes:
        print(f"     · {c}")
    sys.exit(1)
else:
    print(f"  ✅ Los {len(ALL_LIKERT_COLS)} ítems Likert están presentes")


# =============================================================================
# 2. CODIFICACIÓN NUMÉRICA DE ESCALAS LIKERT
# =============================================================================

imprimir_seccion("2. CODIFICACIÓN NUMÉRICA LIKERT")

# EA y FA usan escala de frecuencia (Nunca → Siempre)
for col in EA_COLS + FA_COLS:
    df[col + "_num"] = df[col].map(LIKERT_FRECUENCIA)

# MV, EQ y BP usan escala de acuerdo (Totalmente en desacuerdo → Totalmente de acuerdo)
for col in MV_COLS + EQ_COLS + BP_COLS:
    df[col + "_num"] = df[col].map(LIKERT_ACUERDO)

# P16 usa escala de probabilidad
for col in P16_COLS:
    df[col + "_num"] = df[col].map(LIKERT_PROBABILIDAD)

# Verificar que no quedaron NaN después del mapeo
cols_num = [c + "_num" for c in ALL_LIKERT_COLS + P16_COLS]
na_post_mapeo = df[cols_num].isnull().sum().sum()
if na_post_mapeo > 0:
    print(f"  ⚠️  {na_post_mapeo} valores no mapeados — verificar opciones de respuesta")
    for col in cols_num:
        if df[col].isnull().sum() > 0:
            vals_na = df.loc[df[col].isnull(), col.replace("_num", "")].unique()
            print(f"     · {col[:50]}: valores no reconocidos → {vals_na}")
else:
    print(f"  ✅ Codificación Likert completada sin valores perdidos")


# =============================================================================
# 3. INVERSIÓN DE ÍTEMS REVERSOS (EQ4 y EQ7)
# =============================================================================

imprimir_seccion("3. INVERSIÓN DE ÍTEMS REVERSOS")

for col in EQ_ITEMS_INVERSOS:
    col_num  = col + "_num"
    col_inv  = col + "_inv"
    df[col_inv] = 6 - df[col_num]
    media_orig = df[col_num].mean()
    media_inv  = df[col_inv].mean()
    print(f"  {col[:55]}")
    print(f"     Media original: {media_orig:.3f} → Media invertida: {media_inv:.3f}")

# Crear columnas _final para todos los ítems EQ
# (los inversos usan _inv; los directos usan _num)
for col in EQ_COLS:
    if col in EQ_ITEMS_INVERSOS:
        df[col + "_final"] = df[col + "_inv"]
    else:
        df[col + "_final"] = df[col + "_num"]

print(f"\n  ✅ Ítems invertidos: {len(EQ_ITEMS_INVERSOS)}")


# =============================================================================
# 4. CÓMPUTO DE ÍNDICES (media por dimensión)
# =============================================================================

imprimir_seccion("4. CÓMPUTO DE ÍNDICES")

# EA: media de los 8 ítems de frecuencia
df["IDX_EA"] = df[[c + "_num" for c in EA_COLS]].mean(axis=1)

# FA: media de los 8 ítems de frecuencia
df["IDX_FA"] = df[[c + "_num" for c in FA_COLS]].mean(axis=1)

# MV: media de los 8 ítems de acuerdo
df["IDX_MV"] = df[[c + "_num" for c in MV_COLS]].mean(axis=1)

# EQ: media de los 8 ítems con EQ4 y EQ7 ya invertidos
df["IDX_EQ"] = df[[c + "_final" for c in EQ_COLS]].mean(axis=1)

# BP: media de los 5 ítems de acuerdo (SWLS)
df["IDX_BP"] = df[[c + "_num" for c in BP_COLS]].mean(axis=1)

# Resumen de índices
indices = ["IDX_EA", "IDX_FA", "IDX_MV", "IDX_EQ", "IDX_BP"]
print(f"\n  {'Índice':<12} {'Media':>7} {'DE':>7} {'Mín':>7} {'Máx':>7}")
print(f"  {'-'*42}")
for idx in indices:
    m  = df[idx].mean()
    s  = df[idx].std()
    mn = df[idx].min()
    mx = df[idx].max()
    print(f"  {idx:<12} {m:>7.3f} {s:>7.3f} {mn:>7.3f} {mx:>7.3f}")

print(f"\n  ✅ 5 índices calculados (rango teórico 1–5)")


# =============================================================================
# 5. RECODIFICACIÓN DE VARIABLES DE CARGA (ordinales numéricas)
# =============================================================================

imprimir_seccion("5. VARIABLES DE CARGA")

# ── Homologación de valores no estándar ──────────────────────────────────────
# Los registros UNI-0528 a UNI-0555 usaron un formulario con opciones distintas.
# Se mapean a la escala estándar antes de la codificación numérica.

# P11 — Trabajo
homolog_trabajo = {
    "No tengo trabajo remunerado":  "No trabajo",
    "1-2 Horas":                    "Menos de 5 horas a la semana",
    "3-7 Horas":                    "Entre 5 y 10 horas a la semana",
    "8  Horas":                     "Entre 5 y 10 horas a la semana",
    "Más de 8 horas":               "Entre 5 y 10 horas a la semana",
    "De 0 a 5 horas":               "Menos de 5 horas a la semana",
    "De 5 a 10 horas":              "Entre 5 y 10 horas a la semana",
    "De 10 a 15 horas":             "Entre 10 y 20 horas a la semana",
}

# P12 — Hogar
homolog_hogar = {
    "1-2 Horas":   "Menos de 5 horas a la semana",
    "3-7 Horas":   "Entre 5 y 10 horas a la semana",
    "8  Horas":    "Entre 5 y 10 horas a la semana",
}

# P13 — Cuidado
homolog_cuidado = {
    "No cuido a familiares":  "Ninguna",
    "De 0 a 2 horas":         "Menos de 5 horas a la semana",
    "De 2 a 4 horas":         "Menos de 5 horas a la semana",
    "De 4 a 6 horas":         "Entre 5 y 10 horas a la semana",
    "Más de 6 horas":         "Entre 5 y 10 horas a la semana",
    "1-2 Horas":              "Menos de 5 horas a la semana",
    "3-7 Horas":              "Entre 5 y 10 horas a la semana",
}

# P8 — Sueño
homolog_sueno = {
    "De 4 a 6 horas":  "Entre 4 y 5 horas",
    "De 6 a 8 horas":  "Entre 6 y 7 horas",
}

# Aplicar homologaciones (solo reemplaza los valores no estándar)
col_p11 = "P11. Horas de trabajo a la semana"
col_p12 = "P12. Horas de labores del hogar/semana"
col_p13 = "P13. Horas de cuidado de familiares/semana"
col_p8  = "P8. Horas de sueño"

df[col_p11] = df[col_p11].replace(homolog_trabajo)
df[col_p12] = df[col_p12].replace(homolog_hogar)
df[col_p13] = df[col_p13].replace(homolog_cuidado)
df[col_p8]  = df[col_p8].replace(homolog_sueno)

# Reporte de homologaciones aplicadas
total_homolog = (
    len(homolog_trabajo) + len(homolog_hogar) +
    len(homolog_cuidado) + len(homolog_sueno)
)
print(f"  Homologación aplicada: {total_homolog} reglas en 4 variables")

# ── Mapeo a numérico ──────────────────────────────────────────────────────────

# Mapeo origen → columna original (para reporte de errores)
col_original = {
    "P11_num":      col_p11,
    "P12_num":      col_p12,
    "P13_num":      col_p13,
    "P8_num":       col_p8,
    "SEMESTRE_num": "P3. Semestre",
}

df["P11_num"]      = df[col_p11].map(HORAS_TRABAJO)
df["P12_num"]      = df[col_p12].map(HORAS_HOGAR)
df["P13_num"]      = df[col_p13].map(HORAS_CUIDADO)
df["P8_num"]       = df[col_p8].map(HORAS_SUENO)
df["SEMESTRE_num"] = df["P3. Semestre"].map(SEMESTRE_ORD)

# Verificar NaN post-mapeo
carga_cols = ["P11_num", "P12_num", "P13_num", "P8_num", "SEMESTRE_num"]
na_carga = df[carga_cols].isnull().sum()

if na_carga.sum() > 0:
    print(f"  ⚠️  Valores no mapeados tras homologación:")
    for col_num, n in na_carga[na_carga > 0].items():
        col_orig = col_original[col_num]
        vals = df.loc[df[col_num].isnull(), col_orig].unique()
        print(f"     · {col_num}: {n} caso(s) — valores: {vals}")
else:
    print(f"  ✅ Variables de carga codificadas sin valores perdidos")

# Variable compuesta de carga extracurricular (normalizada 0–1)
df["CARGA_TOTAL_raw"] = df["P11_num"] + df["P12_num"] + df["P13_num"]
df["CARGA_TOTAL"]     = df["CARGA_TOTAL_raw"] / 13.0

print(f"\n  CARGA_TOTAL (normalizada 0–1):")
print(f"    Media: {df['CARGA_TOTAL'].mean():.3f} | DE: {df['CARGA_TOTAL'].std():.3f}")
print(f"    Mín:   {df['CARGA_TOTAL'].min():.3f} | Máx: {df['CARGA_TOTAL'].max():.3f}")


# =============================================================================
# 6. VARIABLES DUMMY Y DICOTÓMICAS
# =============================================================================

imprimir_seccion("6. VARIABLES DUMMY Y DICOTÓMICAS")

# Género: dummy Mujer (ref = Hombre)
# NB y Prefiero no responder → NaN en la dummy (se reportan en descriptivos)
df["GENERO_MUJER"] = df["P4. Género"].map({
    "Mujer":  1,
    "Hombre": 0,
})  # NaN automático para No binario y Prefiero no responder

# Variable multi-categoría de género para descriptivos (mantener original)
# ya está en P4. Género — no hace falta columna nueva

n_mujer  = (df["GENERO_MUJER"] == 1).sum()
n_hombre = (df["GENERO_MUJER"] == 0).sum()
n_na_gen = df["GENERO_MUJER"].isnull().sum()
print(f"  GENERO_MUJER → Mujer: {n_mujer} | Hombre: {n_hombre} | Excluidos SEM: {n_na_gen}")

# Origen: dummy ORIGEN_RURAL
# Rural = Pueblo rural + Comunidad indígena
# Urbano = Ciudad grande + Cabecera municipal pequeña
df["ORIGEN_RURAL"] = df["P6. Lugar de origen"].map({
    "Pueblo, comisaría o localidad rural (menos de 10,000 habitantes)":          1,
    "Comunidad indígena o rural muy pequeña (menos de 2,500 habitantes)":        1,
    "Cabecera municipal pequeña o ciudad intermedia (10,000 a 50,000 habitantes)": 0,
    "Ciudad o cabecera municipal grande (más de 50,000 habitantes)":             0,
})

n_rural  = (df["ORIGEN_RURAL"] == 1).sum()
n_urbano = (df["ORIGEN_RURAL"] == 0).sum()
print(f"  ORIGEN_RURAL  → Rural: {n_rural} | Urbano/semi-urbano: {n_urbano}")

# Experiencia de acoso (P18): binaria
# 1 = experimentó alguna situación | 0 = no | NaN = prefirió no responder
df["ACOSO_EXP"] = df["P18. Situaciones experimentadas en el ITESCAM"].map({
    "No he experimentado ninguna de estas situaciones": 0,
    "Comentarios ofensivos por parte de compañeros/as": 1,
    "Comentarios ofensivos por parte de docentes":      1,
    "Exclusión deliberada de actividades":              1,
    "Prefiero no responder":                            np.nan,
})

n_acoso    = (df["ACOSO_EXP"] == 1).sum()
n_no_acoso = (df["ACOSO_EXP"] == 0).sum()
n_na_acoso = df["ACOSO_EXP"].isnull().sum()
print(f"  ACOSO_EXP     → Sí: {n_acoso} | No: {n_no_acoso} | PNR: {n_na_acoso}")

# Seguridad en campus (P17): binaria
df["SEGURIDAD_CAMPUS"] = df["P17. ¿Te sientes seguro/a dentro del campus del ITESCAM?"].map({
    "Sí": 1,
    "No": 0,
})

n_seg = (df["SEGURIDAD_CAMPUS"] == 1).sum()
print(f"  SEGURIDAD     → Seguro: {n_seg} | Inseguro: {555 - n_seg}")

# Impacto en salud mental (P20): ordinal 1–4
df["IMPACTO_MENTAL"] = df[
    "P20. ¿El estrés/cansancio ha afectado tu salud emocional o mental?"
].map({
    "No, me he sentido bien en general":               1,
    "Ligeramente, pero lo he manejado bien":           2,
    "Sí, ha sido un semestre emocionalmente difícil":  3,
    "Sí, y me ha costado mucho trabajo manejarlo":     4,
    "Prefiero no responder":                           np.nan,
})

print(f"  IMPACTO_MENTAL (1–4):")
for val in [1, 2, 3, 4]:
    n = (df["IMPACTO_MENTAL"] == val).sum()
    print(f"    {val}: {n}")
print(f"    NaN: {df['IMPACTO_MENTAL'].isnull().sum()}")


# =============================================================================
# 7. ETIQUETAS CORTAS DE CARRERA
# =============================================================================

imprimir_seccion("7. ETIQUETAS DE CARRERA")

df["CARRERA_CORTA"] = df["P2. Carrera"].map(CARRERAS_CORTAS)

# Verificar que todas las carreras se mapearon
na_carrera = df["CARRERA_CORTA"].isnull().sum()
if na_carrera > 0:
    vals_no_mapeados = df.loc[df["CARRERA_CORTA"].isnull(), "P2. Carrera"].unique()
    print(f"  ⚠️  {na_carrera} registros sin carrera mapeada: {vals_no_mapeados}")
else:
    distribucion = df["CARRERA_CORTA"].value_counts()
    print(f"  {'Sigla':<8} {'n':>5}")
    print(f"  {'-'*15}")
    for sig in CARRERAS_ORDEN:
        n = distribucion.get(sig, 0)
        print(f"  {sig:<8} {n:>5}")
    print(f"  {'TOTAL':<8} {distribucion.sum():>5}")


# =============================================================================
# 8. CONOCIMIENTO DE PROTOCOLOS (columna [B] → binaria)
# =============================================================================

imprimir_seccion("8. COLUMNAS BLOQUE [B]")

df["CONOCE_PROTOCOLOS"] = df[
    "[B] ¿Conoces los protocolos institucionales de acoso?"
].map({"Sí": 1, "No": 0})

n_conoce = (df["CONOCE_PROTOCOLOS"] == 1).sum()
print(f"  Conoce protocolos: {n_conoce} ({n_conoce/len(df)*100:.1f}%)")
print(f"  No conoce:         {len(df) - n_conoce} ({(len(df)-n_conoce)/len(df)*100:.1f}%)")


# =============================================================================
# 9. REPORTE FINAL Y GUARDADO
# =============================================================================

imprimir_seccion("9. REPORTE FINAL")

# Columnas nuevas creadas
cols_nuevas = (
    [c + "_num" for c in EA_COLS + FA_COLS + MV_COLS + BP_COLS]
    + [c + "_num" for c in EQ_COLS]
    + [c + "_inv" for c in EQ_ITEMS_INVERSOS]
    + [c + "_final" for c in EQ_COLS]
    + [c + "_num" for c in P16_COLS]
    + ["IDX_EA", "IDX_FA", "IDX_MV", "IDX_EQ", "IDX_BP"]
    + ["P11_num", "P12_num", "P13_num", "P8_num", "SEMESTRE_num"]
    + ["CARGA_TOTAL_raw", "CARGA_TOTAL"]
    + ["GENERO_MUJER", "ORIGEN_RURAL", "ACOSO_EXP",
       "SEGURIDAD_CAMPUS", "IMPACTO_MENTAL", "CARRERA_CORTA",
       "CONOCE_PROTOCOLOS"]
)

print(f"  Columnas originales:  {len(df.columns) - len(cols_nuevas)}")
print(f"  Columnas nuevas:      {len(cols_nuevas)}")
print(f"  Total en dataframe:   {len(df.columns)}")
print(f"  Filas:                {len(df)}")

# Validación final: índices sin NaN
print(f"\n  Índices sin valores perdidos:")
for idx in ["IDX_EA", "IDX_FA", "IDX_MV", "IDX_EQ", "IDX_BP"]:
    n_na = df[idx].isnull().sum()
    estado = "✅" if n_na == 0 else f"⚠️  {n_na} NaN"
    print(f"    {idx}: {estado}")

# Guardar pickle
df.to_pickle(ARCHIVO_PROCESADO)
print(f"\n  ✅ Dataset procesado guardado en:")
print(f"     {ARCHIVO_PROCESADO}")

imprimir_seccion("PREPROCESAMIENTO COMPLETADO")