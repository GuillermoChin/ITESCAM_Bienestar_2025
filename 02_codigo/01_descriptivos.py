"""
01_descriptivos.py
Estadísticos descriptivos y tablas de distribución de la muestra.

Entrada:  03_salidas/modelos/datos_procesados.pkl
Salidas:  03_salidas/tablas/tab01_muestra_demografica.txt
          03_salidas/tablas/tab02_estadisticos_indices.txt
          03_salidas/tablas/tab03_distribucion_carga.txt
          03_salidas/tablas/tab04_intenciones_p16.txt
          03_salidas/tablas/tab05_bloque_sensible.txt
          03_salidas/tablas/tab06_impacto_salud_mental.txt
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from scipy import stats

sys.path.append(str(__import__("pathlib").Path(__file__).parent))
from config import (
    ARCHIVO_PROCESADO, TABLAS_DIR,
    INDICES, NOMBRES_INDICES, NOMBRES_INDICES_EN,
    CARRERAS_ORDEN,
    GENERO_ES_EN, SEMESTRE_ES_EN, ORIGEN_ES_EN, DESEMPENO_ES_EN,
)

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def imprimir_seccion(titulo):
    print(f"\n{'='*60}")
    print(f"  {titulo}")
    print(f"{'='*60}")


def guardar_tabla(contenido: str, nombre_archivo: str):
    """Guarda una tabla en la carpeta de tablas."""
    ruta = TABLAS_DIR / nombre_archivo
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(contenido)
    print(f"  ✅ Guardado: {nombre_archivo}")


def separador(ancho=72, char="─"):
    return char * ancho


def encabezado_bilingue(titulo_es: str, titulo_en: str, ancho=72) -> str:
    """Genera encabezado bilingüe para tablas."""
    lineas = [
        "=" * ancho,
        f"  {titulo_es}",
        f"  {titulo_en}",
        "=" * ancho,
    ]
    return "\n".join(lineas)


def fila_n_pct(etiqueta_es: str, etiqueta_en: str,
               n: int, total: int, ancho_et=38) -> str:
    """Formatea una fila con n y porcentaje."""
    pct = n / total * 100
    return (
        f"  {etiqueta_es:<{ancho_et}} "
        f"{etiqueta_en:<{ancho_et}} "
        f"{n:>5}  {pct:>6.1f}%"
    )


def nota_pie(n_total: int) -> str:
    return (
        f"\n  Nota. n = {n_total}. ITESCAM, Campeche, México (2026).\n"
        f"  Note. n = {n_total}. ITESCAM, Campeche, México (2026).\n"
    )


# =============================================================================
# CARGA DE DATOS
# =============================================================================

imprimir_seccion("CARGANDO DATOS")
df = pd.read_pickle(ARCHIVO_PROCESADO)
N = len(df)
print(f"  Filas: {N} | Columnas: {len(df.columns)}")


# =============================================================================
# TAB01 — MUESTRA DEMOGRÁFICA
# =============================================================================

imprimir_seccion("TAB01 — MUESTRA DEMOGRÁFICA")

lineas = []
lineas.append(encabezado_bilingue(
    "Tabla 1. Características sociodemográficas de la muestra",
    "Table 1. Sociodemographic Characteristics of the Sample"
))

# ── Columna de referencia
col_ref = (
    f"\n  {'Variable (ES)':<38} {'Variable (EN)':<38} "
    f"{'n':>5}  {'%':>6}"
)
lineas.append(col_ref)
lineas.append("  " + separador(70))

# ── GÉNERO
lineas.append("\n  Género / Gender")
for cat_es, cat_en in GENERO_ES_EN.items():
    n = (df["P4. Género"] == cat_es).sum()
    lineas.append(fila_n_pct(f"  {cat_es}", f"  {cat_en}", n, N))

# ── CARRERA
lineas.append("\n  Carrera / Academic Program")
carrera_counts = df["P2. Carrera"].value_counts()
# Orden de mayor a menor para cada carrera por sigla
for sigla in CARRERAS_ORDEN:
    carrera_es = next(
        (k for k, v in __import__("config", fromlist=["CARRERAS_CORTAS"])
         .CARRERAS_CORTAS.items() if v == sigla), sigla
    )
    n = carrera_counts.get(carrera_es, 0)
    lineas.append(fila_n_pct(f"  {carrera_es[:36]}", f"  [{sigla}]", n, N))

# ── SEMESTRE
lineas.append("\n  Semestre / Semester")
for cat_es, cat_en in SEMESTRE_ES_EN.items():
    n = (df["P3. Semestre"] == cat_es).sum()
    lineas.append(fila_n_pct(f"  {cat_es}", f"  {cat_en}", n, N))

# ── EDAD
lineas.append("\n  Edad / Age")
edad_cats = [
    ("17 años o menos", "17 years or younger"),
    ("18 años", "18 years"),
    ("19 años", "19 years"),
    ("20 años", "20 years"),
    ("21 años", "21 years"),
    ("22 años", "22 years"),
    ("23 años", "23 years"),
    ("24 años", "24 years"),
    ("25 años o más", "25 years or older"),
]
for cat_es, cat_en in edad_cats:
    n = (df["P5. Edad"] == cat_es).sum()
    if n > 0:
        lineas.append(fila_n_pct(f"  {cat_es}", f"  {cat_en}", n, N))

# ── LUGAR DE ORIGEN
lineas.append("\n  Lugar de origen / Place of Origin")
for cat_es, cat_en in ORIGEN_ES_EN.items():
    n = (df["P6. Lugar de origen"] == cat_es).sum()
    lineas.append(fila_n_pct(f"  {cat_es[:36]}", f"  {cat_en}", n, N))

# ── DESEMPEÑO ACADÉMICO
lineas.append("\n  Desempeño académico autopercibido / Self-perceived Academic Performance")
for cat_es, cat_en in DESEMPENO_ES_EN.items():
    n = (df["P9. Desempeño académico general"] == cat_es).sum()
    lineas.append(fila_n_pct(f"  {cat_es[:36]}", f"  {cat_en}", n, N))

lineas.append(nota_pie(N))
contenido = "\n".join(lineas)
guardar_tabla(contenido, "tab01_muestra_demografica.txt")


# =============================================================================
# TAB02 — ESTADÍSTICOS DESCRIPTIVOS DE ÍNDICES
# =============================================================================

imprimir_seccion("TAB02 — ESTADÍSTICOS DE ÍNDICES")

lineas = []
lineas.append(encabezado_bilingue(
    "Tabla 2. Estadísticos descriptivos de los índices por género",
    "Table 2. Descriptive Statistics of Indices by Gender"
))

# ── Total
lineas.append(
    f"\n  {'Índice / Index':<28} "
    f"{'M':>6} {'DE/SD':>6} {'Mín':>5} {'Máx':>5} "
    f"{'Asim./Skew.':>11} {'Curt./Kurt.':>11}"
)
lineas.append("  " + separador(76))
lineas.append("  Total sample (n = 555)")

for idx in INDICES:
    nombre_es = NOMBRES_INDICES[idx]
    nombre_en = NOMBRES_INDICES_EN[idx]
    serie = df[idx].dropna()
    m   = serie.mean()
    s   = serie.std()
    mn  = serie.min()
    mx  = serie.max()
    sk  = stats.skew(serie)
    ku  = stats.kurtosis(serie)
    lineas.append(
        f"  {nombre_es[:26]:<28} "
        f"{m:>6.3f} {s:>6.3f} {mn:>5.3f} {mx:>5.3f} "
        f"{sk:>11.3f} {ku:>11.3f}"
    )
    lineas.append(
        f"  {nombre_en[:26]:<28}"
    )
    lineas.append("  " + separador(76, "·"))

# ── Por género (Mujer / Hombre)
for genero_es, genero_en in [("Mujer", "Women"), ("Hombre", "Men")]:
    sub = df[df["P4. Género"] == genero_es]
    n_g = len(sub)
    lineas.append(f"\n  {genero_es} / {genero_en} (n = {n_g})")
    lineas.append("  " + separador(76))
    for idx in INDICES:
        nombre_es = NOMBRES_INDICES[idx]
        nombre_en = NOMBRES_INDICES_EN[idx]
        serie = sub[idx].dropna()
        m  = serie.mean()
        s  = serie.std()
        mn = serie.min()
        mx = serie.max()
        sk = stats.skew(serie)
        ku = stats.kurtosis(serie)
        lineas.append(
            f"  {nombre_es[:26]:<28} "
            f"{m:>6.3f} {s:>6.3f} {mn:>5.3f} {mx:>5.3f} "
            f"{sk:>11.3f} {ku:>11.3f}"
        )

lineas.append(nota_pie(N))
lineas.append(
    "  Nota. M = media; DE = desviación estándar; Asim. = asimetría; Curt. = curtosis.\n"
    "  Note. M = mean; SD = standard deviation; Skew. = skewness; Kurt. = kurtosis.\n"
    "  Escala teórica / Theoretical range: 1–5.\n"
)
contenido = "\n".join(lineas)
guardar_tabla(contenido, "tab02_estadisticos_indices.txt")


# =============================================================================
# TAB03 — DISTRIBUCIÓN DE VARIABLES DE CARGA
# =============================================================================

imprimir_seccion("TAB03 — VARIABLES DE CARGA")

lineas = []
lineas.append(encabezado_bilingue(
    "Tabla 3. Distribución de variables de carga estudiantil",
    "Table 3. Distribution of Student Workload Variables"
))

bloques = [
    (
        "Horas de trabajo remunerado / Paid work hours (weekly)",
        "P11. Horas de trabajo a la semana",
        [
            ("No trabajo",                                           "No paid work"),
            ("Menos de 5 horas a la semana",                        "Less than 5 h/week"),
            ("Entre 5 y 10 horas a la semana",                      "5–10 h/week"),
            ("Entre 10 y 20 horas a la semana",                     "10–20 h/week"),
            ("Entre 20 y 30 horas a la semana",                     "20–30 h/week"),
            ("Más de 30 horas a la semana (trabajo tiempo completo)","More than 30 h/week (full-time)"),
        ]
    ),
    (
        "Horas de labores del hogar / Household chores (weekly)",
        "P12. Horas de labores del hogar/semana",
        [
            ("Ninguna",                          "None"),
            ("Menos de 5 horas a la semana",     "Less than 5 h/week"),
            ("Entre 5 y 10 horas a la semana",   "5–10 h/week"),
            ("Entre 10 y 20 horas a la semana",  "10–20 h/week"),
            ("Más de 20 horas a la semana",      "More than 20 h/week"),
        ]
    ),
    (
        "Horas de cuidado de familiares / Family caregiving (weekly)",
        "P13. Horas de cuidado de familiares/semana",
        [
            ("Ninguna",                          "None"),
            ("Menos de 5 horas a la semana",     "Less than 5 h/week"),
            ("Entre 5 y 10 horas a la semana",   "5–10 h/week"),
            ("Entre 10 y 20 horas a la semana",  "10–20 h/week"),
            ("Más de 20 horas a la semana",      "More than 20 h/week"),
        ]
    ),
    (
        "Horas de sueño / Sleep hours (per night)",
        "P8. Horas de sueño",
        [
            ("Menos de 4 horas",   "Less than 4 h"),
            ("Entre 4 y 5 horas",  "4–5 h"),
            ("Entre 6 y 7 horas",  "6–7 h"),
            ("Entre 7 y 8 horas",  "7–8 h"),
            ("Más de 8 horas",     "More than 8 h"),
        ]
    ),
]

col_ref = (
    f"\n  {'Categoría (ES)':<40} {'Category (EN)':<34} "
    f"{'n':>5}  {'%':>6}"
)
lineas.append(col_ref)
lineas.append("  " + separador(88))

for titulo, col, categorias in bloques:
    lineas.append(f"\n  {titulo}")
    for cat_es, cat_en in categorias:
        n = (df[col] == cat_es).sum()
        lineas.append(fila_n_pct(f"  {cat_es[:38]}", f"  {cat_en}", n, N, ancho_et=40))

# Estadísticos de CARGA_TOTAL
lineas.append(f"\n  {'─'*72}")
lineas.append(
    f"  Índice de carga total (0–1) / Total workload index (0–1)\n"
    f"  M = {df['CARGA_TOTAL'].mean():.3f} | "
    f"DE/SD = {df['CARGA_TOTAL'].std():.3f} | "
    f"Mín/Min = {df['CARGA_TOTAL'].min():.3f} | "
    f"Máx/Max = {df['CARGA_TOTAL'].max():.3f}"
)

lineas.append(nota_pie(N))
contenido = "\n".join(lineas)
guardar_tabla(contenido, "tab03_distribucion_carga.txt")


# =============================================================================
# TAB04 — INTENCIONES P16
# =============================================================================

imprimir_seccion("TAB04 — INTENCIONES P16")

p16_info = [
    ("P16a. ¿Qué tan probable: terminar a tiempo?",
     "Probability of graduating on time"),
    ("P16b. ¿Qué tan probable: abandonar o interrumpir estudios?",
     "Probability of dropping out"),
    ("P16c. ¿Qué tan probable: continuar con un posgrado?",
     "Probability of pursuing graduate studies"),
    ("P16d. ¿Qué tan probable: trabajar en tu área al titularte?",
     "Probability of working in field after graduation"),
    ("P16e. ¿Qué tan probable: cambiar de carrera o institución?",
     "Probability of changing program or institution"),
]

cats_prob = [
    ("Muy improbable", "Very unlikely"),
    ("Improbable",     "Unlikely"),
    ("No lo sé",       "Not sure"),
    ("Probable",       "Likely"),
    ("Muy probable",   "Very likely"),
]

lineas = []
lineas.append(encabezado_bilingue(
    "Tabla 4. Distribución de intenciones y expectativas (P16)",
    "Table 4. Distribution of Intentions and Expectations (P16)"
))

col_ref = (
    f"\n  {'Categoría (ES)':<18} {'Category (EN)':<18} "
    f"{'Total':>7}"
    f"{'Mujer/W':>9}"
    f"{'Hombre/M':>10}"
)
lineas.append(col_ref)

for col_es, titulo_en in p16_info:
    lineas.append(f"\n  {col_es}")
    lineas.append(f"  {titulo_en}")
    lineas.append("  " + separador(64))

    sub_m = df[df["P4. Género"] == "Mujer"]
    sub_h = df[df["P4. Género"] == "Hombre"]

    for cat_es, cat_en in cats_prob:
        n_tot = (df[col_es] == cat_es).sum()
        n_m   = (sub_m[col_es] == cat_es).sum()
        n_h   = (sub_h[col_es] == cat_es).sum()
        p_tot = n_tot / N * 100
        lineas.append(
            f"  {cat_es:<18} {cat_en:<18} "
            f"{n_tot:>4} ({p_tot:>4.1f}%)"
            f"  {n_m:>4}"
            f"  {n_h:>6}"
        )

    # Media del índice numérico
    col_num = col_es + "_num"
    m_tot = df[col_num].mean()
    m_m   = sub_m[col_num].mean()
    m_h   = sub_h[col_num].mean()
    lineas.append(
        f"\n  M (1–5) / Mean (1–5): "
        f"Total = {m_tot:.2f} | "
        f"Mujer/Women = {m_m:.2f} | "
        f"Hombre/Men = {m_h:.2f}"
    )

lineas.append(nota_pie(N))
contenido = "\n".join(lineas)
guardar_tabla(contenido, "tab04_intenciones_p16.txt")


# =============================================================================
# TAB05 — BLOQUE SENSIBLE
# =============================================================================

imprimir_seccion("TAB05 — BLOQUE SENSIBLE")

lineas = []
lineas.append(encabezado_bilingue(
    "Tabla 5. Seguridad, acoso, apoyo institucional e identidad",
    "Table 5. Safety, Harassment, Institutional Support and Identity"
))

col_ref = (
    f"\n  {'Categoría (ES)':<46} {'Category (EN)':<36} "
    f"{'n':>5}  {'%':>6}"
)
lineas.append(col_ref)
lineas.append("  " + separador(96))

# P17 — Seguridad en campus
lineas.append("\n  Percepción de seguridad en el campus / Campus safety perception (P17)")
for cat_es, cat_en in [("Sí", "Yes"), ("No", "No")]:
    n = (df["P17. ¿Te sientes seguro/a dentro del campus del ITESCAM?"] == cat_es).sum()
    lineas.append(fila_n_pct(f"  {cat_es}", f"  {cat_en}", n, N, ancho_et=46))

# P18 — Situaciones de acoso
lineas.append("\n  Situaciones experimentadas / Experienced situations (P18)")
p18_cats = [
    ("No he experimentado ninguna de estas situaciones",
     "None experienced"),
    ("Comentarios ofensivos por parte de compañeros/as",
     "Offensive comments from peers"),
    ("Comentarios ofensivos por parte de docentes",
     "Offensive comments from faculty"),
    ("Exclusión deliberada de actividades",
     "Deliberate exclusion from activities"),
    ("Prefiero no responder",
     "Prefer not to answer"),
]
for cat_es, cat_en in p18_cats:
    n = (df["P18. Situaciones experimentadas en el ITESCAM"] == cat_es).sum()
    lineas.append(fila_n_pct(f"  {cat_es[:44]}", f"  {cat_en}", n, N, ancho_et=46))

# P19 — Disposición a reportar
lineas.append("\n  Disposición a reportar situaciones / Willingness to report (P19)")
p19_cats = [
    ("No aplica, no he vivido esas situaciones",
     "Not applicable"),
    ("Muy cómodo/a, confío en que se atendería adecuadamente",
     "Very comfortable, trust in institutional response"),
    ("Algo cómodo/a, aunque con dudas",
     "Somewhat comfortable, with doubts"),
    ("No estoy seguro/a de qué pasaría si lo reporto",
     "Unsure of what would happen"),
    ("Poco cómodo/a, no creo que se atendiera bien",
     "Uncomfortable, doubt proper response"),
    ("No lo reportaría bajo ninguna circunstancia",
     "Would not report under any circumstance"),
]
for cat_es, cat_en in p19_cats:
    n = (df["P19. ¿Qué tan cómodo/a reportarías una situación incómoda?"] == cat_es).sum()
    lineas.append(fila_n_pct(f"  {cat_es[:44]}", f"  {cat_en}", n, N, ancho_et=46))

# [B] Factor desencadenante de estrés
lineas.append("\n  Principal factor de estrés / Main stress trigger ([B])")
b_estres_cats = [
    ("Sobrecarga de tareas y proyectos", "Task and project overload"),
    ("Exámenes y evaluaciones",          "Exams and assessments"),
    ("Falta de tiempo para estudiar",    "Lack of study time"),
]
for cat_es, cat_en in b_estres_cats:
    n = (df["[B] Principal factor desencadenante de estrés"] == cat_es).sum()
    lineas.append(fila_n_pct(f"  {cat_es}", f"  {cat_en}", n, N, ancho_et=46))

# [B] Tipo de apoyo deseado
lineas.append("\n  Apoyo institucional deseado / Desired institutional support ([B])")
b_apoyo_cats = [
    ("Talleres de manejo de estrés",         "Stress management workshops"),
    ("Asesoría psicológica institucional",   "Institutional psychological counseling"),
    ("Actividades deportivas y culturales",  "Sports and cultural activities"),
    ("Ninguno",                              "None"),
]
for cat_es, cat_en in b_apoyo_cats:
    n = (df["[B] Tipo de apoyo institucional que desearías"] == cat_es).sum()
    lineas.append(fila_n_pct(f"  {cat_es}", f"  {cat_en}", n, N, ancho_et=46))

# [B] Conocimiento de protocolos
lineas.append("\n  Conocimiento de protocolos de acoso / Knowledge of harassment protocols ([B])")
for cat_es, cat_en in [("Sí", "Yes"), ("No", "No")]:
    n = (df["[B] ¿Conoces los protocolos institucionales de acoso?"] == cat_es).sum()
    lineas.append(fila_n_pct(f"  {cat_es}", f"  {cat_en}", n, N, ancho_et=46))

# P21 — Identidad
lineas.append("\n  Pertenencia a grupo minoritario / Minority group membership (P21)")
for cat_es, cat_en in [
    ("Sí",                   "Yes"),
    ("No",                   "No"),
    ("Prefiero no responder","Prefer not to answer"),
]:
    n = (df["P21. ¿Te identificas con alguno de estos grupos? (opcional)"] == cat_es).sum()
    lineas.append(fila_n_pct(f"  {cat_es}", f"  {cat_en}", n, N, ancho_et=46))

lineas.append(nota_pie(N))
contenido = "\n".join(lineas)
guardar_tabla(contenido, "tab05_bloque_sensible.txt")


# =============================================================================
# TAB06 — IMPACTO EN SALUD MENTAL (P20 × GÉNERO × SEMESTRE)
# =============================================================================

imprimir_seccion("TAB06 — IMPACTO EN SALUD MENTAL")

col_p20 = "P20. ¿El estrés/cansancio ha afectado tu salud emocional o mental?"
p20_cats = [
    ("No, me he sentido bien en general",               "No impact"),
    ("Ligeramente, pero lo he manejado bien",           "Slight, manageable"),
    ("Sí, ha sido un semestre emocionalmente difícil",  "Yes, difficult semester"),
    ("Sí, y me ha costado mucho trabajo manejarlo",     "Yes, very hard to manage"),
    ("Prefiero no responder",                           "Prefer not to answer"),
]

lineas = []
lineas.append(encabezado_bilingue(
    "Tabla 6. Impacto en salud mental por género y semestre (P20)",
    "Table 6. Mental Health Impact by Gender and Semester (P20)"
))

# ── Total
lineas.append(
    f"\n  {'Categoría (ES)':<46} {'Category (EN)':<28} "
    f"{'n':>5}  {'%':>6}"
)
lineas.append("  " + separador(88))
lineas.append("\n  Total")
for cat_es, cat_en in p20_cats:
    n = (df[col_p20] == cat_es).sum()
    lineas.append(fila_n_pct(f"  {cat_es[:44]}", f"  {cat_en}", n, N, ancho_et=46))

# ── Por género
for genero_es, genero_en in [("Mujer","Women"), ("Hombre","Men")]:
    sub = df[df["P4. Género"] == genero_es]
    n_g = len(sub)
    lineas.append(f"\n  {genero_es} / {genero_en} (n = {n_g})")
    lineas.append("  " + separador(88))
    for cat_es, cat_en in p20_cats:
        n = (sub[col_p20] == cat_es).sum()
        lineas.append(fila_n_pct(f"  {cat_es[:44]}", f"  {cat_en}", n, n_g, ancho_et=46))

# ── Por semestre
for sem_es, sem_en in SEMESTRE_ES_EN.items():
    sub = df[df["P3. Semestre"] == sem_es]
    n_s = len(sub)
    lineas.append(f"\n  {sem_es} / {sem_en} (n = {n_s})")
    lineas.append("  " + separador(88))
    for cat_es, cat_en in p20_cats:
        n = (sub[col_p20] == cat_es).sum()
        lineas.append(fila_n_pct(f"  {cat_es[:44]}", f"  {cat_en}", n, n_s, ancho_et=46))

# ── Medias de índices por categoría P20
lineas.append(f"\n  {'─'*88}")
lineas.append("  Medias de índices por nivel de impacto / Index means by impact level")
lineas.append(
    f"\n  {'Categoría':<46} "
    + "  ".join(f"{'IDX_'+i[4:]:>8}" for i in INDICES)
)
lineas.append("  " + separador(88))
for cat_es, cat_en in p20_cats[:-1]:  # excluir PNR
    sub = df[df[col_p20] == cat_es]
    medias = [f"{sub[idx].mean():>8.3f}" for idx in INDICES]
    lineas.append(
        f"  {cat_en[:44]:<46} " + "  ".join(medias)
    )

lineas.append(nota_pie(N))
lineas.append(
    "  Nota. IDX_EA = Estrés Académico; IDX_FA = Fatiga; IDX_MV = Motivación;\n"
    "        IDX_EQ = Equidad; IDX_BP = Bienestar Percibido.\n"
    "  Note. IDX_EA = Academic Stress; IDX_FA = Fatigue; IDX_MV = Motivation;\n"
    "        IDX_EQ = Equity; IDX_BP = Perceived Well-being.\n"
)
contenido = "\n".join(lineas)
guardar_tabla(contenido, "tab06_impacto_salud_mental.txt")


# =============================================================================
# RESUMEN FINAL
# =============================================================================

imprimir_seccion("RESUMEN — DESCRIPTIVOS COMPLETADOS")
tablas = [
    "tab01_muestra_demografica.txt",
    "tab02_estadisticos_indices.txt",
    "tab03_distribucion_carga.txt",
    "tab04_intenciones_p16.txt",
    "tab05_bloque_sensible.txt",
    "tab06_impacto_salud_mental.txt",
]
for t in tablas:
    ruta = TABLAS_DIR / t
    existe = "✅" if ruta.exists() else "❌"
    print(f"  {existe} {t}")