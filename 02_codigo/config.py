"""
config.py — Fuente única de verdad
Proyecto: Bienestar Académico Estudiantil ITESCAM 2025
Autor: Guillermo Adrián Chin Canché
Institución: ITESCAM / KAABLAB
"""

from pathlib import Path

# =============================================================================
# 1. RUTAS
# =============================================================================

BASE_DIR        = Path(__file__).parent.parent
DATOS_DIR       = BASE_DIR / "01_datos"
SALIDAS_DIR     = BASE_DIR / "03_salidas"

TABLAS_DIR      = SALIDAS_DIR / "tablas"
MODELOS_DIR     = SALIDAS_DIR / "modelos"
FIGURAS_PUB_DIR = SALIDAS_DIR / "figuras_pub"
FIGURAS_EXP_DIR = SALIDAS_DIR / "figuras_exp"

ARCHIVO_DATOS     = DATOS_DIR  / "Dataset_Final_ITESCAM_2025.xlsx"
ARCHIVO_PROCESADO = MODELOS_DIR / "datos_procesados.pkl"

for d in [TABLAS_DIR, MODELOS_DIR, FIGURAS_PUB_DIR, FIGURAS_EXP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 2. ÍTEMS POR ESCALA
# =============================================================================

EA_COLS = [
    "EA1. Las tareas y proyectos me generan presión o angustia",
    "EA2. La cercanía de exámenes me provoca nerviosismo o ansiedad",
    "EA3. La exigencia está por encima de mis capacidades actuales",
    "EA4. Me preocupa no alcanzar las calificaciones necesarias",
    "EA5. El tiempo no me alcanza para cumplir obligaciones escolares",
    "EA6. Me siento sobrecargado/a por materias y contenidos",
    "EA7. La presión me genera síntomas físicos o emocionales",
    "EA8. No puedo descansar mentalmente por pendientes escolares",
]

FA_COLS = [
    "FA1. Me siento emocionalmente agotado/a al final del día escolar",
    "FA2. El cansancio afecta mi concentración en clase o al estudiar",
    "FA3. Llego a casa sin energía para hacer cualquier otra cosa",
    "FA4. Me despierto cansado/a incluso después de dormir",
    "FA5. Siento que mis pilas están al límite",
    "FA6. Me cuesta mantener la atención durante una clase completa",
    "FA7. He tenido ganas de faltar a clases para recuperarme",
    "FA8. El cansancio acumulado afecta la calidad de mi trabajo académico",
]

MV_COLS = [
    "MV1. Tengo claro por qué estoy estudiando esta carrera",
    "MV2. Lo que aprendo es útil para mi futuro profesional",
    "MV3. Tengo metas académicas o profesionales concretas",
    "MV4. Me imagino terminando mi carrera y alcanzando mis objetivos",
    "MV5. Cuando las cosas se ponen difíciles encuentro razones para seguir",
    "MV6. Me interesa continuar estudiando (posgrado, certificaciones)",
    "MV7. Tengo expectativas positivas sobre mis oportunidades laborales",
    "MV8. Pese al estrés/cansancio sigo comprometido/a con terminar",
]

EQ_COLS = [
    "EQ1. Soy tratado/a de manera justa por mis docentes",
    "EQ2. Las calificaciones reflejan de forma objetiva mi desempeño",
    "EQ3. Todos tienen las mismas oportunidades de participar en clase",
    "EQ4. He percibido que algunos estudiantes reciben trato preferencial",  # INVERSO
    "EQ5. Me siento cómodo/a participando y expresando mis opiniones",
    "EQ6. El ambiente en mi carrera es inclusivo y respetuoso",
    "EQ7. Se espera menos/más de mí por razones ajenas a mi desempeño",     # INVERSO
    "EQ8. El ambiente académico me hace sentir que pertenezco aquí",
]

EQ_ITEMS_INVERSOS = [
    "EQ4. He percibido que algunos estudiantes reciben trato preferencial",
    "EQ7. Se espera menos/más de mí por razones ajenas a mi desempeño",
]

BP_COLS = [
    "BP1. En la mayoría de aspectos, mi vida está como yo quisiera",
    "BP2. Las condiciones de mi vida son excelentes",
    "BP3. Estoy satisfecho/a con mi vida en este momento",
    "BP4. He logrado las cosas importantes que quería en la vida",
    "BP5. Si pudiera volver a vivir mi vida no cambiaría casi nada",
]

ALL_LIKERT_COLS = EA_COLS + FA_COLS + MV_COLS + EQ_COLS + BP_COLS

P16_COLS = [
    "P16a. ¿Qué tan probable: terminar a tiempo?",
    "P16b. ¿Qué tan probable: abandonar o interrumpir estudios?",
    "P16c. ¿Qué tan probable: continuar con un posgrado?",
    "P16d. ¿Qué tan probable: trabajar en tu área al titularte?",
    "P16e. ¿Qué tan probable: cambiar de carrera o institución?",
]

INDICES = ["IDX_EA", "IDX_FA", "IDX_MV", "IDX_EQ", "IDX_BP"]

NOMBRES_INDICES = {
    "IDX_EA": "Estrés Académico",
    "IDX_FA": "Fatiga / Agotamiento",
    "IDX_MV": "Motivación",
    "IDX_EQ": "Percepción de Equidad",
    "IDX_BP": "Bienestar Percibido",
}

NOMBRES_INDICES_CORTOS = {
    "IDX_EA": "Estrés",
    "IDX_FA": "Fatiga",
    "IDX_MV": "Motivación",
    "IDX_EQ": "Equidad",
    "IDX_BP": "Bienestar",
}

# =============================================================================
# 3. MAPEOS TEXTO → NUMÉRICO
# =============================================================================

LIKERT_FRECUENCIA = {
    "Nunca": 1,
    "Pocas veces": 2,
    "Algunas veces": 3,
    "Frecuentemente": 4,
    "Siempre o casi siempre": 5,
}

LIKERT_ACUERDO = {
    "Totalmente en desacuerdo": 1,
    "En desacuerdo": 2,
    "Ni de acuerdo ni en desacuerdo": 3,
    "De acuerdo": 4,
    "Totalmente de acuerdo": 5,
}

LIKERT_PROBABILIDAD = {
    "Muy improbable": 1,
    "Improbable": 2,
    "No lo sé": 3,
    "Probable": 4,
    "Muy probable": 5,
}

HORAS_TRABAJO = {
    "No trabajo": 0,
    "Menos de 5 horas a la semana": 1,
    "Entre 5 y 10 horas a la semana": 2,
    "Entre 10 y 20 horas a la semana": 3,
    "Entre 20 y 30 horas a la semana": 4,
    "Más de 30 horas a la semana (trabajo tiempo completo)": 5,
}

HORAS_HOGAR = {
    "Ninguna": 0,
    "Menos de 5 horas a la semana": 1,
    "Entre 5 y 10 horas a la semana": 2,
    "Entre 10 y 20 horas a la semana": 3,
    "Más de 20 horas a la semana": 4,
}

HORAS_CUIDADO = {
    "Ninguna": 0,
    "Menos de 5 horas a la semana": 1,
    "Entre 5 y 10 horas a la semana": 2,
    "Entre 10 y 20 horas a la semana": 3,
    "Más de 20 horas a la semana": 4,
}

HORAS_SUENO = {
    "Menos de 4 horas": 1,
    "Entre 4 y 5 horas": 2,
    "Entre 6 y 7 horas": 3,
    "Entre 7 y 8 horas": 4,
    "Más de 8 horas": 5,
}

SEMESTRE_ORD = {
    "Segundo Semestre": 2,
    "Cuarto Semestre": 4,
    "Sexto Semestre": 6,
    "Octavo Semestre": 8,
}

# =============================================================================
# 4. ETIQUETAS DE CARRERAS
# =============================================================================

CARRERAS_CORTAS = {
    "Ingeniería en Sistemas Computacionales":         "ISC",
    "Ingeniería Bioquímica":                          "BQ",
    "Ingeniería Industrial":                          "IND",
    "Ingeniería en Mecatrónica":                      "MEC",
    "Ingeniería Informática":                         "INF",
    "Ingeniería en Industrias Alimentarias":          "IA",
    "Ingeniería en Innovación Agrícola Sustentable":  "IAS",
    "Ingeniería en Materiales":                       "MAT",
}

CARRERAS_ORDEN = ["ISC", "BQ", "IND", "MEC", "INF", "IA", "IAS", "MAT"]

# =============================================================================
# 5. PALETAS DE COLOR
# =============================================================================

COLORES_INDICES = {
    "IDX_EA": "#E05C5C",
    "IDX_FA": "#E08C3A",
    "IDX_MV": "#4DAF7C",
    "IDX_EQ": "#5B8DB8",
    "IDX_BP": "#9B70C4",
}

LISTA_COLORES_INDICES = list(COLORES_INDICES.values())

COLORES_GENERO = {
    "Mujer":                  "#D96B9A",
    "Hombre":                 "#5B8DB8",
    "No binario":             "#6FBFA0",
    "Prefiero no responder":  "#AAAAAA",
}

COLORES_CARRERAS = {
    "ISC": "#4E79A7",
    "BQ":  "#F28E2B",
    "IND": "#E15759",
    "MEC": "#76B7B2",
    "INF": "#59A14F",
    "IA":  "#EDC948",
    "IAS": "#B07AA1",
    "MAT": "#FF9DA7",
}

COLORES_SEMESTRES = {2: "#B3C9E8", 4: "#6499CC", 6: "#3469A0", 8: "#0D3768"}

# =============================================================================
# 6. PARÁMETROS DE FIGURAS
# =============================================================================

DPI_PUB  = 300
DPI_EXP  = 150
FMT_PUB  = "png"
FMT_EXP  = "png"

FIGSIZE_GRANDE   = (14, 10)
FIGSIZE_MEDIANA  = (12, 7)
FIGSIZE_ANGOSTA  = (8, 10)
FIGSIZE_CUADRADA = (10, 10)

FS_TITULO  = 14
FS_EJE     = 12
FS_TICK    = 10
FS_LEYENDA = 10

ESTILO_BASE = "seaborn-v0_8-whitegrid"

# =============================================================================
# 7. UMBRALES ESTADÍSTICOS
# =============================================================================

ALPHA_CRONBACH_MIN  = 0.70
ALPHA_CRONBACH_OK   = 0.80
CFI_MIN             = 0.95
TLI_MIN             = 0.95
RMSEA_MAX           = 0.08
SRMR_MAX            = 0.08
AVE_MIN             = 0.50
ALPHA_SIG           = 0.05
N_BOOTSTRAP         = 1000

# =============================================================================
# 8. TEXTOS DE PIE DE FIGURA
# =============================================================================

NOTA_MUESTRA = (
    "Nota. n = 555 estudiantes del ITESCAM, Campeche, México (2026). "
)
NOTA_SIG = "***p < .001, **p < .01, *p < .05."
NOTA_LIKERT = (
    "Nota. Escala: 1 = Nunca / Totalmente en desacuerdo, "
    "5 = Siempre / Totalmente de acuerdo."
)

# =============================================================================
# 9. ETIQUETAS EN INGLÉS (para figuras y tablas bilingües)
# =============================================================================

NOMBRES_INDICES_EN = {
    "IDX_EA": "Academic Stress",
    "IDX_FA": "Fatigue / Burnout",
    "IDX_MV": "Motivation",
    "IDX_EQ": "Perceived Equity",
    "IDX_BP": "Perceived Well-being",
}

NOMBRES_INDICES_CORTOS_EN = {
    "IDX_EA": "Stress",
    "IDX_FA": "Fatigue",
    "IDX_MV": "Motivation",
    "IDX_EQ": "Equity",
    "IDX_BP": "Well-being",
}

COLORES_INDICES_EN = COLORES_INDICES  # misma paleta, distinto nombre por claridad

CARRERAS_CORTAS_EN = CARRERAS_CORTAS  # las siglas son iguales en ambos idiomas

COLORES_GENERO_EN = {
    "Woman":              "#D96B9A",
    "Man":                "#5B8DB8",
    "Non-binary":         "#6FBFA0",
    "Prefer not to say":  "#AAAAAA",
}

GENERO_ES_EN = {
    "Mujer":                  "Woman",
    "Hombre":                 "Man",
    "No binario":             "Non-binary",
    "Prefiero no responder":  "Prefer not to say",
}

SEMESTRE_ES_EN = {
    "Segundo Semestre": "2nd Semester",
    "Cuarto Semestre":  "4th Semester",
    "Sexto Semestre":   "6th Semester",
    "Octavo Semestre":  "8th Semester",
}

ORIGEN_ES_EN = {
    "Pueblo, comisaría o localidad rural (menos de 10,000 habitantes)":
        "Rural town (< 10,000 inh.)",
    "Cabecera municipal pequeña o ciudad intermedia (10,000 a 50,000 habitantes)":
        "Small city (10k–50k inh.)",
    "Ciudad o cabecera municipal grande (más de 50,000 habitantes)":
        "Large city (> 50,000 inh.)",
    "Comunidad indígena o rural muy pequeña (menos de 2,500 habitantes)":
        "Indigenous / very rural (< 2,500 inh.)",
}

DESEMPENO_ES_EN = {
    "Muy bueno (promedio alto, pocas dificultades)":             "Very good",
    "Bueno (promedio regular, algunas dificultades)":            "Good",
    "Regular (promedio medio, varias dificultades)":             "Average",
    "Bajo (dificultades frecuentes, riesgo de reprobación)":     "Low",
    "No lo sé / No tengo una idea clara":                        "Not sure",
}

# Encabezados bilingües para tablas (ES | EN)
# Formato: clave → (español, inglés)
HEADERS_BILINGUE = {
    "n":             ("n",                   "n"),
    "pct":           ("%",                   "%"),
    "media":         ("Media",               "Mean"),
    "de":            ("DE",                  "SD"),
    "min":           ("Mín.",                "Min."),
    "max":           ("Máx.",                "Max."),
    "asimetria":     ("Asimetría",           "Skewness"),
    "curtosis":      ("Curtosis",            "Kurtosis"),
    "alpha":         ("α de Cronbach",       "Cronbach's α"),
    "omega":         ("ω de McDonald",       "McDonald's ω"),
    "r_it":          ("r ítem-total",        "Item-total r"),
    "lambda":        ("λ (estand.)",         "λ (std.)"),
    "error_std":     ("Error estándar",      "Std. Error"),
    "valor_z":       ("z",                   "z"),
    "p_valor":       ("p",                   "p"),
    "r_pearson":     ("r de Pearson",        "Pearson's r"),
    "d_cohen":       ("d de Cohen",          "Cohen's d"),
    "beta_std":      ("β estand.",           "Std. β"),
    "r2":            ("R²",                  "R²"),
    "r2_adj":        ("R² ajust.",           "Adj. R²"),
    "ic_95":         ("IC 95%",              "95% CI"),
    "cfi":           ("CFI",                 "CFI"),
    "tli":           ("TLI",                 "TLI"),
    "rmsea":         ("RMSEA",               "RMSEA"),
    "srmr":          ("SRMR",                "SRMR"),
    "chi2":          ("χ²",                  "χ²"),
    "gl":            ("gl",                  "df"),
    "ave":           ("VME",                 "AVE"),
    "cr":            ("FC",                  "CR"),
    "carrera":       ("Carrera",             "Program"),
    "semestre":      ("Semestre",            "Semester"),
    "genero":        ("Género",              "Gender"),
    "origen":        ("Lugar de origen",     "Place of origin"),
    "edad":          ("Edad",                "Age"),
    "indice":        ("Índice",              "Index"),
    "constructo":    ("Constructo",          "Construct"),
    "item":          ("Ítem",                "Item"),
    "variable":      ("Variable",            "Variable"),
    "categoria":     ("Categoría",           "Category"),
    "frecuencia":    ("Frecuencia",          "Frequency"),
}

# Notas al pie para figuras en inglés
NOTA_MUESTRA_EN = (
    "Note. n = 555 students from ITESCAM, Campeche, México (2026)."
)
NOTA_SIG_EN = "***p < .001, **p < .01, *p < .05."
NOTA_LIKERT_EN = (
    "Note. Scale: 1 = Never / Strongly disagree, "
    "5 = Always / Strongly agree."
)