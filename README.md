# Bienestar Académico Estudiantil — ITESCAM 2025

Proyecto de investigación cuantitativa sobre estrés, fatiga, motivación, equidad y 
bienestar percibido en estudiantes del Instituto Tecnológico Superior de Calkiní (ITESCAM).

**Autor:** Guillermo Adrián Chin Canché  
**Institución:** ITESCAM — Laboratorio de Ambientes Inteligentes  
**ORCID:** 0000-0003-2104-6625  
**Estado:** En análisis / preparación para publicación

---

## Estructura del proyecto
ITESCAM_Bienestar_2025/
├── 01_datos/              # Dataset (no incluido en el repositorio)
├── 02_codigo/             # Scripts de análisis Python
├── 03_salidas/            # Outputs generados (tablas, figuras, modelos)
└── 04_recursos/           # Notas metodológicas y referencias

## Requisitos

```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels semopy pingouin scikit-learn networkx
```

## Ejecución

```bash
# Pipeline completo
python 02_codigo/pipeline.py

# Script individual
python 02_codigo/pipeline.py --solo 02_psicometria

# Desde un script en adelante
python 02_codigo/pipeline.py --desde 05_sem
```

## Nota sobre los datos

El dataset no se incluye en este repositorio por razones de confidencialidad 
y protección de datos de los participantes. Para acceso, contactar al autor.