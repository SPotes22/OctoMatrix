# OctoMatrix

OctoMatrix is a research-oriented Machine Learning pipeline designed to detect
malicious web payloads based on OWASP Top 10 attack patterns.

## What this is
- ML research pipeline
- Synthetic + heuristic dataset
- RandomForest-based classifier
- Exportable `.pkl` model (82%+ accuracy)

## What this is NOT
- A WAF replacement
- A production IDS
- A real-time SOC system

## Quick start
```bash
pip install -r requirements.txt
python octomatrix_pipeline.py
```
Output

* `output/security_model.pkl`

* `output/training_dataset.csv`

# OctoMatrix — Sistema de Detección de Ataques Web con Machine Learning

OctoMatrix es un pipeline de seguridad basado en ML para la detección de ataques web
(SQL Injection, XSS, Path Traversal, Command Injection, XXE, entre otros).

## Características
- Pipeline completo de entrenamiento ML
- Datos sintéticos estilo Kaggle + OWASP
- Modelo RandomForest con características avanzadas
- Sistema de inferencia en tiempo real
- Exportación a modelos `.pkl` para despliegue

## Instalación
```bash
pip install -r requirements.txt
```

Uso rápido
```
from octomatrix.pipeline import SecurityMLPipeline

pipeline = SecurityMLPipeline()
pipeline.collect_kaggle_style_data()
pipeline.train_model()
pipeline.quick_test()
```
---

Estructura del proyecto

```
ARCHITECTURE.md
LICENSEMODEL.md
octomatrix_pipeline.py
PIPELINE.md
README.md
requirements.txt
docs/
- informe_laboratorio.pdf
- Manual_de_implementacion.pdf
- Paper_Octomatrix.pdf
    -research/
        - sources.md
```
---

1. `octomatrix/pipeline/octomatrix_pipeline.py`

Archivo principal del **pipeline de entrenamiento**.

# (código completo en octomatrix_pipeline.py)


Responsabilidades:

* Recolección de datos

* Entrenamiento del modelo

* Evaluación

* Exportación del .pkl

2. requirements.txt

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
scipy>=1.10.0
requests>=2.28.0
```

3. PIPELINE.md

Documento conceptual, no técnico.

Contenido:

* Flujo de datos

* Extracción de features

* Entrenamiento

* Exportación

Uso

4. research/sources.md
# Fuentes de datos Kaggle-style


Responsabilidad:

* Justificar datasets

* Transparencia académica

* No código

5. ARCHITECTURE.md
# Arquitectura de OctoMatrix

Describe:

* Pipeline

* Inference

* Utilities

* Separación de responsabilidades

6. docs/Manual_de_implementacion.pdf

Documento paso a paso para humanos.

Incluye:

- Instalación

- Entrenamiento

- Inferencia

- Scripts

- Personalización

7. LICENSEMODEL.md

Modelo de licencia ética + técnica.

Incluye:

- Libre uso

- Disclaimer de confianza

- Prohibición bélica

- Responsabilidad del modificador

8. .gitignore

```
# Python
__pycache__/
*.py[cod]

# Modelos
*.pkl
models/
!models/.gitkeep
```

9. run_pipeline.py (opcional)
```
from octomatrix.pipeline.security_ml_pipeline import run_complete_pipeline

if __name__ == "__main__":
    run_complete_pipeline()
```

10. agradecimientos

- semillero de investigacion en ciberseguridad BlackByte UTP

