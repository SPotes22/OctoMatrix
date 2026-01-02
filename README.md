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

Ver ARCHITECTURE.md


---

## 2. `octomatrix/pipeline/security_ml_pipeline.py`

Archivo principal del **pipeline de entrenamiento**.

# (código completo en security_ml_pipeline.py)


Responsabilidades:

* Recolección de datos

* Entrenamiento del modelo

* Evaluación

* Exportación del .pkl

Pruebas rápidas

3. octomatrix/pipeline/feature_extraction.py
# Extrae características avanzadas (no ML)


Responsabilidades:

* Features heurísticas

* Entropía

* Patrones sintácticos

4. octomatrix/utils/entropy.py
# Utilidad matemática pura


Responsabilidad:

* Cálculo de entropía de Shannon

* Sin dependencias del pipeline

5. octomatrix/pipeline/__init__.py

```
from .security_ml_pipeline import SecurityMLPipeline
from .feature_extraction import extract_advanced_features

__all__ = ['SecurityMLPipeline', 'extract_advanced_features']
```

6. octomatrix/inference/inference_engine.py

Motor de inferencia para producción / runtime.

# (código completo del inference engine)


Responsabilidades:

* Cargar .pkl

* Predecir eventos individuales o en batch

* No entrena, solo ejecuta

7. octomatrix/pipeline/export.py
# Serialización controlada del modelo


Responsabilidad:

* Exportación limpia del modelo entrenado

* Metadatos de entrenamiento

8. setup.py
```
from setuptools import setup, find_packages

setup(
    name="octomatrix",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "scipy"
    ],
    python_requires=">=3.8",
)
```

9. requirements.txt

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
scipy>=1.10.0
requests>=2.28.0
```

10. scripts/run_pipeline.sh
```
#!/bin/bash
echo "🚀 Ejecutando pipeline OctoMatrix"
cd "$(dirname "$0")/.."

python -c "
from octomatrix.pipeline import SecurityMLPipeline
pipeline = SecurityMLPipeline()
pipeline.collect_kaggle_style_data()
pipeline.train_model()
pipeline.export_model('models')
```

11. scripts/run_inference.sh
```
#!/bin/bash
echo "🤖 Inferencia OctoMatrix"
cd "$(dirname "$0")/.."
```
12. PIPELINE.md

Documento conceptual, no técnico.

Contenido:

* Flujo de datos

* Extracción de features

* Entrenamiento

* Exportación

Uso

13. research/kaggle_style_datasets.md
# Fuentes de datos Kaggle-style


Responsabilidad:

* Justificar datasets

* Transparencia académica

* No código

14. ARCHITECTURE.md
# Arquitectura de OctoMatrix


Describe:

* Pipeline

* Inference

* Utilities

* Separación de responsabilidades

15. docs/Manual_de_implementacion.pdf

Documento paso a paso para humanos.

Incluye:

- Instalación

- Entrenamiento

- Inferencia

- Scripts

- Personalización

16. LICENSEMODEL.md

Modelo de licencia ética + técnica.

Incluye:

- Libre uso

- Disclaimer de confianza

- Prohibición bélica

- Responsabilidad del modificador

17. .gitignore

```
# Python
__pycache__/
*.py[cod]

# Modelos
*.pkl
models/
!models/.gitkeep
```

18. run_pipeline.py (opcional)
```
from octomatrix.pipeline.security_ml_pipeline import run_complete_pipeline

if __name__ == "__main__":
    run_complete_pipeline()
```

19. agradecimientos

- semillero de investigacion en ciberseguridad BlackByte UTP

