# 🛡️ OctoMatrix — detección de ataques OWASP en tráfico HTTP

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)](https://scikit-learn.org/)

Detector de anomalías OWASP sobre peticiones HTTP, pensado como capa heurística para un WAF o un monitor de tráfico. Clasifica una petición como **normal** o **ataque** con una probabilidad calibrada y un umbral elegido para una tasa de falsos positivos (FPR) objetivo.

> **Qué cambió respecto al POC.** El prototipo inicial se entrenaba con **72 frases escritas a mano** y reportaba "83%" sobre un test de 15 muestras. Esta versión se entrena con **~75.500 peticiones reales** y se evalúa con métricas propias de un WAF. La historia completa (afirmación inflada → realidad) está en [`presentation/README.md`](presentation/README.md).

## Resultados reales

| escenario | qué mide | resultado |
|---|---|---|
| in_source | dentro de las fuentes vistas | ROC-AUC **0,989** · recall **89,6 %** @FPR 1 % |
| in_source | WAF estricto | recall **59,1 %** @FPR 0,1 % |
| cross-source | generalización a una fuente **nueva** | ROC-AUC **0,53** (aún no generaliza) |
| adversarial (MoE) | cobertura 19 categorías OWASP | recall **~100 %** · FPR 0,53 % |
| flujos (CIC-IDS2018) | ataques de red (DoS, fuerza bruta) | ROC-AUC **1,00** (dataset trivialmente separable) |

El hallazgo clave es honesto: el cuello de botella no es la cantidad de ataques, sino la **diversidad de tráfico benigno**. Detalle en [`docs/METRICS.md`](docs/METRICS.md).

## Arquitectura — Mixture-of-Experts por procedencia

Para sumar cobertura OWASP sin que los datos generados contaminen (data poisoning) los patrones reales, cada experto se entrena por separado y se combinan por **OR** (alerta si cualquiera dispara):

```
petición HTTP ─┬─► experto REAL      (CSIC + ECML + CRS)        ─┐
               └─► experto GENERADO  (payloads OWASP propios)   ─┤→ OR → ataque
flujo de red  ───► experto FLUJOS    (CIC-IDS2018, DoS/brute)   ─┘  (a nivel de alerta)
```

Cada experto de texto es un único `sklearn.Pipeline`: TF-IDF de n-gramas de caracteres + características de seguridad (regex + entropía de Shannon) + Random Forest, con probabilidad **calibrada** (`CalibratedClassifierCV`), **post-filtro regex** (allowlist) y umbral por FPR. El artefacto se guarda con `joblib` + verificación `sha256`.

## Reproducir desde cero (un comando)

```bash
git clone https://github.com/SPotes22/OctoMatrix && cd OctoMatrix
bash reproducir.sh                 # texto (experto real + generado)
WITH_FLOWS=1 bash reproducir.sh    # además, experto de flujos (~0,8 GB)
```

`reproducir.sh` crea el entorno, descarga las fuentes (no se versionan), unifica/limpia/particiona, entrena, ensambla el MoE y evalúa. Es idempotente.

## Etapas del pipeline (paso a paso)

```bash
pip install -r requirements.txt
python scripts/download_data.py                         # 1. fuentes reales -> data/raw/
python -m octomatrix.data.build                         # 2. unificar al esquema común
python -m octomatrix.data.clean                         # 3. limpiar + deduplicar
python -m octomatrix.data.splits                        # 4. splits por grupos / escenarios
python -m octomatrix.data.adversarial                   #    set adversarial propio (19 OWASP)
python scripts/train.py --scenario in_source --fpr 0.01 # 5. entrenar experto real
python scripts/build_moe.py --fpr 0.01                  # 6. ensamblar MoE (+ generado)
python scripts/waf_report.py --scenario in_source       # 7. métricas honestas
```

Demo de las etapas de presentación (TRAIN · TEST · POST-TEST):

```bash
python presentation/1_generar_modelo.py
python presentation/2_test_modelo.py
python presentation/3_post_training.py
```

## Uso del modelo

```python
from octomatrix.moe import ScoreUnionEnsemble
from octomatrix.data.text import request_text

moe = ScoreUnionEnsemble.load("models/moe")             # una vez por proceso
texto = request_text("POST", "/login", "", "user=admin&pass=admin' OR '1'='1")
ataque = bool(moe.predict([texto])[0])                  # True = ataque
```

Pasos de integración (middleware Kafka/Pereira, post-entrenamiento) en [`docs/papers/Manual_de_implementacion.pdf`](docs/papers/Manual_de_implementacion.pdf).

## Documentación

| Documento | Contenido |
|---|---|
| [`docs/DATA.md`](docs/DATA.md) | fuentes, esquema común, limpieza y deduplicación |
| [`docs/MODEL.md`](docs/MODEL.md) | arquitectura del detector, calibración, post-filtro, umbral |
| [`docs/METRICS.md`](docs/METRICS.md) | métricas WAF, set adversarial, cobertura OWASP |
| [`docs/MOE.md`](docs/MOE.md) | MoE por procedencia, experto de flujos, feedback de producción |
| [`docs/BENIGN.md`](docs/BENIGN.md) | cómo meter tráfico benigno propio (`--own-logs`) |
| [`docs/papers/`](docs/papers/) | informe de laboratorio, paper IEEE y manual (PDF + fuentes) |
| [`presentation/README.md`](presentation/README.md) | informe v1.1 vs. realidad (correcciones) |

## Limitaciones conocidas

- No generaliza aún a sitios no vistos: necesita tráfico benigno real y variado.
- El set adversarial es **sólo evaluación**: entrenar con él inflaría el resultado por memorización.
- CSIC marca como anómalas peticiones ilegales pero no maliciosas: hay ruido de etiqueta.
- IDOR y escalada de privilegios son poco detectables sólo por el payload.
- Es una **capa heurística**: no reemplaza un WAF de reglas ni validaciones determinísticas.

---

El `security_ml_pipeline.py` del POC original quedó como `ThePipeLine/setup.py` (histórico).
Tests: `python -m pytest -q tests` (45 pruebas).
