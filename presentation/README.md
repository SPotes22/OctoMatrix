# OctoMatrix — presentación limpia

Esta carpeta presenta el proyecto por sus etapas reales, con **números honestos**.
Reemplaza el discurso inflado del informe v1.1 por lo que el código hace de verdad.

## Las tres etapas

```bash
python presentation/1_generar_modelo.py   # TRAIN + EXPORT: genera los artefactos
python presentation/2_test_modelo.py      # TEST: evaluación honesta, métricas reales
python presentation/3_post_training.py    # POST-TEST: lazo de feedback de producción
```

Son envoltorios finos sobre los módulos probados (`octomatrix/…`, `scripts/…`); no reimplementan lógica, así que no se desincronizan del código con tests.

## Informe v1.1 vs. realidad

El informe de laboratorio describía un sistema distinto del que el código implementaba. Tabla de correcciones:

| El informe decía | La realidad del código | Ahora |
|---|---|---|
| "Se entrena con datasets estilo Kaggle + datos sintéticos" | **72 frases escritas a mano** en `collect_kaggle_style_data()`; las 4 fuentes citadas (CSIC, CRS, HackOWASP, IDS2018) **nunca se cargaban** | **75.458 peticiones reales** de CSIC 2010 + ECML/PKDD 2007 + OWASP CRS, unificadas y deduplicadas |
| "Búsqueda segura del 83%" / "precisión >82%" | `accuracy` de un `train_test_split` sobre 72 filas (**test = 15 muestras**); además `export_model` guardaba el **método** `self.model.score`, no el número | Evaluación real: **recall 89,6% @FPR 1%**, y **59,1% @FPR 0,1%** sobre test separado. Se reportan ambos, no un número suelto |
| "Protección de tráfico en tiempo real" probada | `quick_test` afirmaba detecciones; el modelo marcaba `/api/users` como ataque al 52% | `presentation/2_test_modelo.py` imprime las métricas reales; hay fuga de etiqueta medida y documentada |
| "Regex para interceptar ataques con alta prioridad" | el regex **sólo generaba features**, no filtraba nada | Regex **sí** filtra falsos positivos ahora (`postfilter.py`), además de ser features |
| "% de confianza" | `predict_proba` de un RF **sin calibrar** | Probabilidad **calibrada** (`CalibratedClassifierCV`) + umbral por FPR |
| Nombre "POC-MoE-OWASP" | **no existía** ningún MoE en el código | MoE real por procedencia: experto real + generado + flujos (`moe.py`, `flows.py`) |
| "Entrenado con OWASP Top 10" | 4–5 categorías | **19 categorías** OWASP en el set adversarial propio |
| Serialización con `pickle` | cargar un `.pkl` ajeno ejecuta código | `joblib` + verificación `sha256` antes de deserializar |
| "Impacto a escala planetaria", "Safe by Design is the foundation" | lemas sin respaldo técnico (frases que se colaron de notas) | fuera |
| Gemini "Pay to Ask", tokens blockchain, "Event Crawler ético 60-70%" | **no son de OctoMatrix**: pertenecen a Parchate Pereira (el Event Crawler es un Selenium que guarda eventos de FB/IG) | fuera del alcance de OctoMatrix; es un detector de anomalías OWASP, nada más |

## Números reales (resumen)

| escenario | qué mide | resultado |
|---|---|---|
| in_source | dentro de las fuentes vistas | ROC 0,989 · recall 89,6% @FPR 1% |
| cross-source | generalización a fuente NUEVA | **ROC 0,53** (no generaliza: falta diversidad de benignos) |
| adversarial (19 OWASP) | cobertura con el MoE | recall ~100% con el experto generado |
| flujos (CIC-IDS2018) | ataques de red (DoS, fuerza bruta) | ROC 1,0 — pero el dataset es trivialmente separable |

Detalle y matices en `docs/DATA.md`, `docs/MODEL.md`, `docs/METRICS.md`, `docs/MOE.md`, `docs/BENIGN.md`.

## Lo que el manual de implementación sí acertó

El `Manual de implementación` (OctoMatrix como capa de validación, no reemplaza seguridad, limitaciones honestas, no detecta ataques nuevos) está bien planteado y sigue siendo válido. La diferencia es que ahora el modelo que carga ese middleware es real y medido, no 72 frases.
