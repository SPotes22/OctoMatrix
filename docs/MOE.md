# Fase 5: MoE por procedencia + feedback de producción

Dos piezas:
1. **MoE por procedencia de datos** (`octomatrix/moe.py`): expertos independientes combinados por OR.
2. **POST-TEST / feedback de producción** (`octomatrix/feedback.py`, `scripts/postprod.py`): el lazo que faltaba sobre `{Generate+Collect → Train → Test → Export}`.

## Por qué separar modelos por procedencia

El riesgo, planteado por el proyecto:

| cómo entrenas | qué pasa |
|---|---|
| sólo data de internet | te faltan categorías sin ejemplos públicos (SSRF, XXE, CRLF, privesc…) |
| internet limpia **+ generada en el mismo modelo** | **data poisoning**: las plantillas del generador sesgan los patrones reales |
| sólo generada | **overfitting**: memoriza plantillas, no generaliza |

La salida es un **experto por procedencia**, nunca un modelo mezclado:

```
petición ──┬─► experto REAL      (CSIC+ECML+CRS)      score_r ≥ umbral_r ─┐
           └─► experto GENERADO  (payloads OWASP)     score_g ≥ umbral_g ─┤
                                                                           ▼
                                                              OR  →  ataque si CUALQUIERA alerta
```

- Cada experto es un `OctoMatrixModel` entrenado y calibrado **aparte**, con su **propio umbral**.
- El combinador es **OR por score**: ataque si cualquier experto supera su umbral. El experto generado sólo puede **sumar** detecciones; no reentrena ni toca al experto real. Aislamiento total → no hay poisoning.
- **No hay meta-modelo** entrenado sobre ambos: eso reacoplaría los expertos (lo que se quería evitar).
- Coste: el FPR del experto generado se **suma** al del real. Por eso se mide y se controla (abajo).

### Matiz clave: el experto generado necesita benigno realista

El experto generado se entrena con **ataques generados + tráfico benigno REAL** como negativos. Darle benigno real **no rompe el aislamiento** (el experto real sigue intocado; sólo el generado aprende qué es tráfico normal). Sin ese contexto, su umbral para 1% de FPR se iba a ~1,0: marcaba el tráfico legítimo de producción como ataque y, al subir el umbral para evitarlo, dejaba de disparar del todo. Fue un hallazgo medido, no una suposición.

Los **ataques generados** nunca entran al experto real: ahí sí habría poisoning.

### Armar y evaluar

```bash
python scripts/build_moe.py --fpr 0.01     # entrena generado, ensambla, evalúa -> models/moe/
```

El experto generado se entrena con `payload_split='train'` y se evalúa con `'holdout'` (payloads DISTINTOS de la misma categoría): mide si generaliza a payloads nuevos, no si memoriza.

### Resultados

Experto real reusado (`in_source`), experto generado con 19 categorías, ambos a FPR objetivo 1%.

**Aislamiento — sobre el TEST real (`in_source`):** el experto generado no debe estropear al real.

| | FPR | recall |
|---|---:|---:|
| experto real solo | 1,01% | 89,62% |
| ensemble (OR) | 1,01% | 89,65% |

FPR extra del generado sobre tráfico real: **+0,00%**. Aislamiento confirmado: sumar el experto generado no mueve el FPR ni el recall del real.

**Ganancia — sobre el HOLDOUT generado (payloads NO vistos en entrenamiento):**

| | FPR | recall |
|---|---:|---:|
| experto real solo | 0,55% | 12,63% |
| ensemble (OR) | 0,70% | **100,0%** |

El experto real por sí solo sólo veía el 13% de estos ataques (es el hueco de la Fase 4). El ensemble sube a **100%** de recall sobre payloads **distintos** de los de entrenamiento, con FPR 0,70%. Recall por categoría en holdout: las 19 (sqli, xss, rce, ssrf, xxe, lfi, rfi, crlf, ssti, nosqli, ldap, xpath, command_injection, deserialization, buffer_overflow, open_redirect, path_traversal, privilege_escalation, idor) a **1,0**.

Que el recall sea 1,0 sobre payloads no vistos indica que el experto generado aprendió la **forma** de cada familia de ataque, no plantillas concretas. Es lo esperable: los payloads de una misma categoría comparten estructura. IDOR y privesc también salen 1,0 aquí porque el holdout los presenta como payloads; en tráfico real siguen siendo poco detectables sólo por el texto (ver Fase 4).

**Lectura:** el MoE por procedencia cumple lo que se buscaba — el experto generado **suma** toda la cobertura OWASP que faltaba **sin** contaminar al experto real (poisoning) y sin coste de FPR apreciable. El truco está en los negativos: el experto generado se entrena con benigno sobre las mismas rutas que sus ataques, para separar por payload y no por vocabulario de ruta.

## POST-TEST: feedback de producción (sin reentreno automático)

El lazo nuevo sobre el pipeline del proyecto:

```
producción → prediction_log → muestreo por incertidumbre → review_queue
           → revisión HUMANA → labeled_store → (reentreno MANUAL)
```

```bash
python scripts/postprod.py log   --model models/moe --traffic trafico.jsonl
python scripts/postprod.py queue --band 0.1 --max 300
# una persona rellena 'label' (0/1) y 'reviewer' en data/feedback/review_queue.jsonl
python scripts/postprod.py ingest
```

- **log**: corre el MoE sobre tráfico sin etiqueta y registra decisión, qué expertos dispararon y el **margen** (cercanía a la frontera de decisión).
- **queue** (muestreo por incertidumbre): selecciona las peticiones con `|margen|` más pequeño — donde algún experto quedó justo en su umbral y una etiqueta humana aporta más. Excluye lo ya revisado.
- **ingest**: pasa al `labeled_store` sólo las filas con `label` ∈ {0,1} y `reviewer`; es idempotente.

**El lazo se rompe a propósito en la revisión humana.** No se reentrena solo: ahí es donde reaparecería el data poisoning (alguien envenenando el feedback). El `labeled_store` es la entrada, ya revisada, para un reentreno manual posterior, que alimentaría al experto que corresponda (tráfico real → experto real; ataques nuevos confirmados → su categoría).
