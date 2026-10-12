# El detector OctoMatrix (Fase 0 + Fase 3)

`octomatrix/pipeline.py`. Un único artefacto con todo el pre-procesado dentro.

## Arquitectura

```
petición HTTP
   │  request_text(method, path, query, body, headers)
   ▼
FeatureUnion
   ├─ TfidfVectorizer(char_wb, n-grams 1–3)      contexto de tokens
   └─ SecurityFeatures (regex + entropía Shannon)  toxicidad estructural
   ▼
RandomForestClassifier                            (entropía va como feature, no como filtro)
   ▼
CalibratedClassifierCV (isotonic)                 probabilidad calibrada
   ▼
RegexPostFilter (allowlist)                       rebaja falsos positivos; nunca sube el score
   ▼
% de confianza  →  umbral elegido para un FPR objetivo
```

Es exactamente el flujo que describía el README (entrada → RF con entropía → probabilidad calibrada → regex como post-filtro → % de confianza → umbral por FPR), pero implementado de verdad y en un solo `sklearn.Pipeline`.

## Por qué un solo artefacto

El POC guardaba el modelo y el `TfidfVectorizer` por separado y reconstruía las 8 features a mano en cada script. Eso causaba el error real `X has 407 features, but RandomForest is expecting 415`: las features se desajustaban entre entrenamiento e inferencia. Ahora el `Pipeline` hace el pre-proceso siempre igual, así que ese desajuste es imposible.

El artefacto se guarda con `joblib` + `sha256` (y un `_meta.json`), no con `pickle` suelto; al cargar se verifica el hash.

## Features de seguridad (`octomatrix/features.py`)

6 numéricas + 13 conteos de regex:
- `length`, `entropy` (Shannon, bits/char), `digit_ratio`, `non_alnum_ratio`, `whitespace_ratio`, `max_run` (racha máxima de un carácter, detecta buffer overflow `AAAA…`).
- conteos de: SQL keywords, tautologías SQL, comentarios SQL, tags XSS, handlers XSS, path traversal, command injection, SSI/XXE, LDAP, null byte, caracteres especiales, URL-encoding y doble URL-encoding.

## Calibración (Fase 3)

`CalibratedClassifierCV(method="isotonic", cv=3)` envuelve el pipeline completo. Un Random Forest da probabilidades mal calibradas (sesgadas al centro); la calibración hace que "0,9" signifique de verdad "90% de las veces es ataque". Es lo que da sentido al **% de confianza** y a elegir un umbral por FPR.

## Post-filtro regex (Fase 3)

`octomatrix/postfilter.py`. Una allowlist explícita (recursos estáticos, `/health`, `/metrics`, `favicon.ico`...). Si una petición cae en la allowlist **y** no dispara ninguna firma fuerte, su score se rebaja a 0. Solo rescata falsos positivos; nunca convierte un normal en ataque. Es configurable por proyecto.

## Umbral por FPR

`set_threshold_for_fpr(val, target_fpr)` fija el umbral en **validación** para no superar el FPR objetivo (por defecto 1%), y se aplica tal cual en test. En un WAF el coste de bloquear tráfico legítimo es alto, así que se controla el FPR y se reporta el recall que se consigue a ese FPR, no un accuracy global.

## Entrenar y evaluar

```bash
python scripts/train.py    --scenario in_source --fpr 0.01 [--augment]
python scripts/evaluate.py --scenario in_source
python matrix_mechanic.py  --scenario in_source      # matriz de confusión + errores
```

## Coste y límites

El RF trabaja sobre char n-grams densos, así que el espacio de features se acota a propósito (`max_features=3000`, `max_depth=40`) para que entrene en una máquina normal. Subirlos mejora algo el recall pero dispara memoria y tiempo. Para el experto de flujos (CIC-IDS2018) el modelo será otro, porque esos datos no son texto.

## Resultados

Se añaden aquí tras correr `scripts/train.py` + `scripts/evaluate.py` en cada escenario. El diagnóstico de la Fase 2 (modelo lineal simple) ya mostró la conclusión que se espera confirmar: bien dentro de una fuente, mal entre fuentes, por falta de diversidad de tráfico normal.
