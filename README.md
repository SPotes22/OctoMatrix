# 🛡️ Security ML Pipeline: Web Attack Anomaly Detector (v1.0.0)

[Python](https://www.python.org/)
[SciKit](https://scikit-learn.org/)
[Github](https://github.com/SPotes22/OctoMatrix)
[kaggle](https://www.kaggle.com/code/santiagopotes/octomatrix-poc-moe-owasp)

> **Estado (rama `development`)**: reescritura sobre datos reales.
> - **Datos**: CSIC 2010, ECML/PKDD 2007 y los tests de regresión del OWASP CRS, unificados, limpiados
>   y deduplicados a ~75.500 peticiones (el POC entrenaba con 72 frases escritas a mano). Ver [docs/DATA.md](docs/DATA.md).
> - **Modelo**: un único `sklearn.Pipeline` (char n-grams + regex/entropía + Random Forest) con
>   probabilidad calibrada, post-filtro regex y umbral por FPR. Ver [docs/MODEL.md](docs/MODEL.md).
> - **Métricas WAF**: PR-AUC, recall a FPR ≤0,1%, por categoría OWASP, por fuente, latencia por request,
>   y un **set adversarial propio** de 19 categorías OWASP (`scripts/waf_report.py`). Ver [docs/METRICS.md](docs/METRICS.md).
> - **MoE por procedencia** (`octomatrix/moe.py`): experto REAL + experto GENERADO combinados por OR, para
>   sumar cobertura OWASP sin que lo generado contamine (poisoning) los patrones reales. Más un
>   **feedback de producción** (POST-TEST, `scripts/postprod.py`) con revisión humana. Ver [docs/MOE.md](docs/MOE.md).
> - **Experto de flujos** (`octomatrix/flows.py`): CIC-IDS2018, detector tabular de ataques de red
>   (DoS/DDoS, fuerza bruta, botnet) que el payload HTTP no puede ver. Tercer experto del MoE.
> - **Diversidad de benignos** (`docs/BENIGN.md`): mete tráfico real de tus apps con `--own-logs` para
>   cerrar el hueco de generalización entre fuentes.

# POC-miniMVP for BlackByte iteration
## 🎯 Visión General: De la Telemetría Cruda a la Inteligencia de Amenazas

Este proyecto implementa un **Pipeline de Machine Learning de Seguridad (SecMLOps)** *end-to-end*. Su objetivo es clasificar el tráfico de entrada (e.g., *queries* HTTP, cargas útiles Kafka) como **Normal** o **Ataque** con alta precisión, sirviendo como una capa heurística avanzada para un **Web Application Firewall (WAF)** o un monitor de tráfico en tiempo real.

Hemos priorizado una metodología **Híbrida de Detección**, combinando la potencia del **NLP (TF-IDF)** para la contextualización de tokens y la **Ingeniería de Características de Seguridad Avanzada** para capturar patrones específicos de *exploits* (el toque de alta ingeniería que lo hace robusto).

-----

## 🏗️ Arquitectura del Pipeline: El Flujo TIN $\rightarrow$ TAN

El *core* del sistema se basa en una secuencia lineal y reproducible, encapsulada en la clase `SecurityMLPipeline`, garantizando que la transición del modelo de **Prototipo (P)** a **Producción Apta (PA)** sea trazable.

| TIN (Tarea Inicial) | TAN (Acción Final) | Componente Clave | Proceso / Output |
| :--- | :--- | :--- | :--- |
| **Data Void** | **Colectar Datos 'Kaggle-Style'** | `collect_kaggle_style_data()` | Generación sintética de **OWASP Top 10** (SQLi, XSS, XXE, etc.) + Patrones **CSIC 2010**. |
| **Dataset Crudo** | **Ingeniería de Características Híbridas** | `TfidfVectorizer` + `extract_advanced_features()` | Combina el vector **TF-IDF** (contexto semántico) con **Features Estadísticos/Heurísticos**. |
| **Feature Matrix (X, y)** | **Entrenamiento y Validación Estratificada** | `train_model()` | Uso de **Random Forest** con `train_test_split(stratify=y)` para asegurar la representación equitativa de las clases minoritarias (*Ataque*). |
| **Modelo Validado** | **Exportación de Artefactos PKL** | `export_model()` | Serializa el **Modelo (`RandomForest`)** y el **Vectorizador (`TfidfVectorizer`)** en un único archivo `.pkl` para un despliegue ligero en producción. |

-----

## ✨ El Toque Kaggle: Ingeniería de Características Avanzada

La alta precisión del modelo se logra no solo por el TF-IDF, sino por un robusto set de **Características de Baja Latencia** que miden la "toxicidad" estructural de la entrada, permitiendo una rápida identificación en el *inferencing* de tráfico masivo (e.g., en un *consumer* Kafka).

### 1\. **Métricas de Complejidad y Entropía**

  * **Shannon Entropy (`_calculate_entropy`):** Mide la aleatoriedad y complejidad del *payload*. *Exploits* como *Buffer Overflows* o *Injection Obfuscation* tienden a elevar este valor.
  * **Longitud y Proporción de Whitespace.**

### 2\. **Heurísticas Específicas de Seguridad (Regex-Driven)**

Se utilizan expresiones regulares para la detección *zero-shot* de patrones críticos, incluso antes de la clasificación ML:

  * **SQL Keywords:** Cuenta de tokens como `SELECT`, `UNION`, `DROP`, `EXEC` (sin importar caso).
  * **XSS Patterns:** Detección de `javascript:`, `<script>`, `on\w+=` (event handlers).
  * **Path Traversal:** Detección de `../`, `..\`, `etc/passwd`, `win.ini`.
  * **Special Characters:** Cuantificación de caracteres típicos de *injection* (`<`, `>`, `;`, `'`, `"`).

### 3\. **TF-IDF (NLP Contextual)**

El **TfidfVectorizer** es entrenado para capturar la importancia de *n-grams* (hasta 3-grams) en el corpus, permitiendo al modelo entender el contexto de las secuencias de *tokens* más allá de las *keywords* directas.

-----

## 🚀 Implementación y Ejecución

### Requisitos Técnicos

```bash
pip install -r requirements.txt
```

### Ejecución del pipeline (Fase 0–3)

```bash
python scripts/download_data.py        # baja las fuentes a data/raw/ (no se versiona)
python -m octomatrix.data.build        # unifica al esquema común
python -m octomatrix.data.clean        # limpia y deduplica -> http_clean.parquet
python -m octomatrix.data.splits       # splits por grupos y escenarios
python scripts/train.py --scenario in_source --fpr 0.01   # entrena el artefacto
python scripts/evaluate.py --scenario in_source           # métricas en test
```

o todo de una: `./run_mechanic.sh in_source`.

> El `security_ml_pipeline.py` del POC original quedó como `ThePipeLine/setup.py` (histórico).
> El detector actual es `octomatrix/pipeline.py`: un único `sklearn.Pipeline`
> (char n-grams + features de seguridad + Random Forest) con probabilidad calibrada,
> post-filtro regex y umbral por FPR. Ver [docs/MODEL.md](docs/MODEL.md).

### 📦 Artefacto generado

`models/octomatrix.joblib`: un solo archivo con el pipeline calibrado, el post-filtro,
el umbral y la metadata. Se acompaña de `octomatrix.joblib.sha256` (integridad) y
`octomatrix_meta.json`. Se carga con `OctoMatrixModel.load(...)`.

-----

## 🧪 Quick Test

`python -m octomatrix.quick_test --model models/octomatrix.joblib` corre unos casos
de humo contra el modelo entrenado y muestra el % de confianza real. Las etiquetas
esperadas son:

| Path | Payload | Esperado |
| :--- | :--- | :--- |
| `/api/users?id=42` | — | ✅ NORMAL |
| `/login` | `admin' OR '1'='1` | 🚨 SQL Injection |
| `/search` | `<script>alert(1)</script>` | 🚨 XSS |
| `/download` | `../../../etc/passwd` | 🚨 Path Traversal |

(La tabla del POC anterior afirmaba estas detecciones pero el modelo de 72 frases
marcaba `/api/users` como ataque al 52%. Ahora `quick_test` imprime el resultado real,
no uno prometido.)
