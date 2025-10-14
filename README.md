# 🛡️ Security ML Pipeline: Web Attack Anomaly Detector (v1.0.0)

[Python](https://www.python.org/)
[SciKit](https://scikit-learn.org/)
[Github](https://github.com/SPotes22/OctoMatrix)
[kaggle](https://www.kaggle.com/code/santiagopotes/octomatrix-poc-moe-owasp)

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

Asegúrate de tener las librerías necesarias instaladas:

```bash
pip install pandas numpy scikit-learn
```

### Ejecución del Pipeline

El archivo `security_ml_pipeline.py` está diseñado para ejecutarse directamente, iniciando todo el ciclo de vida del ML y **generando los artefactos** en la carpeta `output/`.

```bash
python security_ml_pipeline.py
```

### 📦 Artefactos Generados

Tras la ejecución exitosa, la carpeta `output/` contendrá los siguientes archivos, listos para ser cargados en un entorno de producción (e.g. security_model.pkl ):

1.  `output/security_model.pkl`: Contiene el objeto completo del modelo (`RandomForest`) y el **Vectorizador** pre-entrenado.
2.  `output/training_dataset.csv`: El dataset sintético completo utilizado para el entrenamiento, crucial para la **trazabilidad de QA/auditoría**.

-----

## 🧪 Quick Test: Demostración de Detección

El método `quick_test()` valida la capacidad del modelo para distinguir entre tráfico legítimo y los principales vectores de ataque con alta confianza.
```
| Path | Payload | Detección Esperada |
| :--- | :--- | :--- |
| `/api/users` | `normal query` | **✅ NORMAL** |
| `/login` | `admin' OR '1'='1` | **🚨 ATAQUE** (SQL Injection) |
| `/search` | `<script>alert(1)</script>` | **🚨 ATAQUE** (Cross-Site Scripting) |
| `/download` | `../../../etc/passwd` | **🚨 ATAQUE** (Path Traversal) |
```
Simulación de tráficos
```
/login/auth, /transfer/internal, /api/balance/check, /support/ticket
```

 se marcan como ✅ legítimos.

```
/etc/passwd, /api/admin/export?table=credit_cards
```

se levantan como 🚨 sospechosos.

→ Eso ya es detección de intrusiones en vivo.

----
# OCTOMATRIX the spider of the web2 - THE SIMPLIEST IMPLEMENTATION SO FAR-

## 🔥 OctoMatrix Modular Warrior - POC MVP

**Sistema de seguridad bancaria con detección de amenazas en tiempo real usando Kafka + ML**

## 🚀 ¿Qué hace este proyecto?

Simula un banco legacy inseguro y monitorea el tráfico en busca de amenazas usando:
- **Kafka** para streaming de datos
- **ML Models** para detección de anomalías  
- **Dashboard Flask** en tiempo real
- **Arquitectura microservicios**

## 🏗️ Arquitectura

Frontend (HTML) → Kafka Producer → Kafka → Consumer → ML Analysis → Dashboard

text

## 📦 Instalación Rápida

```bash
git clone [tu-repo]
cd OctoMatrix_modular_warrior

# Instalar dependencias
./install_deps.sh

# Iniciar Kafka
docker-compose -f kafka/docker-compose.kafka.yml up -d

# Ejecutar sistema completo
cd theUnSafeApp && ./start_fixed_system_v2.sh
```
# 🎯 Características Principales

# 🔍 Detección de Amenazas tipo OWASP top 10 2021

* SQL Injection

* Path Traversal

* Credential Stuffing

* XSS Patterns

📊 Dashboard en Tiempo Real
Métricas live de seguridad

Gráficos interactivos

API REST para integración

🔧 Tech Stack

* Python 3.11 + Flask

* Apache Kafka + Docker

* Machine Learning (scikit-learn) -> [ random forest + shannon entropy + regex ]

Chart.js para visualización

🎮 Uso Rápido
bash
# Terminal 1 - Backend Kafka
cd theUnSafeApp && python mook_kafka_producer_fixed.py

# Terminal 2 - Dashboard  
cd . && python dashboard_integrator.py

# Acceder: http://localhost:5000/dashboard
📁 Estructura del Proyecto
```text
OctoMatrix_modular_warrior/
├── theUnSafeApp/          # Backend & Kafka components
├── templates/             # Dashboard frontend
├── kafka/                # Docker Kafka setup
├── ThePipeLine/          # ML pipeline & utilities
└── README.md
```
🛡️ Seguridad
Detección automática de patrones OWASP

Análisis de confianza en tiempo real

Dashboard de monitoreo continuo

🤝 Contribución
Este es un POC/MVP educativo. ¡PRs son bienvenidos!

📄 Licencia
MIT GPLv3 - ¡Usa, modifica, comparte!

# Archivos CORE para el público
---
```bash
git add theUnSafeApp/mook_kafka_producer_fixed.py -> refactor kafka consumer
git add theUnSafeApp/kafka_consumer_fixed.py   -> hot_fix kafka logs 
git add theUnSafeApp/mook_analyzer.py -> send random logs
git add theUnSafeApp/mook_html/ -> static front 
git add dashboard_integrator.py -> security dashboard ( chartjs + flask )
git add templates/ -> xd
git add kafka/docker-compose.kafka.yml -> manifest
git add ThePipeLine/ -> Update your model pipeline 
git add install_deps.sh -> if u have pip problems
```
---
# RESUMEN

- Backend Kafkaa para tráfico bancario simulado
- Dashboard Flask con métricas en tiempo real  
- Detección ML de amenazas (SQLi, XSS, Path Traversal)
- Arquitectura microservicios escalable
- Documentación completa y ejemplos de uso"

----
# Secret Details (not so secret)

¿QUÉ VA AL REPO PÚBLICO? ✅
```
-> theUnSafeApp/ (solo los archivos core, no logs)

-> templates/ (dashboard frontend)

-> kafka/docker-compose.kafka.yml (setup Kafka)

-> ThePipeLine/ (ML utilities)

-> dashboard_integrator.py (servidor Flask)

-> README.md + .gitignore
```
¿QUÉ SE QUEDA LOCAL? ❌
```
-> logs/, *.log (archivos de log)

-> secrets/ (configuraciones sensibles)

-> quick_fix*, restart_system* (scripts de desarrollo)

-> Archivos temporales y de debug
```
---
