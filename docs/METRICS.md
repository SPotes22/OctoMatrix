# Fase 4: métricas para un WAF + cobertura OWASP

```bash
python -m octomatrix.data.adversarial                       # genera el set adversarial propio
python scripts/waf_report.py --scenario in_source           # métricas sobre test de un escenario
python scripts/waf_report.py --adversarial                  # métricas sobre el set adversarial
```

## Por qué estas métricas

Un WAF no se mide por accuracy. Bloquear tráfico legítimo es caro, así que se fija un **FPR objetivo bajo** y se reporta el recall que se consigue ahí. `waf_report.py` da:

- **PR-AUC** y ROC-AUC (independientes del umbral).
- **recall a FPR 0,1% y 1%** (el umbral se recalcula sobre el test sólo para esta foto; en producción es el de validación).
- **recall por categoría OWASP 2021** y por `attack_type`.
- **recall por fuente** (generalización).
- **latencia por request** (p50/p95/p99), medida de a una petición (peor caso).

## Resultado: modelo `in_source` (entrenado con CSIC+ECML+CRS)

| | ROC-AUC | PR-AUC | recall @FPR 1% | recall @FPR 0,1% |
|---|---:|---:|---:|---:|
| test `in_source` | 0,989 | 0,987 | 89,6% | **59,1%** |

El salto de 1% a 0,1% de FPR cuesta **30 puntos de recall**: para no molestar casi nunca a usuarios legítimos, el modelo deja pasar 4 de cada 10 ataques. Es la disyuntiva real de un WAF y hay que decidirla con el negocio, no esconderla en un accuracy.

### Latencia por request

| | p50 | p95 | p99 |
|---|---:|---:|---:|
| una petición (n_jobs=1) | 39 ms | 55 ms | 78 ms |
| en lote | ~2 ms/req | | |

El modelo fija `n_jobs=1` al cargar: con `n_jobs=-1` el arranque de hilos añadía ~170 ms por petición suelta (205 ms → 39 ms). En un consumidor Kafka que procesa en lote, ~2 ms/req. Los 39 ms sueltos son sobre todo la calibración (3 bosques); para menos latencia se puede bajar `cv` o calibrar con `sigmoid`.

## Resultado: set adversarial propio (19 categorías OWASP)

El set adversarial (`octomatrix/data/adversarial.py`) tiene 7.600 peticiones (3.800 ataques de 19 categorías + 3.800 benignas emparejadas sobre las mismas rutas). Es **nuestro** (payloads = hechos técnicos) y sirve de test honesto.

El modelo `in_source` sobre el adversarial: **ROC-AUC 0,79**, y al umbral de producción atrapa el **12%**. Recall por categoría a FPR 1%:

| categoría OWASP | recall |
|---|---:|
| A05 (XXE) | 0,43 |
| X_memory (buffer overflow) | 0,44 |
| A03 (inyección: sqli, xss, rce, ldap, xpath…) | 0,18 |
| A01 (traversal, lfi, idor, privesc) | 0,15 |
| A08 (deserialización) | 0,07 |
| A10 (SSRF) | 0,02 |

Por tipo, lo que el modelo **no ve en absoluto** (recall ~0): CRLF, open_redirect, RFI, escalada de privilegios. SSRF 2%, NoSQLi 8%, deserialización 7%.

### Qué significa

El modelo aprendió a detectar lo que había en CSIC/ECML/CRS, sobre rutas tipo `/tienda1/...`. Ante el espacio OWASP completo, sobre rutas genéricas (`/api/...`, `/proxy`), **no reconoce la mayoría de los ataques**. Confirma, categoría por categoría, el hueco que faltaba cubrir.

Dos matices honestos:
- **IDOR y escalada de privilegios** son poco detectables sólo por el payload (un `?account_id=1001` es idéntico a uno legítimo). Están en el set para medir ese límite, no para subir el número; necesitan señal de sesión/autorización, no sólo el texto de la petición.
- El set adversarial es **sólo evaluación**. No se entrena con él: si se entrenara, el modelo memorizaría nuestras plantillas y el número saldría inflado sin significar que generaliza.

## Siguiente paso (Fase 5)

Para cerrar el hueco hacen falta **muestras reales y variadas** de estas categorías (más fuentes, o tráfico propio etiquetado), no sólo nuestro generador. El generador sirve para **medir** la cobertura y para pruebas de regresión, no para enseñarle al modelo de memoria.
