# Datos de OctoMatrix (Fase 1)

## Cómo reproducirlo

```bash
pip install -r requirements.txt
python scripts/download_data.py              # CSIC 2010, ECML/PKDD 2007, CRS (~50 MB)
python -m octomatrix.data.build              # -> data/processed/http_requests.parquet
python -m pytest -q tests

# Experto de flujos (opcional, pesado)
python scripts/download_data.py --skip-http --cicids web   # 2 días con ataques web, ~0,8 GB
python scripts/download_data.py --skip-http --cicids all   # 10 días, ~6,9 GB
python -m octomatrix.data.build --sources "" --cicids

# Tráfico normal propio (access logs Apache/Nginx/http.server)
python -m octomatrix.data.build --own-logs ruta/a/access.log:pereira
```

`data/` no se versiona. `data/raw/MANIFEST.json` guarda el commit o el sha256 exacto de cada fuente.

## Esquema común (experto HTTP)

Definido en `octomatrix/data/schema.py`. Una fila es una petición HTTP.

| columna | qué es |
|---|---|
| `method`, `path`, `query`, `body` | la petición **sin decodificar** (`%27`, `+`... tal cual) |
| `headers` | cabeceras que pueden llevar payload (`Cookie`, `User-Agent`, `Referer`...), una por línea. Se quitan las de transporte y los IDs de sesión |
| `label` | 1 ataque, 0 normal |
| `attack_type` | `normal`, `sqli`, `xss`, `path_traversal`, `command_injection`, ... o `unspecified` |
| `source` | `csic2010`, `ecml_pkdd2007`, `crs_regression`, `crs_regression_negative`, `own:<nombre>` |
| `origin_split` | split original de la fuente (`train` / `test` / `-`) |
| `sample_id` | id dentro de la fuente, para trazabilidad |

`headers` se agregó a las 7 columnas acordadas porque hay ataques que **sólo** van en cabeceras. En ECML, 717 ataques llevan el payload únicamente en `Cookie`, y las pruebas de escáner del CRS están en `User-Agent`.

## Resultado actual

Tras deduplicar quedan **88.814 peticiones únicas** (34.520 ataques y 54.294 normales). El pipeline anterior tenía 72.

| fuente | cargadas | únicas | normales | ataques |
|---|---:|---:|---:|---:|
| csic2010 | 97.065 | 34.593 | 18.640 | 15.953 |
| ecml_pkdd2007 | 50.116 | 50.116 | 35.006 | 15.110 |
| crs_regression | 3.806 | 3.457 | – | 3.457 |
| crs_regression_negative | 794 | 648 | 648 | – |

Por tipo de ataque:

| attack_type | csic | ecml | crs | total |
|---|---:|---:|---:|---:|
| unspecified | 15.953 | | | 15.953 |
| sqli | | 2.274 | 844 | 3.118 |
| command_injection | | 2.302 | 701 | 3.003 |
| path_traversal | | 2.295 | 65 | 2.360 |
| ldap_injection | | 2.279 | | 2.279 |
| xpath_injection | | 2.279 | | 2.279 |
| xss | | 1.825 | 198 | 2.023 |
| ssi_injection | | 1.856 | | 1.856 |
| java_injection | | | 968 | 968 |
| php_injection | | | 303 | 303 |
| generic_injection | | | 237 | 237 |
| protocol_attack | | | 59 | 59 |
| session_fixation | | | 42 | 42 |
| rfi | | | 34 | 34 |
| scanner | | | 6 | 6 |

### Deduplicación

- **Duplicado exacto** (misma petición y misma etiqueta): se conserva la primera aparición.
- **Conflicto** (misma petición con etiquetas distintas): se eliminan todas las copias. Hoy son 120 filas, todas del CRS: positivos y negativos que comparten petición.
- **CSIC 2010 tiene un 64% de duplicados.** De las 36.000 normales de train sólo 9.018 son distintas. El "36.000 + 25.000" de la ficha es en realidad unas 34.600 peticiones únicas.

## Notas por fuente

**CSIC 2010.** Viene del mirror `msudol/Web-Application-Attack-Datasets`, porque el sitio del CSIC está caído.
- Es una sola tienda JSP.
- No trae tipo de ataque, así que todos sus ataques quedan como `unspecified`.
- Lo "anómalo" incluye peticiones ilegales pero no maliciosas (por ejemplo `caracteristicas.jsp?idA=2`), lo que mete ruido de etiqueta.
- El split original deja el train sólo con tráfico normal, así que en la Fase 2 hay que rehacerlo.

**ECML/PKDD 2007.** Mismo mirror.
- Trae 7 tipos de ataque etiquetados.
- Es sintético: nombres de parámetros, hosts y cabeceras aleatorios. Sirve para cubrir variedad de ataques, pero no representa tráfico real, así que hay que evaluarlo por separado.
- El parser acepta URIs con espacios. Sin eso se perdían 1.125 peticiones: 507 ataques (271 de ellos de `command_injection`) y 618 normales.

**OWASP CRS regression tests.** Vienen de `coreruleset/coreruleset`, que reemplaza al repo archivado de SpiderLabs.
- Sólo se usan las familias de ataque a aplicación: 913, 921, 930-934 y 941-944.
- Los tests `no_expect_ids` van como `crs_regression_negative` (label 0). Garantizan que *esa* regla no salta, no que la petición sea inocua, así que se pueden filtrar con `source`.

**Access logs propios** (`--own-logs`). Se asume label 0 y por defecto se descartan las respuestas 4xx/5xx. Aun así, revísalos antes de entrenar: un log de producción puede traer ataques.

## Descartadas

- **HackOWASP (`ravi20076/hackowasp2025lb1`)**: `train.csv` son 10.000 filas con 100 columnas numéricas anónimas (`f1..f100`) y 20 clases. No hay texto HTTP, así que no encaja con ningún experto. Además, el kernel enlazado sólo copia `Thapar.csv` (las respuestas filtradas) como submission.
- **PiBackLogger y los CSV de `owasp_traffic_scanner/02_log_examples`**: son eventos de aplicación (backlog, chat), no peticiones HTTP.

## Experto de flujos: CSE-CIC-IDS2018

Va por separado (`octomatrix/data/loaders/cicids2018.py`) porque **no tiene payload**: cada fila es un flujo con 78 estadísticas de CICFlowMeter.
- Salida: un parquet por día en `data/processed/cicids2018/`, con `label`, `attack_type`, `timestamp` y `day_file`.
- Se quitan las IPs, el puerto de origen y el Flow ID para que el modelo no memorice máquinas.
- También se quitan las cabeceras repetidas dentro del CSV, y los `inf` pasan a NaN.
- Los ataques web (`brute_force_web`, `brute_force_xss`, `sql_injection`) están en los días 22-02 y 23-02.
