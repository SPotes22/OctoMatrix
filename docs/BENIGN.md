# Diversidad de tráfico benigno (cerrar el hueco entre fuentes)

## El problema

La Fase 2 y la Fase 4 mostraron el mismo cuello de botella: el modelo generaliza mal a una fuente nueva (`csic_ecml_to_crs`: ROC-AUC 0,53). La causa no es la cantidad de ataques, es la **poca diversidad de tráfico NORMAL**: sólo una tienda JSP (CSIC) y peticiones sintéticas (ECML). El modelo aprende "cómo es lo normal *en esos sitios*" y marca como ataque cualquier tráfico legítimo distinto.

La palanca para cerrarlo es meter **tráfico benigno real y variado**: de tus propias apps.

## Cómo meter tus logs (`--own-logs`)

El loader `octomatrix/data/loaders/access_log.py` entiende el formato *combined* de Apache/Nginx y el de `http.server` de Python:

```
10.0.0.5 - - [12/Oct/2026:09:01:12 +0000] "GET /eventos?categoria=rock HTTP/1.1" 200 4210 "-" "Mozilla/5.0"
127.0.0.1 - - [03/Oct/2025 17:47:05] "GET /health HTTP/1.1" 200 -
```

Se incorpora al dataset así:

```bash
python -m octomatrix.data.build \
    --own-logs /ruta/pereira/access.log:pereira \
    --own-logs /ruta/theunsafeapp/access.log:unsafeapp
python -m octomatrix.data.clean
python -m octomatrix.data.splits
```

Cada log entra con `source=own:<nombre>`, `label=0`. Por defecto se **descartan las respuestas 4xx/5xx** (los sondeos y escáneres suelen acabar en 404/400), así que lo que entra es tráfico que el servidor sirvió de verdad.

Verificado con un log de ejemplo: de 6 líneas, el loader cargó 5 benignas y descartó el sondeo `GET /wp-admin/setup.php` (404).

## Importante (honesto)

- **Revisa los logs antes de entrenar.** Un access log de producción NO está garantizado limpio: puede traer ataques que respondieron 200. El descarte de 4xx/5xx ayuda, pero no sustituye una revisión. El lazo de la Fase 5 (`scripts/postprod.py`) sirve justo para esto.
- **El tráfico propio es tuyo y no se versiona** (`data/` está en `.gitignore`). Sólo tú puedes aportarlo: en este entorno cloud no están tus logs de producción, así que el mecanismo queda listo y verificado, pero el cierre real del hueco ocurre cuando corres lo de arriba con tus logs.
- **A qué experto va.** El tráfico benigno propio alimenta al **experto real** (es tráfico real). Cuanto más variado (varias apps, varios días), mejor generaliza y menos falsos positivos en producción.

## Cómo generar logs si aún no los tienes

- **Apache/Nginx**: ya los escriben (`/var/log/apache2/access.log`, `/var/log/nginx/access.log`).
- **Flask/Gunicorn**: activa el access log (`gunicorn --access-logfile -`).
- **theUnSafeApp / Pereira** (tus apps): corre la app, navega/usa tus endpoints reales un rato y guarda el access log. Ese tráfico, variado y real, es exactamente lo que falta.

Medir el efecto: entrena con y sin tus logs y compara el escenario entre fuentes (`scripts/waf_report.py` / `split_diagnostic.py`). La métrica a vigilar es el **FPR sobre tu propio tráfico** y el recall en `*_to_*`.
