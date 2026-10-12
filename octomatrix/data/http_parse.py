"""Parseo de peticiones HTTP crudas al esquema común.

No se decodifica nada (%xx, +, unicode): el texto se guarda tal cual llegó.
Decodificar es una decisión de features que corresponde a la fase de modelo.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit

# Cabeceras de transporte/negociación: no llevan payload útil y en datasets
# sintéticos (ECML) son ruido aleatorio que el modelo memorizaría.
DROP_HEADERS = {
    "host",
    "connection",
    "content-length",
    "content-type",
    "accept",
    "accept-charset",
    "accept-encoding",
    "accept-language",
    "cache-control",
    "pragma",
    "date",
    "keep-alive",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "via",
    "max-forwards",
    "expect",
    "etag",
    "if-match",
    "if-none-match",
    "if-range",
    "if-modified-since",
    "if-unmodified-since",
    "range",
    "mime-version",
    "warning",
}

# Cookies de sesión: identificadores aleatorios, cero señal.
SESSION_COOKIE_RE = re.compile(
    r"(?i)\b(JSESSIONID|PHPSESSID|ASP\.NET_SessionId|ASPSESSIONID\w*|sid|sessionid)=[^;]*;?\s*"
)

# La URI puede traer espacios sin codificar (muchos ataques de ECML los llevan:
# `GET /scripts/file.bat" & dir c:/ .exe? HTTP/1.0`), así que se toma todo lo
# que hay entre el método y la versión final, no sólo el primer token.
REQUEST_LINE_RE = re.compile(r"^([A-Z]+)\s+(.+?)(?:\s+(HTTP/\d(?:\.\d)?))?\s*$")


def split_uri(uri: str) -> tuple[str, str]:
    """Devuelve (path, query) quitando esquema y host si la URI es absoluta."""
    if "://" in uri.split("?", 1)[0]:
        parts = urlsplit(uri)
        path = parts.path or "/"
        return path, parts.query
    path, _, query = uri.partition("?")
    return path or "/", query


def clean_headers(pairs) -> str:
    """Filtra cabeceras de ruido y serializa 'Nombre: valor' por línea."""
    out = []
    for name, value in pairs:
        name = name.strip()
        if not name or name.lower() in DROP_HEADERS:
            continue
        # Separadores falsos que trae ECML ("----: -----", "~~~~~: ~~~~").
        if set(name) <= {"-", "~"}:
            continue
        value = value.strip()
        if name.lower() == "cookie":
            value = SESSION_COOKIE_RE.sub("", value).strip().rstrip(";").strip()
        if value:
            out.append(f"{name}: {value}")
    return "\n".join(out)


def parse_raw_request(text: str) -> dict | None:
    """Parsea una petición HTTP en texto. Devuelve None si no es parseable."""
    lines = text.replace("\r\n", "\n").split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        return None
    m = REQUEST_LINE_RE.match(lines[0].strip())
    if not m:
        return None
    method, uri = m.group(1), m.group(2)

    header_pairs = []
    i = 1
    while i < len(lines) and lines[i].strip():
        name, sep, value = lines[i].partition(":")
        if sep:
            header_pairs.append((name, value))
        i += 1
    body = "\n".join(lines[i + 1 :]).strip()
    if body == "null":  # ECML marca así el cuerpo vacío
        body = ""

    path, query = split_uri(uri)
    return {
        "method": method,
        "path": path,
        "query": query,
        "body": body,
        "headers": clean_headers(header_pairs),
    }
