"""Representaciones de texto de una petición.

- request_text: el texto que verá el modelo (sin decodificar).
- skeleton: forma canónica para agrupar casi-duplicados: decodifica %xx/+
  UNA vez, pasa a minúsculas, cambia cada número por 0 y colapsa espacios.
  `id=1` e `id=27` caen en el mismo grupo, y también `%27%20OR` y `' or`.

  Se decodifica una sola vez a propósito: la doble codificación (`%2520`) es
  en sí misma una evasión, y en CSIC es justo lo que distingue un ataque
  (`Confirmar%252B`) de su versión normal (`Confirmar%2B`). Decodificar más
  rondas los mezclaría en el mismo grupo.

Las cabeceras no entran en el skeleton: en ECML son aleatorias y harían
único cada grupo. Que dos peticiones que sólo difieren en cabeceras caigan
en el mismo grupo es seguro (agrupar de más no filtra información).
"""

from __future__ import annotations

import hashlib
import re
from urllib.parse import unquote_plus

_DIGITS = re.compile(r"\d+")
_SPACES = re.compile(r"\s+")


def request_text(method: str, path: str, query: str, body: str, headers: str = "") -> str:
    text = f"{method} {path}"
    if query:
        text += f"?{query}"
    if body:
        text += f"\n\n{body}"
    if headers:
        text += f"\n{headers}"
    return text


def decode(s: str, rounds: int = 3) -> str:
    for _ in range(rounds):
        nxt = unquote_plus(s, encoding="latin-1")
        if nxt == s:
            break
        s = nxt
    return s


def skeleton(method: str, path: str, query: str, body: str) -> str:
    t = decode(f"{method} {path}?{query} {body}", rounds=1).lower()
    t = _DIGITS.sub("0", t)
    return _SPACES.sub(" ", t).strip()


def group_id(method: str, path: str, query: str, body: str) -> str:
    return hashlib.sha1(skeleton(method, path, query, body).encode("utf-8", "surrogatepass")).hexdigest()[:16]
