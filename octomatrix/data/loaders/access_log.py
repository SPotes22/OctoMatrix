"""Access logs propios (Apache/Nginx 'combined' o 'common', y http.server de Python).

Sirve para meter tráfico normal de tus apps (Pereira, PiChat, theUnSafeApp...).
Un access log no trae body ni la mayoría de cabeceras, así que sólo se llenan
method, path, query y, si vienen, Referer y User-Agent.

Se asume label=0, pero un log de producción NO está garantizado limpio:
por defecto se descartan las respuestas 4xx/5xx (escáneres y sondas suelen
acabar en 404/400) y conviene revisar a mano antes de usarlo para entrenar.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

from ..http_parse import clean_headers, split_uri
from ..schema import NORMAL, HttpSample

LINE_RE = re.compile(
    r'^(?P<ip>\S+) \S+ \S+ \[(?P<ts>[^\]]+)\] '
    r'"(?P<method>[A-Z]+) (?P<uri>\S+)(?: HTTP/[\d.]+)?" '
    r'(?P<status>\d{3}) (?P<size>\S+)'
    r'(?: "(?P<referer>[^"]*)" "(?P<ua>[^"]*)")?'
)


def load(
    path: str | Path,
    source: str,
    drop_error_status: bool = True,
) -> Iterator[HttpSample]:
    path = Path(path)
    files = sorted(path.glob("*.log")) if path.is_dir() else [path]
    if not source.startswith("own:"):
        source = f"own:{source}"
    for f in files:
        with open(f, encoding="utf-8", errors="replace") as fh:
            for lineno, line in enumerate(fh, 1):
                m = LINE_RE.match(line)
                if not m:
                    continue
                if drop_error_status and int(m["status"]) >= 400:
                    continue
                req_path, query = split_uri(m["uri"])
                headers = []
                if m["referer"] not in (None, "", "-"):
                    headers.append(("Referer", m["referer"]))
                if m["ua"] not in (None, "", "-"):
                    headers.append(("User-Agent", m["ua"]))
                yield HttpSample(
                    method=m["method"],
                    path=req_path,
                    query=query,
                    headers=clean_headers(headers),
                    label=0,
                    attack_type=NORMAL,
                    source=source,
                    sample_id=f"{f.name}:{lineno}",
                )
