"""Lector del formato 'Start - Id / class / petición / End - Id'.

Es el formato en que el mirror msudol/Web-Application-Attack-Datasets
publica CSIC 2010 y ECML/PKDD 2007:

    Start - Id: 11044
    class: Attack
    GET http://... HTTP/1.1
    Cabecera: valor
    <línea vacía>
    cuerpo | null
    <línea vacía>
    End - Id: 11044
"""

from __future__ import annotations

import re
from typing import IO, Iterator

START_RE = re.compile(r"^Start - Id:\s*(\S+)\s*$", re.IGNORECASE)
END_RE = re.compile(r"^End - Id:\s*(\S+)\s*$", re.IGNORECASE)
CLASS_RE = re.compile(r"^class:\s*(\S+)\s*$", re.IGNORECASE)


def iter_records(fh: IO[str]) -> Iterator[tuple[str, str, str]]:
    """Itera (id, clase, petición_cruda). Ignora registros mal formados."""
    rec_id = cls = None
    buf: list[str] = []
    for raw in fh:
        line = raw.rstrip("\r\n")
        if rec_id is None:
            m = START_RE.match(line)
            if m:
                rec_id, cls, buf = m.group(1), None, []
            continue
        m = END_RE.match(line)
        if m:
            if cls is not None:
                yield rec_id, cls, "\n".join(buf)
            rec_id = None
            continue
        if cls is None:
            m = CLASS_RE.match(line)
            if m:
                cls = m.group(1)
                continue
        buf.append(line)
