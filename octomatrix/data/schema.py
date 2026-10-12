"""Esquema común para el experto de payloads HTTP.

Cada fila es una petición HTTP. Las columnas mínimas acordadas son
method, path, query, body, label, attack_type y source; además se guardan:

- headers: cabeceras que pueden llevar payload (User-Agent, Referer, Cookie...)
  serializadas como "Nombre: valor" una por línea. Se descartan cabeceras de
  transporte y los IDs de sesión, que son ruido aleatorio.
- origin_split: el split original de la fuente ("train"/"test"/"-"), para que
  la Fase 2 pueda respetar o rehacer los splits.
- sample_id: identificador estable dentro de la fuente (trazabilidad).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field

import pandas as pd

COLUMNS = [
    "method",
    "path",
    "query",
    "body",
    "headers",
    "label",
    "attack_type",
    "source",
    "origin_split",
    "sample_id",
]

# Campos que identifican una petición (sin etiqueta ni procedencia).
REQUEST_KEY = ["method", "path", "query", "body", "headers"]

NORMAL = "normal"

# Vocabulario cerrado de attack_type. "unspecified" = la fuente sólo dice
# que es anómalo, sin categoría (caso CSIC 2010).
ATTACK_TYPES = {
    NORMAL,
    "unspecified",
    "sqli",
    "xss",
    "path_traversal",
    "rfi",
    "command_injection",
    "php_injection",
    "generic_injection",
    "session_fixation",
    "java_injection",
    "ldap_injection",
    "xpath_injection",
    "ssi_injection",
    "scanner",
    "protocol_attack",
}


@dataclass
class HttpSample:
    method: str
    path: str
    query: str = ""
    body: str = ""
    headers: str = ""
    label: int = 0
    attack_type: str = NORMAL
    source: str = ""
    origin_split: str = "-"
    sample_id: str = ""
    extra: dict = field(default_factory=dict, repr=False)

    def to_row(self) -> dict:
        row = asdict(self)
        row.pop("extra")
        return row


def to_frame(samples) -> pd.DataFrame:
    df = pd.DataFrame([s.to_row() for s in samples], columns=COLUMNS)
    return validate(df)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Comprueba tipos y coherencia label/attack_type. Lanza ValueError."""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"faltan columnas: {missing}")
    df = df[COLUMNS].copy()
    for c in ("method", "path", "query", "body", "headers", "attack_type", "source", "origin_split", "sample_id"):
        df[c] = df[c].fillna("").astype(str)
    df["label"] = df["label"].astype("int8")

    bad_labels = set(df["label"].unique()) - {0, 1}
    if bad_labels:
        raise ValueError(f"label fuera de {{0,1}}: {bad_labels}")
    unknown = set(df["attack_type"].unique()) - ATTACK_TYPES
    if unknown:
        raise ValueError(f"attack_type desconocido: {unknown}")
    incoherent = ((df["label"] == 0) != (df["attack_type"] == NORMAL)).sum()
    if incoherent:
        raise ValueError(f"{incoherent} filas con label/attack_type incoherentes")
    if (df["source"] == "").any():
        raise ValueError("hay filas sin source")
    return df.reset_index(drop=True)
