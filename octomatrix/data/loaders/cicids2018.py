"""CSE-CIC-IDS2018: flujos de red (CICFlowMeter) para el experto de flujos.

OJO: aquí NO hay payload HTTP. Cada fila es un flujo TCP/UDP descrito por
~80 estadísticas (duración, paquetes, bytes, flags, tiempos entre llegadas).
Por eso no entra al esquema HttpSample: tiene su propio esquema y su propio
modelo (el "experto de flujos" del MoE).

Fuente: bucket público s3://cse-cic-ids2018, carpeta
"Processed Traffic Data for ML Algorithms/" (10 CSV, ~6,9 GB en total).

Limpieza que hace este loader:
- Quita las cabeceras repetidas que vienen a mitad de algunos CSV.
- Quita columnas identificadoras (Flow ID, IPs, puerto origen): el modelo las
  memorizaría y no generalizan. Sólo el CSV del 20-02 las trae.
- inf/-inf → NaN; las features se guardan como float32.
- Normaliza las etiquetas a snake_case y añade label binaria.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

SOURCE = "cicids2018"
S3_BASE = "https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com"
S3_PREFIX = "Processed Traffic Data for ML Algorithms/"

DAY_FILES = [
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv",  # sic, así se llama en S3
    "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-23-02-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",
]

# Días que contienen los ataques web (Brute Force -Web, Brute Force -XSS, SQL Injection).
WEB_ATTACK_DAYS = DAY_FILES[5:7]

ID_COLUMNS = ["Flow ID", "Src IP", "Src Port", "Dst IP"]
META_COLUMNS = ["Timestamp", "Label"]


def normalize_label(raw: str) -> str:
    """'DoS attacks-Hulk' -> 'dos_attacks_hulk', 'Benign' -> 'normal'."""
    s = raw.strip().lower()
    if s == "benign":
        return "normal"
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def _clean_chunk(chunk: pd.DataFrame, day: str) -> pd.DataFrame:
    chunk = chunk[chunk["Label"] != "Label"]  # cabeceras repetidas
    chunk = chunk.drop(columns=[c for c in ID_COLUMNS if c in chunk.columns])
    features = [c for c in chunk.columns if c not in META_COLUMNS]
    X = chunk[features].apply(pd.to_numeric, errors="coerce").astype("float32")
    X = X.replace([np.inf, -np.inf], np.nan)
    attack_type = chunk["Label"].map(normalize_label)
    out = X
    out["timestamp"] = pd.to_datetime(chunk["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    out["attack_type"] = attack_type.values
    out["label"] = (attack_type != "normal").astype("int8").values
    out["source"] = SOURCE
    out["day_file"] = day
    return out


def load_day(path: str | Path, chunksize: int = 200_000, max_rows: int | None = None) -> Iterator[pd.DataFrame]:
    """Itera chunks limpios de un CSV diario."""
    path = Path(path)
    read = 0
    for chunk in pd.read_csv(path, chunksize=chunksize, dtype=str, low_memory=False):
        if max_rows is not None:
            if read >= max_rows:
                return
            chunk = chunk.iloc[: max_rows - read]
        read += len(chunk)
        yield _clean_chunk(chunk, path.name)


def build(raw_dir: str | Path, out_dir: str | Path, days=None, max_rows_per_day: int | None = None) -> dict:
    """Convierte los CSV diarios presentes a parquet (uno por día). Devuelve conteos."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name in days or DAY_FILES:
        src = raw_dir / name
        if not src.exists():
            continue
        parts = list(load_day(src, max_rows=max_rows_per_day))
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True)
        df.to_parquet(out_dir / (Path(name).stem + ".parquet"), index=False)
        stats[name] = {
            "rows": int(len(df)),
            "attack_types": {k: int(v) for k, v in df["attack_type"].value_counts().items()},
        }
    return stats
