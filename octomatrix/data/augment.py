"""Augmentación de evasiones. SÓLO para train (load_scenario se encarga).

Transformaciones (sobre los VALORES de query/body `k=v&k2=v2`, no sobre los
separadores, para que la petición siga siendo parseable por el servidor):

    url_encode     codifica todos los caracteres no alfanuméricos   ' -> %27
    double_encode  doble codificación                               ' -> %2527
    case_swap      mayúsculas/minúsculas aleatorias                 select -> SeLeCt
    sql_comment    espacios -> comentarios inline (sólo SQLi)       union select -> union/**/select
    sql_whitespace espacios -> tab/salto codificados (sólo SQLi)    ' or 1=1 -> '%09or%091=1

Las de codificación se aplican también al tráfico NORMAL. Si sólo se
codificaran ataques, el modelo aprendería el atajo "codificado = ataque" y
bloquearía tráfico legítimo codificado. Excepciones:
- En CSIC la doble codificación ya es una señal de anomalía, así que
  `double_encode` no se aplica a sus normales (fabricaría contradicciones).
- `case_swap` sólo se aplica a ataques: en tráfico real nadie escribe
  `modo=iNSeRTaR`, y en CSIC un valor fuera de formato cuenta como anómalo.
- Las de SQL sólo tienen sentido sobre SQLi.
"""

from __future__ import annotations

import re
from typing import Callable
from urllib.parse import quote

import numpy as np
import pandas as pd

from .text import decode

SQL_HINT = re.compile(r"(?i)(\bunion\b|\bselect\b|\bor\b\s*['\"\d]|\band\b\s*['\"\d]|--|;|\bdrop\b|\bwaitfor\b|\bsleep\s*\()")


def _encode_all(v: str) -> str:
    return quote(decode(v, rounds=1), safe="", encoding="latin-1")


def _double(v: str) -> str:
    return quote(_encode_all(v), safe="")


def _case_swap(v: str, rng: np.random.Generator) -> str:
    flips = rng.random(len(v)) < 0.5
    return "".join(c.swapcase() if f else c for c, f in zip(v, flips))


def _sql_comment(v: str) -> str:
    return re.sub(r"(\s|\+|%20)+", "/**/", v)


def _sql_ws(v: str, rng: np.random.Generator) -> str:
    choices = ["%09", "%0a", "%0d", "%0b", "%a0"]
    return re.sub(r"(\s|\+|%20)+", lambda _: choices[rng.integers(len(choices))], v)


TRANSFORMS: dict[str, Callable] = {
    "url_encode": lambda v, rng: _encode_all(v),
    "double_encode": lambda v, rng: _double(v),
    "case_swap": _case_swap,
    "sql_comment": lambda v, rng: _sql_comment(v),
    "sql_whitespace": _sql_ws,
}
GENERIC = ["url_encode", "double_encode", "case_swap"]
SQL_ONLY = ["sql_comment", "sql_whitespace"]


def _map_values(s: str, fn) -> str:
    """Aplica fn a cada valor de un 'k=v&k2=v2'. Si no tiene esa forma, a todo."""
    if "=" not in s:
        return fn(s)
    out = []
    for pair in s.split("&"):
        k, sep, v = pair.partition("=")
        out.append(f"{k}{sep}{fn(v)}" if sep and v else pair)
    return "&".join(out)


def is_sql_like(row) -> bool:
    if row["attack_type"] == "sqli":
        return True
    if row["attack_type"] == "unspecified":
        return bool(SQL_HINT.search(decode(f"{row['query']} {row['body']}", rounds=2)))
    return False


def applicable(row) -> list[str]:
    if row["label"] == 0:
        return ["url_encode"] if row["source"] == "csic2010" else ["url_encode", "double_encode"]
    names = list(GENERIC)
    if is_sql_like(row):
        names += SQL_ONLY
    return names


def augment_row(row, name: str, rng: np.random.Generator) -> dict | None:
    fn = TRANSFORMS[name]
    new = dict(row)
    changed = False
    for field in ("query", "body"):
        if row[field] and not (field == "body" and "=" not in row[field] and row["label"] == 0):
            new[field] = _map_values(row[field], lambda v: fn(v, rng))
            changed |= new[field] != row[field]
    return new if changed else None


def augment_train(
    train: pd.DataFrame,
    frac: float = 0.5,
    n_variants: int = 1,
    seed: int = 42,
) -> pd.DataFrame:
    """Devuelve train + variantes aumentadas (augmented=True, aug=<transformación>).

    - frac: fracción de filas de train que reciben variantes.
    - n_variants: variantes por fila elegida (transformación aleatoria entre las aplicables).
    Las variantes heredan group_id: si en algún momento se re-particiona, caen
    con su original.
    """
    if "augmented" in train.columns and train["augmented"].any():
        raise ValueError("train ya contiene filas aumentadas")
    rng = np.random.default_rng(seed)
    base = train.copy()
    base["augmented"] = False
    base["aug"] = ""
    chosen = base.sample(frac=frac, random_state=seed) if frac < 1 else base
    seen = set(zip(base["method"], base["path"], base["query"], base["body"], base["headers"]))
    new_rows = []
    for _, row in chosen.iterrows():
        names = applicable(row)
        for k in range(n_variants):
            name = names[rng.integers(len(names))]
            new = augment_row(row, name, rng)
            if new is None:
                continue
            key = (new["method"], new["path"], new["query"], new["body"], new["headers"])
            if key in seen:
                continue
            seen.add(key)
            new.update(augmented=True, aug=name, sample_id=f"{row['sample_id']}#aug{k}:{name}")
            new_rows.append(new)
    if not new_rows:
        return base.reset_index(drop=True)
    return pd.concat([base, pd.DataFrame(new_rows)], ignore_index=True)
