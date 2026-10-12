"""Features de seguridad: regex + entropía de Shannon, como transformer de sklearn.

Entra una lista de textos de petición (ver data.text.request_text) y salen
columnas numéricas de baja latencia que miden la "toxicidad" estructural del
payload. Es `entropía como feature`, no como filtro: el filtro va aparte en
postfilter.py (Fase 3).

El transformer no aprende nada en fit (es determinista), así que es seguro
meterlo en un Pipeline junto al TF-IDF sin riesgo de fuga.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Patrones de ataque (hechos, no texto creativo). Cada uno es una columna: su
# conteo de coincidencias en el texto. El modelo decide cómo pesarlos.
PATTERNS: dict[str, re.Pattern] = {
    "sql_keywords": re.compile(
        r"\b(SELECT|UNION|INSERT|UPDATE|DELETE|DROP|EXEC|WAITFOR|BENCHMARK|SLEEP|xp_cmdshell)\b", re.I
    ),
    "sql_tautology": re.compile(r"('|\")?\s*\b(OR|AND)\b\s*('|\")?\s*\d+\s*=\s*\d+", re.I),
    "sql_comment": re.compile(r"(--|#|/\*|\*/)"),
    "xss_tag": re.compile(r"<\s*(script|img|svg|iframe|body|object|embed)\b", re.I),
    "xss_handler": re.compile(r"\bon\w+\s*=|javascript:", re.I),
    "path_traversal": re.compile(r"\.\./|\.\.\\|%2e%2e|/etc/passwd|win\.ini|boot\.ini", re.I),
    "cmd_injection": re.compile(r"[;&|`]\s*\w|\$\(|\$\{|\bcat\b|\bnc\b|/bin/|cmd\.exe|powershell", re.I),
    "ssi_xxe": re.compile(r"<!--\s*#\s*(exec|include)|<!DOCTYPE|<!ENTITY|SYSTEM\s+[\"']", re.I),
    "ldap": re.compile(r"\(\s*\|\s*\(|objectclass=|\(\s*&\s*\(", re.I),
    "null_byte": re.compile(r"%00|\x00"),
    "special_chars": re.compile(r"[<>;='\"&|%$`]"),
    "url_encoded": re.compile(r"%[0-9a-fA-F]{2}"),
    "double_encoded": re.compile(r"%25[0-9a-fA-F]{2}"),
}

FEATURE_NAMES = (
    ["length", "entropy", "digit_ratio", "non_alnum_ratio", "whitespace_ratio", "max_run"]
    + [f"re_{k}" for k in PATTERNS]
)


def shannon_entropy(text: str) -> float:
    """Entropía de Shannon en bits/carácter. Payloads ofuscados/empaquetados suben."""
    if not text:
        return 0.0
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in Counter(text).values())


def _max_run(text: str) -> int:
    """Longitud de la racha más larga del mismo carácter (detecta buffer overflow: 'AAAA...')."""
    best = run = 0
    prev = None
    for ch in text:
        run = run + 1 if ch == prev else 1
        prev = ch
        best = max(best, run)
    return best


def extract_one(text: str) -> list[float]:
    text = text or ""
    n = len(text)
    inv = 1.0 / n if n else 0.0
    row = [
        float(n),
        shannon_entropy(text),
        sum(ch.isdigit() for ch in text) * inv,
        sum(not ch.isalnum() for ch in text) * inv,
        sum(ch.isspace() for ch in text) * inv,
        float(_max_run(text)),
    ]
    row += [float(len(p.findall(text))) for p in PATTERNS.values()]
    return row


class SecurityFeatures(BaseEstimator, TransformerMixin):
    """Transformer determinista: texto -> matriz de features de seguridad."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray([extract_one(t) for t in X], dtype=np.float32)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(FEATURE_NAMES, dtype=object)
