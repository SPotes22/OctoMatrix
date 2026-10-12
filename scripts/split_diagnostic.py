#!/usr/bin/env python3
"""Diagnóstico de los splits: ¿cuánto infla las métricas un split ingenuo y
cuánto cae el modelo al cambiar de fuente?

Usa un modelo deliberadamente simple (TF-IDF de caracteres + regresión
logística). NO es el modelo de OctoMatrix: sólo sirve para comparar
escenarios entre sí con la misma vara.

    python scripts/split_diagnostic.py [--augment]

El umbral se fija en val para un FPR del 1% y se aplica tal cual en test,
como pasaría en producción.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from octomatrix.data.splits import SCENARIOS, load_scenario  # noqa: E402
from octomatrix.data.text import request_text  # noqa: E402

TARGET_FPR = 0.01


def texts(df: pd.DataFrame) -> list[str]:
    return [request_text(*r) for r in zip(df.method, df.path, df["query"], df.body, df.headers)]


def threshold_at_fpr(y, scores, fpr: float) -> float:
    neg = np.sort(scores[y == 0])
    if len(neg) == 0:
        return 0.5
    return float(neg[min(len(neg) - 1, int(np.ceil((1 - fpr) * len(neg))))])


def evaluate(train, val, test) -> dict:
    vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=100_000, sublinear_tf=True,
                          dtype=np.float32)
    Xtr = vec.fit_transform(texts(train))
    clf = LogisticRegression(max_iter=2000, C=4.0)
    clf.fit(Xtr, train.label)
    sv = clf.predict_proba(vec.transform(texts(val)))[:, 1]
    thr = threshold_at_fpr(val.label.to_numpy(), sv, TARGET_FPR)
    st = clf.predict_proba(vec.transform(texts(test)))[:, 1]
    y = test.label.to_numpy()
    pred = st >= thr
    out = {"n_test": len(test), "attacks": int(y.sum())}
    if 0 < y.sum() < len(y):
        out["roc_auc"] = roc_auc_score(y, st)
        out["pr_auc"] = average_precision_score(y, st)
        out["fpr"] = float(pred[y == 0].mean())
    out["recall"] = float(pred[y == 1].mean()) if y.sum() else np.nan
    return out


def naive_split(processed: Path):
    """El split 'de siempre': aleatorio fila a fila, sobre todas las fuentes."""
    df = pd.read_parquet(processed / "http_requests.parquet")
    tr, rest = train_test_split(df, test_size=0.3, random_state=42, stratify=df.label)
    va, te = train_test_split(rest, test_size=0.5, random_state=42, stratify=rest.label)
    return tr, va, te


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--augment", action="store_true")
    args = ap.parse_args()

    rows = {"naive_random_rows": evaluate(*naive_split(args.processed_dir))}
    for name in SCENARIOS:
        rows[name] = evaluate(*load_scenario(name, args.processed_dir, augment=args.augment))
    table = pd.DataFrame(rows).T
    print(f"umbral fijado en val para FPR={TARGET_FPR:.0%} | augment={args.augment}")
    print(table.to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
