#!/usr/bin/env python3
"""Evalúa un artefacto OctoMatrix en el test de un escenario.

    python scripts/evaluate.py --model models/octomatrix.joblib --scenario in_source

Reporta métricas útiles para un WAF: ROC-AUC, PR-AUC, FPR y recall al umbral
guardado, recall por tipo de ataque y efecto del post-filtro.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from octomatrix.data.splits import SCENARIOS, load_scenario  # noqa: E402
from octomatrix.data.text import request_text  # noqa: E402
from octomatrix.pipeline import OctoMatrixModel  # noqa: E402


def texts(df):
    return [request_text(*r) for r in zip(df.method, df.path, df["query"], df.body, df.headers)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=Path("models/octomatrix.joblib"))
    ap.add_argument("--scenario", choices=SCENARIOS, default="in_source")
    ap.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--out", type=Path, default=None, help="guardar métricas JSON")
    args = ap.parse_args()

    model = OctoMatrixModel.load(args.model)
    _, _, test = load_scenario(args.scenario, args.processed_dir)
    tx = texts(test)
    y = test.label.to_numpy()
    scores = model.scores(tx)
    pred = (scores >= model.threshold.value).astype(int)

    report = {
        "scenario": args.scenario,
        "threshold": model.threshold.value,
        "n_test": int(len(y)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "pr_auc": float(average_precision_score(y, scores)),
        "fpr": float(pred[y == 0].mean()),
        "recall": float(pred[y == 1].mean()),
        "recall_by_attack_type": {},
    }
    for at, g in test[test.label == 1].groupby("attack_type"):
        idx = g.index.to_numpy()
        report["recall_by_attack_type"][at] = {
            "n": int(len(g)), "recall": float((scores[idx] >= model.threshold.value).mean())
        }

    # efecto del post-filtro: cuántos positivos del modelo rescata en benignos
    raw = model.pipeline.predict_proba(tx)[:, 1]
    rescued = ((raw >= model.threshold.value) & (scores < model.threshold.value) & (y == 0)).sum()
    report["postfilter_rescued_fp"] = int(rescued)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
