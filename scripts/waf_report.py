#!/usr/bin/env python3
"""Reporte de métricas orientadas a WAF para un artefacto OctoMatrix.

    python scripts/waf_report.py --model models/octomatrix.joblib --scenario in_source
    python scripts/waf_report.py --model models/octomatrix.joblib --adversarial

Mide lo que importa en un WAF:
- PR-AUC y ROC-AUC.
- recall al umbral guardado y a FPR objetivo (0,1% y 1%), calculando el umbral
  SOBRE EL MISMO test sólo para esta foto (se avisa: en producción el umbral se
  fija en validación, no en test).
- recall por categoría OWASP y por attack_type.
- recall por fuente.
- latencia por request (p50/p95/p99), medida de a una petición (peor caso).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from octomatrix.data.splits import SCENARIOS, load_scenario  # noqa: E402
from octomatrix.data.text import request_text  # noqa: E402
from octomatrix.owasp import owasp_label  # noqa: E402
from octomatrix.pipeline import OctoMatrixModel  # noqa: E402


def texts(df):
    return [request_text(*r) for r in zip(df.method, df.path, df["query"], df.body, df.headers)]


def recall_at_fpr(y, scores, fpr):
    neg = np.sort(scores[y == 0])
    if len(neg) == 0:
        return None, None
    thr = float(np.nextafter(neg[min(len(neg) - 1, int(np.ceil((1 - fpr) * len(neg))))], 1.0))
    pred = scores >= thr
    return float(pred[y == 1].mean()), thr


def group_recall(df, scores, thr, key):
    out = {}
    for name, g in df[df.label == 1].groupby(key):
        if name is None:
            continue
        idx = g.index.to_numpy()
        out[str(name)] = {"n": int(len(g)), "recall": round(float((scores[idx] >= thr).mean()), 4)}
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["n"]))


def latency_per_request(model, sample_texts, n=300):
    sample = sample_texts[:n] if len(sample_texts) >= n else sample_texts
    model.scores([sample[0]])  # calienta
    times = []
    for t in sample:
        t0 = time.perf_counter()
        model.scores([t])
        times.append((time.perf_counter() - t0) * 1000)
    a = np.array(times)
    return {"p50_ms": round(float(np.percentile(a, 50)), 3),
            "p95_ms": round(float(np.percentile(a, 95)), 3),
            "p99_ms": round(float(np.percentile(a, 99)), 3),
            "n": len(a)}


def build_report(model, test, label):
    test = test.reset_index(drop=True)
    tx = texts(test)
    y = test.label.to_numpy()
    scores = model.scores(tx)
    test = test.assign(_owasp=test.attack_type.map(owasp_label))

    rep = {
        "dataset": label,
        "n": int(len(y)), "attacks": int(y.sum()), "normals": int((y == 0).sum()),
        "roc_auc": round(float(roc_auc_score(y, scores)), 4) if 0 < y.sum() < len(y) else None,
        "pr_auc": round(float(average_precision_score(y, scores)), 4) if y.sum() else None,
        "at_saved_threshold": {},
        "at_target_fpr": {},
        "latency_per_request": latency_per_request(model, tx),
    }
    thr0 = model.threshold.value
    pred0 = scores >= thr0
    rep["at_saved_threshold"] = {
        "threshold": round(thr0, 4),
        "fpr": round(float(pred0[y == 0].mean()), 4) if (y == 0).any() else None,
        "recall": round(float(pred0[y == 1].mean()), 4) if y.sum() else None,
    }
    for fpr in (0.001, 0.01):
        r, thr = recall_at_fpr(y, scores, fpr)
        if r is not None:
            rep["at_target_fpr"][f"{fpr:.1%}"] = {
                "recall": round(r, 4), "threshold_on_test": round(thr, 4),
                "by_owasp": group_recall(test, scores, thr, "_owasp"),
                "by_attack_type": group_recall(test, scores, thr, "attack_type"),
                "by_source": group_recall(test, scores, thr, "source"),
            }
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=Path("models/octomatrix.joblib"))
    ap.add_argument("--scenario", choices=SCENARIOS, default="in_source")
    ap.add_argument("--adversarial", action="store_true", help="evaluar sobre data/processed/adversarial.parquet")
    ap.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    model = OctoMatrixModel.load(args.model)
    if args.adversarial:
        test = pd.read_parquet(args.processed_dir / "adversarial.parquet")
        label = "adversarial"
    else:
        _, _, test = load_scenario(args.scenario, args.processed_dir)
        label = args.scenario

    rep = build_report(model, test, label)
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    print("\nNOTA: 'at_target_fpr' calcula el umbral sobre el propio test (foto "
          "optimista); en producción se usa el umbral de validación ('at_saved_threshold').",
          file=sys.stderr)
    if args.out:
        args.out.write_text(json.dumps(rep, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
