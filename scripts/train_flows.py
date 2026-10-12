#!/usr/bin/env python3
"""Entrena y evalúa el experto de FLUJOS (CIC-IDS2018).

    python scripts/train_flows.py --out models/flows.joblib

Lee los parquet de data/processed/cicids2018/ (generados por
`python -m octomatrix.data.build --sources "" --cicids`).

Split temporal POR DÍA: no se parte un flujo a la mitad ni se mezclan días en
train y test al azar. Se entrena con una parte temprana de cada día y se evalúa
con la parte tardía (split por timestamp), que es lo más parecido a "entrenar
con el pasado y detectar el futuro".

Reporta PR-AUC, recall a FPR 0,1% y 1%, recall por tipo de ataque y latencia.
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
from octomatrix.flows import FlowModel  # noqa: E402


def load_flows(processed_dir: Path, max_rows: int | None) -> pd.DataFrame:
    files = sorted((processed_dir / "cicids2018").glob("*.parquet"))
    if not files:
        sys.exit(f"no hay parquet en {processed_dir/'cicids2018'}; corre build --cicids primero")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=42).reset_index(drop=True)
    return df


def temporal_split(df: pd.DataFrame, frac_train=0.6, frac_val=0.15):
    """Por día: ordena por timestamp y corta train/val/test en el tiempo."""
    parts = {"train": [], "val": [], "test": []}
    for _, day in df.groupby("day_file"):
        day = day.sort_values("timestamp", kind="stable")
        n = len(day)
        i, j = int(n * frac_train), int(n * (frac_train + frac_val))
        parts["train"].append(day.iloc[:i])
        parts["val"].append(day.iloc[i:j])
        parts["test"].append(day.iloc[j:])
    return (pd.concat(parts["train"]), pd.concat(parts["val"]), pd.concat(parts["test"]))


def recall_at_fpr(y, scores, fpr):
    neg = np.sort(scores[y == 0])
    if not len(neg):
        return None
    thr = float(np.nextafter(neg[min(len(neg) - 1, int(np.ceil((1 - fpr) * len(neg))))], 1.0))
    return float((scores[y == 1] >= thr).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--out", type=Path, default=Path("models/flows.joblib"))
    ap.add_argument("--fpr", type=float, default=0.01)
    ap.add_argument("--max-rows", type=int, default=400_000, help="subsample total (POC)")
    args = ap.parse_args()

    df = load_flows(args.processed_dir, args.max_rows)
    train, val, test = temporal_split(df)
    print(f"flujos: train={len(train):,} val={len(val):,} test={len(test):,} "
          f"(ataques test={int(test.label.sum()):,})")
    print(f"tipos de ataque: {sorted(set(df[df.label==1].attack_type))}")

    model = FlowModel.fit(train)
    thr = model.set_threshold_for_fpr(val, args.fpr)
    print(f"umbral={thr.value:.4f} FPR(val)={thr.measured_fpr:.4f} recall(val)={thr.measured_recall:.4f}")
    model.metadata["days"] = sorted(set(df.day_file))
    model.save(args.out)

    scores = model.scores(test)
    y = test.label.to_numpy()
    t0 = time.perf_counter(); model.scores(test.iloc[:2000]); lat = (time.perf_counter() - t0) / 2 * 1000
    report = {
        "n_test": int(len(y)), "attacks": int(y.sum()),
        "roc_auc": round(float(roc_auc_score(y, scores)), 4),
        "pr_auc": round(float(average_precision_score(y, scores)), 4),
        "recall_at_fpr": {f"{f:.1%}": round(recall_at_fpr(y, scores, f), 4) for f in (0.001, 0.01)},
        "latency_ms_per_flow_batched": round(lat, 4),
        "recall_by_attack_type": {},
    }
    for at, g in test[test.label == 1].groupby("attack_type"):
        report["recall_by_attack_type"][at] = {
            "n": int(len(g)), "recall": round(float((scores[g.index] >= thr.value).mean()), 4)}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    (args.processed_dir.parent / "output").mkdir(exist_ok=True)
    Path("output/flows_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
