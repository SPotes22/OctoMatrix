#!/usr/bin/env python3
"""Entrena el detector OctoMatrix y lo guarda como un único artefacto.

    python scripts/train.py --scenario in_source --fpr 0.01 --augment
    python scripts/train.py --scenario csic_ecml_to_crs --out models/octo_cross.joblib

El umbral se fija en validación para el FPR objetivo. El artefacto incluye
pipeline calibrado + post-filtro + umbral + metadata, con su sha256.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from octomatrix.data.splits import SCENARIOS, load_scenario  # noqa: E402
from octomatrix.data.text import request_text  # noqa: E402
from octomatrix.pipeline import OctoMatrixModel  # noqa: E402


def texts(df):
    return [request_text(*r) for r in zip(df.method, df.path, df["query"], df.body, df.headers)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=SCENARIOS, default="in_source")
    ap.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--out", type=Path, default=Path("models/octomatrix.joblib"))
    ap.add_argument("--fpr", type=float, default=0.01, help="FPR objetivo para el umbral")
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--no-calibrate", action="store_true")
    ap.add_argument("--n-estimators", type=int, default=200)
    args = ap.parse_args()

    train, val, test = load_scenario(args.scenario, args.processed_dir, augment=args.augment)
    print(f"escenario {args.scenario}: train={len(train):,} val={len(val):,} test={len(test):,}")

    model = OctoMatrixModel.fit(
        texts(train), train.label.to_numpy(),
        calibrate=not args.no_calibrate, n_estimators=args.n_estimators,
    )
    thr = model.set_threshold_for_fpr(texts(val), val.label.to_numpy(), args.fpr)
    model.metadata["scenario"] = args.scenario
    model.metadata["augmented"] = args.augment
    print(f"umbral={thr.value:.4f}  FPR(val)={thr.measured_fpr:.4f}  recall(val)={thr.measured_recall:.4f}")

    path = model.save(args.out)
    print(f"artefacto: {path}  (+ .sha256, _meta.json)")


if __name__ == "__main__":
    main()
