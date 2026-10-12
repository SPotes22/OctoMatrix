#!/usr/bin/env python3
"""Arma el MoE por procedencia: experto real + experto generado (OR por score).

    python scripts/build_moe.py --fpr 0.01

Pasos:
1. Experto REAL: reusa models/octomatrix.joblib si existe (entrenado con el
   escenario in_source), o lo entrena.
2. Experto GENERADO: entrena con payloads generados de TRAIN y fija su umbral
   contra tráfico real benigno (para que su FPR sobre lo real quede acotado).
3. Ensambla (ScoreUnionEnsemble) y guarda en models/moe/.
4. Evalúa el aislamiento y la ganancia:
   - sobre el TEST real (in_source): el FPR no debe dispararse, el recall real
     no debe caer.
   - sobre el HOLDOUT generado (payloads NO vistos): recall del ensemble vs
     recall del experto real solo.

El experto generado se entrena con payload_split='train' y se evalúa con
'holdout' (payloads distintos de la misma categoría): mide generalización, no
memoria.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from octomatrix.data.adversarial import AdversarialGenerator  # noqa: E402
from octomatrix.data.splits import load_scenario  # noqa: E402
from octomatrix.data.text import request_text  # noqa: E402
from octomatrix.moe import Expert, ScoreUnionEnsemble  # noqa: E402
from octomatrix.pipeline import OctoMatrixModel  # noqa: E402


def texts(df):
    return [request_text(*r) for r in zip(df.method, df.path, df["query"], df.body, df.headers)]


def fpr_recall(fired: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    fpr = float(fired[y == 0].mean()) if (y == 0).any() else float("nan")
    rec = float(fired[y == 1].mean()) if (y == 1).any() else float("nan")
    return fpr, rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--real-model", type=Path, default=Path("models/octomatrix.joblib"))
    ap.add_argument("--out", type=Path, default=Path("models/moe"))
    ap.add_argument("--fpr", type=float, default=0.01, help="FPR objetivo por experto")
    ap.add_argument("--per-category", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Datos reales
    real_train, real_val, real_test = load_scenario("in_source", args.processed_dir)

    # 1. Experto real
    if args.real_model.exists():
        print(f"experto real: reuso {args.real_model}")
        real = OctoMatrixModel.load(args.real_model)
    else:
        print("experto real: entrenando in_source...")
        real = OctoMatrixModel.fit(texts(real_train), real_train.label.to_numpy())
        real.set_threshold_for_fpr(texts(real_val), real_val.label.to_numpy(), args.fpr)

    # 2. Experto generado: ATAQUES generados (payloads de TRAIN) + BENIGNO REAL
    #    como negativos. Darle benigno real NO rompe el aislamiento: el experto
    #    real sigue intocado; sólo el experto generado aprende qué es tráfico
    #    normal de verdad, para no marcar producción legítima como ataque. Sin
    #    ese contexto, su umbral se iría a ~1.0 (nunca dispara) y no aportaría.
    print("experto generado: generando train/holdout y entrenando...")
    gen_attacks = AdversarialGenerator(seed=args.seed).build(per_category=args.per_category, payload_split="train")
    gen_attacks = gen_attacks[gen_attacks.label == 1]
    real_benign_train = real_train[real_train.label == 0]
    gen_train = pd.concat([gen_attacks, real_benign_train], ignore_index=True)
    gen_hold = AdversarialGenerator(seed=args.seed + 1).build(per_category=args.per_category, payload_split="holdout")
    generated = OctoMatrixModel.fit(texts(gen_train), gen_train.label.to_numpy())
    # Umbral del experto generado: SÓLO contra benigno real (val). Eso es lo que
    # controla su FPR sobre el tráfico que de verdad veremos. Su recall se mide
    # después en el holdout, sin que el holdout toque el umbral (sin fuga).
    real_benign_val = real_val[real_val.label == 0]
    generated.set_threshold_for_fpr(texts(real_benign_val), real_benign_val.label.to_numpy(), args.fpr)
    generated.metadata["expert"] = "generated"

    # 3. Ensamblar y guardar
    ensemble = ScoreUnionEnsemble([Expert("real", real), Expert("generated", generated)])
    ensemble.save(args.out)
    print(f"MoE guardado en {args.out}/ (umbral real={real.threshold.value:.3f}, "
          f"generado={generated.threshold.value:.3f})")

    # 4. Evaluación
    print("\n=== AISLAMIENTO: sobre TEST real (in_source) ===")
    yr = real_test.label.to_numpy()
    real_fired = real.scores(texts(real_test)) >= real.threshold.value
    ens_fired = ensemble.predict(texts(real_test)).astype(bool)
    f0, r0 = fpr_recall(real_fired, yr)
    f1, r1 = fpr_recall(ens_fired, yr)
    print(f"  experto real solo: FPR={f0:.4f} recall={r0:.4f}")
    print(f"  ensemble (OR):     FPR={f1:.4f} recall={r1:.4f}   (FPR extra={f1 - f0:+.4f})")

    print("\n=== GANANCIA: sobre HOLDOUT generado (payloads no vistos) ===")
    yg = gen_hold.label.to_numpy()
    rf, rr = fpr_recall(real.scores(texts(gen_hold)) >= real.threshold.value, yg)
    ef, er = fpr_recall(ensemble.predict(texts(gen_hold)).astype(bool), yg)
    print(f"  experto real solo: FPR={rf:.4f} recall={rr:.4f}")
    print(f"  ensemble (OR):     FPR={ef:.4f} recall={er:.4f}   (recall +{er - rr:+.4f})")
    # recall por categoría que aporta el generado
    gh = gen_hold.assign(_fired=ensemble.predict(texts(gen_hold)).astype(bool))
    by = gh[gh.label == 1].groupby("attack_type")["_fired"].mean().round(3).sort_values()
    print("  recall ensemble por categoría (holdout):")
    for at, v in by.items():
        print(f"    {at:22s} {v}")


if __name__ == "__main__":
    main()
