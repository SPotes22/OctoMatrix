#!/usr/bin/env python3
"""ETAPA: TEST DEL MODELO  (evaluación HONESTA)

El POC original validaba con un `train_test_split` sobre 72 frases escritas a
mano (test de 15 muestras) y reportaba "83% de precisión". Eso no es una prueba.

Este archivo corre la evaluación de verdad sobre conjuntos de test reales y
separados, e imprime las métricas que importan para un WAF, sin maquillar:

  - IN-SOURCE: test real de las mismas fuentes (línea base optimista).
  - CROSS-SOURCE: entrenar con unas fuentes y probar con OTRA (generalización).
  - ADVERSARIAL: set propio de 19 categorías OWASP (cobertura).
  - FLUJOS: experto de red, si existe el artefacto.

Requiere haber corrido antes presentation/1_generar_modelo.py.

    python presentation/2_test_modelo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from octomatrix.data.splits import load_scenario  # noqa: E402
from octomatrix.data.text import request_text  # noqa: E402
from octomatrix.pipeline import OctoMatrixModel  # noqa: E402


def texts(df):
    return [request_text(*r) for r in zip(df.method, df.path, df["query"], df.body, df.headers)]


def recall_at_fpr(y, s, fpr):
    neg = np.sort(s[y == 0])
    if not len(neg):
        return float("nan")
    thr = float(np.nextafter(neg[min(len(neg) - 1, int(np.ceil((1 - fpr) * len(neg))))], 1.0))
    return float((s[y == 1] >= thr).mean())


def line(name, y, s):
    roc = roc_auc_score(y, s) if 0 < y.sum() < len(y) else float("nan")
    print(f"  {name:28s} ROC={roc:5.3f}  PR={average_precision_score(y, s):5.3f}  "
          f"recall@FPR1%={recall_at_fpr(y, s, 0.01):5.3f}  recall@FPR0.1%={recall_at_fpr(y, s, 0.001):5.3f}")


def main() -> None:
    proc = ROOT / "data/processed"
    print("=" * 78)
    print("OCTOMATRIX · TEST DEL MODELO (evaluación honesta)")
    print("=" * 78)

    model_path = ROOT / "models/octomatrix.joblib"
    if not model_path.exists():
        sys.exit("falta models/octomatrix.joblib — corre presentation/1_generar_modelo.py primero")
    real = OctoMatrixModel.load(model_path)

    print("\n[1] EXPERTO DE TEXTO (real)")
    for scen in ("in_source", "csic_ecml_to_crs"):
        _, _, test = load_scenario(scen, proc)
        s = real.scores(texts(test)); y = test.label.to_numpy()
        line(scen, y, s)
    print("     in_source = optimista; cross-source mide generalización a una fuente NUEVA.")

    print("\n[2] COBERTURA OWASP (set adversarial propio, 19 categorías)")
    moe_dir = ROOT / "models/moe"
    adv = proc / "adversarial.parquet"
    if adv.exists() and moe_dir.exists():
        from octomatrix.moe import ScoreUnionEnsemble
        ens = ScoreUnionEnsemble.load(moe_dir)
        df = pd.read_parquet(adv)
        fired = ens.predict(texts(df)).astype(bool); y = df.label.to_numpy()
        print(f"  MoE (real+generado)          recall={fired[y==1].mean():5.3f}  FPR={fired[y==0].mean():5.3f}")
        real_fired = (real.scores(texts(df)) >= real.threshold.value)
        print(f"  sólo experto real            recall={real_fired[y==1].mean():5.3f}  "
              f"(el generado suma la cobertura que faltaba)")
    else:
        print("  (genera el MoE y el set adversarial con 1_generar_modelo.py)")

    print("\n[3] EXPERTO DE FLUJOS (red: DoS, fuerza bruta)")
    flows = ROOT / "models/flows.joblib"
    if flows.exists():
        from octomatrix.flows import FlowModel
        import scripts.train_flows as tf
        fdf = tf.load_flows(proc, 200_000); _, _, ftest = tf.temporal_split(fdf); ftest = ftest.reset_index(drop=True)
        fm = FlowModel.load(flows); s = fm.scores(ftest); y = ftest.label.to_numpy()
        line("cicids2018", y, s)
        print("     NOTA: CIC-IDS2018 es trivialmente separable; no es 100% en producción.")
    else:
        print("  (entrena con 1_generar_modelo.py --with-flows)")

    print("\n✅ Fin del test. Resumen honesto en presentation/README.md")


if __name__ == "__main__":
    main()
