#!/usr/bin/env python3
"""POST-TEST: lazo de feedback de producción (Fase 5), sin reentreno automático.

Subcomandos:
  log       corre el MoE sobre tráfico (JSONL o parquet con 'text') y lo registra
  queue     selecciona las predicciones dudosas (margen pequeño) para revisión
  ingest    pasa las filas revisadas (label 0/1 + reviewer) al store etiquetado

Ejemplo de ciclo:
  python scripts/postprod.py log   --model models/moe --traffic trafico.jsonl
  python scripts/postprod.py queue --band 0.1 --max 300
  # (una persona rellena 'label' y 'reviewer' en data/feedback/review_queue.jsonl)
  python scripts/postprod.py ingest

El store etiquetado (data/feedback/labeled_store.jsonl) NO reentrena solo:
es la entrada, ya revisada por humanos, para un reentreno manual posterior.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from octomatrix import feedback  # noqa: E402
from octomatrix.moe import ScoreUnionEnsemble  # noqa: E402

FEEDBACK_DIR = Path("data/feedback")
PRED_LOG = FEEDBACK_DIR / "prediction_log.jsonl"
REVIEW_QUEUE = FEEDBACK_DIR / "review_queue.jsonl"
LABELED_STORE = FEEDBACK_DIR / "labeled_store.jsonl"


def _load_traffic_texts(path: Path) -> list[str]:
    from octomatrix.data.text import request_text

    if path.suffix == ".parquet":
        import pandas as pd

        df = pd.read_parquet(path)
        if "text" in df.columns:
            return df["text"].tolist()
        return [request_text(*r) for r in zip(df.method, df.path, df["query"], df.body, df.get("headers", ""))]
    texts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        texts.append(obj["text"] if isinstance(obj, dict) else str(obj))
    return texts


def cmd_log(args):
    ensemble = ScoreUnionEnsemble.load(args.model)
    texts = _load_traffic_texts(Path(args.traffic))
    n = feedback.log_predictions(ensemble, texts, args.pred_log)
    print(f"registradas {n} predicciones -> {args.pred_log}")


def cmd_queue(args):
    preds = feedback.read_jsonl(args.pred_log)
    already = {r["request_hash"] for r in feedback.read_jsonl(args.store)}
    items = feedback.select_for_review(preds, band=args.band, max_items=args.max, exclude_hashes=already)
    n = feedback.write_review_queue(items, args.queue)
    print(f"{n} casos dudosos (|margen|<={args.band}) -> {args.queue}")
    print("rellena 'label' (0/1) y 'reviewer' y luego corre: postprod.py ingest")


def cmd_ingest(args):
    stats = feedback.ingest_reviewed(args.queue, args.store)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"store etiquetado: {args.store} (reentreno manual, no automático)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("log")
    p.add_argument("--model", type=Path, default=Path("models/moe"))
    p.add_argument("--traffic", required=True, help="JSONL con {'text':...} o parquet del esquema")
    p.add_argument("--pred-log", type=Path, default=PRED_LOG)
    p.set_defaults(func=cmd_log)

    p = sub.add_parser("queue")
    p.add_argument("--pred-log", type=Path, default=PRED_LOG)
    p.add_argument("--queue", type=Path, default=REVIEW_QUEUE)
    p.add_argument("--store", type=Path, default=LABELED_STORE)
    p.add_argument("--band", type=float, default=0.1)
    p.add_argument("--max", type=int, default=500)
    p.set_defaults(func=cmd_queue)

    p = sub.add_parser("ingest")
    p.add_argument("--queue", type=Path, default=REVIEW_QUEUE)
    p.add_argument("--store", type=Path, default=LABELED_STORE)
    p.set_defaults(func=cmd_ingest)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
