#!/usr/bin/env python3
"""ETAPA: POST-TRAINING  (feedback de producción, el lazo que faltaba)

El pipeline del proyecto era {Generate/Collect -> Train -> Test -> Export}.
Esta etapa añade el POST-TEST: qué pasa DESPUÉS de desplegar. El modelo ve
tráfico real, marca lo dudoso, una persona lo revisa, y eso alimenta un
reentrenamiento MANUAL (nunca automático: ahí entraría el data poisoning).

Este archivo es una DEMO del lazo sobre tráfico de ejemplo:
  producción -> log -> muestreo por incertidumbre -> (revisión humana) -> store

En real, el tráfico sale de tus apps (access logs) y la revisión la hace una
persona en el review_queue. Aquí se simula la revisión para mostrar el flujo.

    python presentation/3_post_training.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from octomatrix import feedback  # noqa: E402
from octomatrix.data.text import request_text  # noqa: E402
from octomatrix.moe import ScoreUnionEnsemble  # noqa: E402

DEMO_TRAFFIC = [
    ("GET", "/api/users", "id=42", ""),
    ("GET", "/static/app.js", "", ""),
    ("POST", "/login", "", "user=admin&pass=admin' OR '1'='1"),
    ("GET", "/search", "q=<script>alert(1)</script>", ""),
    ("GET", "/download", "f=../../../etc/passwd", ""),
    ("GET", "/eventos", "categoria=rock", ""),
    ("GET", "/proxy", "url=http://169.254.169.254/latest/meta-data/", ""),
]


def main() -> None:
    moe_dir = ROOT / "models/moe"
    if not moe_dir.exists():
        sys.exit("falta models/moe — corre presentation/1_generar_modelo.py primero")
    ens = ScoreUnionEnsemble.load(moe_dir)

    fb = ROOT / "data/feedback"
    pred_log, queue, store = fb / "prediction_log.jsonl", fb / "review_queue.jsonl", fb / "labeled_store.jsonl"
    for p in (pred_log, queue, store):
        p.unlink(missing_ok=True)

    print("=" * 70)
    print("OCTOMATRIX · POST-TRAINING (feedback de producción) — DEMO")
    print("=" * 70)

    texts = [request_text(m, p, q, b) for m, p, q, b in DEMO_TRAFFIC]

    print("\n[1] LOG: el MoE evalúa el tráfico en producción")
    feedback.log_predictions(ens, texts, pred_log)
    for rec in feedback.read_jsonl(pred_log):
        verdict = "🚨 ATAQUE" if rec["decision"] else "✅ normal"
        who = ",".join(rec["fired_experts"]) or "-"
        print(f"  {verdict}  margen={rec['margin']:+.3f}  expertos={who:16s}  {rec['text'][:48]}")

    print("\n[2] QUEUE: se seleccionan los casos DUDOSOS (margen pequeño) para revisión")
    items = feedback.select_for_review(feedback.read_jsonl(pred_log), band=0.15, max_items=20)
    feedback.write_review_queue(items, queue)
    print(f"  {len(items)} casos encolados en {queue.relative_to(ROOT)}")
    for it in items:
        print(f"    margen={it['margin']:+.3f}  {it['text'][:56]}")

    print("\n[3] REVISIÓN HUMANA (aquí simulada): una persona pone label y su nombre")
    rows = feedback.read_jsonl(queue)
    for r in rows:
        r["label"] = "1" if ("OR '1'='1" in r["text"] or "etc/passwd" in r["text"] or "169.254" in r["text"]) else "0"
        r["reviewer"] = "analista_demo"
    queue.write_text("\n".join(__import__("json").dumps(r, ensure_ascii=False) for r in rows))

    print("\n[4] INGEST: lo revisado entra al store etiquetado (base del reentreno MANUAL)")
    stats = feedback.ingest_reviewed(queue, store)
    print(f"  {stats}")
    print(f"\n  El store {store.relative_to(ROOT)} NO reentrena solo: es la entrada, ya")
    print("  revisada por humanos, para un reentreno manual del experto que corresponda.")
    print("\n✅ Lazo de post-training demostrado.")


if __name__ == "__main__":
    main()
