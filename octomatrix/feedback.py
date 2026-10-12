"""POST-TEST: feedback de producción (Fase 5), sin reentreno automático.

El lazo es:  producción -> log -> muestreo por incertidumbre -> revisión humana
             -> store etiquetado -> (reentreno MANUAL, revisado)

No se reentrena solo: cada etiqueta nueva pasa por una persona antes de poder
entrar a un dataset de entrenamiento. Ahí es donde reaparecería el riesgo de
data poisoning (alguien envenenando el feedback), así que el lazo queda
explícitamente roto en la revisión humana.

Formatos (JSONL, una petición por línea):
- prediction log:  lo que el modelo decidió en producción.
- review queue:    subconjunto dudoso, con 'label' y 'reviewer' en blanco.
- labeled store:   lo ya revisado; de aquí puede salir un dataset de reentreno.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


def request_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()[:16]


@dataclass
class Prediction:
    request_hash: str
    text: str
    decision: int                      # 1 ataque, 0 normal (OR del ensemble)
    fired_experts: list[str]           # qué expertos dispararon
    margin: float                      # cercanía a la frontera (ver moe.margin)
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def log_predictions(ensemble, texts: Iterable[str], path: str | Path) -> int:
    """Corre el ensemble sobre tráfico (sin etiqueta) y añade al prediction log."""
    texts = list(texts)
    if not texts:
        return 0
    decisions = ensemble.predict(texts)
    attribs = ensemble.attributions(texts)
    margins = ensemble.margin(texts)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for text, d, experts, m in zip(texts, decisions, attribs, margins):
            rec = Prediction(request_hash(text), text, int(d), experts, float(m))
            fh.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
    return len(texts)


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_for_review(
    predictions: list[dict],
    band: float = 0.1,
    max_items: int = 500,
    exclude_hashes: set[str] | None = None,
) -> list[dict]:
    """Muestreo por incertidumbre: las peticiones cuyo |margin| es más pequeño.

    Un margin cercano a 0 significa que algún experto quedó justo en su frontera:
    son los casos donde una etiqueta humana aporta más. `band` acota qué cuenta
    como dudoso; dentro de la banda se ordena por cercanía a la frontera.
    exclude_hashes evita re-encolar lo ya revisado.
    """
    exclude = exclude_hashes or set()
    cand = [p for p in predictions if abs(p.get("margin", 1.0)) <= band and p["request_hash"] not in exclude]
    cand.sort(key=lambda p: abs(p.get("margin", 1.0)))
    # dedup por request_hash conservando el primero (el más dudoso)
    seen, out = set(), []
    for p in cand:
        if p["request_hash"] in seen:
            continue
        seen.add(p["request_hash"])
        out.append({**p, "label": "", "reviewer": "", "attack_type": ""})
        if len(out) >= max_items:
            break
    return out


def write_review_queue(items: list[dict], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    return len(items)


def ingest_reviewed(review_path: str | Path, labeled_store: str | Path) -> dict:
    """Pasa al store etiquetado SÓLO las filas con label válido (0/1) y revisor.

    Devuelve conteos. No reentrena nada: eso es un paso manual posterior.
    """
    rows = read_jsonl(review_path)
    accepted, skipped = [], 0
    for r in rows:
        label, reviewer = str(r.get("label", "")).strip(), str(r.get("reviewer", "")).strip()
        if label in ("0", "1") and reviewer:
            accepted.append({
                "request_hash": r["request_hash"], "text": r["text"],
                "label": int(label), "attack_type": r.get("attack_type", "") or ("unspecified" if label == "1" else "normal"),
                "reviewer": reviewer, "reviewed_at": datetime.now(timezone.utc).isoformat(),
            })
        else:
            skipped += 1
    store = Path(labeled_store)
    store.parent.mkdir(parents=True, exist_ok=True)
    existing = {r["request_hash"] for r in read_jsonl(store)}
    added = 0
    with open(store, "a", encoding="utf-8") as fh:
        for r in accepted:
            if r["request_hash"] in existing:
                continue
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            added += 1
    return {"reviewed": len(rows), "accepted": len(accepted), "added_to_store": added,
            "skipped_unlabeled": skipped}
