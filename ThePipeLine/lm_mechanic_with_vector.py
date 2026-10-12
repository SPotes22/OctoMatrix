# Copyright (c) 2025 Santiago Potes Giraldo - BlackByte
# Licencia de Paz Digital v1.0 (LPD). Ver LICENSE.
"""
LM MECHANIC (VectorOfThought): memoria de payloads vista por similitud.

Reescrito en la Fase 0. La versión anterior tenía tres bugs que lo hacían
inservible:
  1. Aplicaba normalize(l2) a la entrada antes de predecir, pero el modelo se
     entrenó sin normalizar, así que las predicciones no valían. -> Ahora la
     inferencia la hace el artefacto OctoMatrixModel, con su propio pre-proceso.
  2. Leía `security_model.csv` (quitando .pkl), que no existe. -> Eliminado.
  3. Sin API key de Gemini, generaba vectores con np.random.rand, es decir,
     "similitudes" aleatorias. -> Ahora el embedding por defecto es local y
     determinista (hashing char n-gram); Gemini es opcional y explícito.

Esto NO es el detector (ese es octomatrix.pipeline). Es una herramienta de
análisis: dado un payload nuevo, encuentra el más parecido ya visto y su
etiqueta, para explicar por qué el modelo reacciona como reacciona.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(message)s")


class VectorMemory:
    """Memoria de payloads: embedding local determinista + vecino más cercano."""

    def __init__(self, n_features: int = 2048):
        # HashingVectorizer: sin estado que entrenar, reproducible, sin red.
        self._vec = HashingVectorizer(
            analyzer="char_wb", ngram_range=(2, 4), n_features=n_features, alternate_sign=False, norm="l2"
        )
        self.texts: list[str] = []
        self.labels: list[int] = []
        self._matrix = None

    def add(self, texts, labels) -> None:
        self.texts.extend(texts)
        self.labels.extend(int(x) for x in labels)
        self._matrix = self._vec.transform(self.texts)

    def most_similar(self, text: str, k: int = 1) -> list[dict]:
        if self._matrix is None or not self.texts:
            return []
        q = self._vec.transform([text])
        sims = cosine_similarity(q, self._matrix)[0]
        top = np.argsort(sims)[::-1][:k]
        return [
            {"text": self.texts[i], "label": self.labels[i], "similarity": float(sims[i])}
            for i in top
        ]


def build_from_training(parquet: str | Path, limit: int | None = 5000) -> VectorMemory:
    import pandas as pd

    from octomatrix.data.text import request_text

    df = pd.read_parquet(parquet)
    if limit:
        df = df.sample(min(limit, len(df)), random_state=42)
    texts = [request_text(*r) for r in zip(df.method, df.path, df["query"], df.body, df.headers)]
    mem = VectorMemory()
    mem.add(texts, df.label.to_numpy())
    return mem


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/processed/http_clean.parquet")
    ap.add_argument("--query", default="GET /login?user=admin' OR '1'='1' --")
    ap.add_argument("-k", type=int, default=3)
    args = ap.parse_args()

    mem = build_from_training(args.data)
    print(f"memoria: {len(mem.texts)} payloads")
    for hit in mem.most_similar(args.query, k=args.k):
        tag = "ATAQUE" if hit["label"] else "normal"
        logging.info(f"  sim={hit['similarity']:.3f}  [{tag}]  {hit['text'][:90]}")


if __name__ == "__main__":
    main()
