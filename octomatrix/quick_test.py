"""Prueba de humo del detector: imprime el % de confianza REAL en unos casos.

A diferencia de la tabla del POC original (que prometía detecciones), esto
carga el artefacto entrenado y muestra lo que de verdad predice.

    python -m octomatrix.quick_test --model models/octomatrix.joblib
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import OctoMatrixModel

CASES = [
    ("GET", "/api/users?id=42", "", "normal"),
    ("GET", "/static/js/app.js", "", "normal (allowlist)"),
    ("POST", "/login", "user=admin&pass=admin' OR '1'='1", "sqli"),
    ("GET", "/search?q=<script>alert(1)</script>", "", "xss"),
    ("GET", "/download?f=../../../etc/passwd", "", "path_traversal"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=Path("models/octomatrix.joblib"))
    args = ap.parse_args()

    from .data.text import request_text

    model = OctoMatrixModel.load(args.model)
    thr = model.threshold.value
    print(f"modelo: {args.model}  umbral={thr:.3f}  (FPR objetivo={model.threshold.target_fpr})")
    print(f"{'esperado':20s} {'veredicto':9s} {'P(ataque)':>9s}  petición")
    for method, path, body, expected in CASES:
        q = path.split("?", 1)[1] if "?" in path else ""
        p = path.split("?", 1)[0]
        text = request_text(method, p, q, body)
        score = float(model.scores([text])[0])
        verdict = "🚨 ATAQUE" if score >= thr else "✅ NORMAL"
        print(f"{expected:20s} {verdict:9s} {score:9.1%}  {method} {path} {body}".rstrip())


if __name__ == "__main__":
    main()
