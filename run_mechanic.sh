#!/usr/bin/env bash
# run_mechanic.sh - entrena y analiza el detector OctoMatrix.
# Las rutas son relativas al repo (antes apuntaban a /home/arachne/..., fijo).
set -euo pipefail
cd "$(dirname "$0")"

SCENARIO="${1:-in_source}"
MODEL="${MODEL:-models/octomatrix.joblib}"

echo "🛠️  OctoMatrix Mechanic | escenario: $SCENARIO"

# 0. Datos (idempotente: no rebaja si ya están)
[ -f data/processed/http_clean.parquet ] || {
  echo "📥 Preparando datos..."
  python scripts/download_data.py
  python -m octomatrix.data.build
  python -m octomatrix.data.clean
  python -m octomatrix.data.splits
}

# 1. Entrenar artefacto único
echo "🎯 Entrenando..."
python scripts/train.py --scenario "$SCENARIO" --out "$MODEL" --fpr 0.01

# 2. Evaluar
echo "📊 Evaluando..."
python scripts/evaluate.py --model "$MODEL" --scenario "$SCENARIO" --out output/eval_"$SCENARIO".json

# 3. Matriz de confusión + análisis de errores
python matrix_mechanic.py --model "$MODEL" --scenario "$SCENARIO"

echo "✅ Listo. Ver output/confusion_matrix.png y output/mechanic_report.json"
