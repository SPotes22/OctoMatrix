#!/usr/bin/env bash
# reproducir.sh — replica OctoMatrix de cero en una máquina limpia (p. ej. Cloud).
# Uso:  git clone https://github.com/SPotes22/OctoMatrix && cd OctoMatrix && bash reproducir.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "== 1/6 Entorno =="
python3 -m venv .venv && source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo "== 2/6 Descargar fuentes reales (CSIC 2010 + ECML/PKDD 2007 + OWASP CRS) =="
python scripts/download_data.py              # ~50 MB, no se versiona

echo "== 3/6 Unificar, limpiar y particionar =="
python -m octomatrix.data.build              # esquema común
python -m octomatrix.data.clean              # -> data/processed/http_clean.parquet
python -m octomatrix.data.splits             # splits por grupos y escenarios
python -m octomatrix.data.adversarial        # set adversarial propio (19 cat. OWASP)

echo "== 4/6 Entrenar y ensamblar el MoE =="
python scripts/train.py --scenario in_source --fpr 0.01   # experto real -> models/octomatrix.joblib
python scripts/build_moe.py --fpr 0.01                     # + experto generado -> models/moe/

echo "== 5/6 (opcional) Experto de flujos CIC-IDS2018 (~0,8 GB) =="
if [ "${WITH_FLOWS:-0}" = "1" ]; then
  python scripts/download_data.py --skip-http --cicids web
  python -m octomatrix.data.build --sources "" --cicids
  python scripts/train_flows.py --out models/flows.joblib
fi

echo "== 6/6 Evaluar (métricas honestas) =="
python scripts/waf_report.py --scenario in_source
python scripts/waf_report.py --adversarial

echo "✅ Listo. Artefactos en models/. Ver presentation/ para la demo end-to-end."
