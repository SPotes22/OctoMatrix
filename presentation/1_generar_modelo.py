#!/usr/bin/env python3
"""ETAPA: GENERACIÓN DEL MODELO ENTRENADO  (TRAIN + EXPORT)

Presentación limpia del proyecto OctoMatrix. Este archivo genera TODOS los
artefactos entrenados, en orden, llamando a los módulos ya probados (no
reimplementa nada):

  1. Datos reales (CSIC 2010 + ECML/PKDD 2007 + OWASP CRS)  -> descarga + unifica
  2. Limpieza y deduplicación                               -> http_clean.parquet
  3. Splits por grupos y escenarios                         -> splits.parquet
  4. Experto REAL (texto)                                   -> models/octomatrix.joblib
  5. MoE: experto real + experto GENERADO (OWASP)           -> models/moe/
  6. Experto de FLUJOS (CIC-IDS2018, opcional)              -> models/flows.joblib

A diferencia del POC original (72 frases escritas a mano), aquí se entrena con
~75.000 peticiones reales. Cada paso es idempotente: si el insumo ya existe, no
se rehace.

    python presentation/1_generar_modelo.py              # texto (real + generado)
    python presentation/1_generar_modelo.py --with-flows # además, experto de flujos
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(desc: str, *cmd: str) -> None:
    print(f"\n▶ {desc}\n  $ {' '.join(cmd)}")
    subprocess.run([sys.executable, *cmd], cwd=ROOT, check=True)


def exists(*parts: str) -> bool:
    return (ROOT / Path(*parts)).exists()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--with-flows", action="store_true", help="entrenar también el experto de flujos (pesado)")
    ap.add_argument("--force", action="store_true", help="rehacer aunque los insumos existan")
    args = ap.parse_args()
    f = args.force

    print("=" * 70)
    print("OCTOMATRIX · GENERACIÓN DEL MODELO ENTRENADO (TRAIN + EXPORT)")
    print("=" * 70)

    if f or not exists("data/raw/MANIFEST.json"):
        run("1/6 Descargar fuentes reales", "scripts/download_data.py")
    if f or not exists("data/processed/http_requests.parquet"):
        run("2/6 Unificar al esquema común", "-m", "octomatrix.data.build")
    if f or not exists("data/processed/http_clean.parquet"):
        run("3/6 Limpiar y deduplicar", "-m", "octomatrix.data.clean")
    if f or not exists("data/processed/splits.parquet"):
        run("4/6 Splits por grupos y escenarios", "-m", "octomatrix.data.splits")
    if f or not exists("models/octomatrix.joblib"):
        run("5/6 Entrenar experto REAL", "scripts/train.py", "--scenario", "in_source", "--fpr", "0.01")
    run("6/6 Ensamblar MoE (real + GENERADO)", "scripts/build_moe.py", "--fpr", "0.01")

    if args.with_flows:
        if f or not exists("data/processed/cicids2018"):
            run("+ Descargar días de flujos (CIC-IDS2018)", "scripts/download_data.py", "--skip-http", "--cicids", "web")
            run("+ Construir parquet de flujos", "-m", "octomatrix.data.build", "--sources", "", "--cicids")
        run("+ Entrenar experto de FLUJOS", "scripts/train_flows.py", "--out", "models/flows.joblib")

    print("\n✅ Artefactos generados en models/. Siguiente: presentation/2_test_modelo.py")


if __name__ == "__main__":
    main()
