"""Limpieza y unificación del dataset HTTP.

Produce `data/processed/http_clean.parquet`: nuestro dataset unificado, a
partir de las fuentes externas ya normalizadas por los loaders. Las reglas
son deliberadamente CONSERVADORAS y quedan justificadas en docs/DATA.md.

Principio: no borrar ataques legítimos. Al analizar los datos se vio que lo
que "parece roto" suele ser el ataque:
- 386 peticiones cuyo path no empieza por "/" son todas command injection o
  path traversal de ECML (p. ej. `..%5c../winnt/system32/cmd.exe`).
- 76 peticiones con bytes de control son ataques con null byte incrustado.
Por eso NO se filtra por "forma rara" ni por "ausencia de firma de ataque"
(una regex de firmas incompleta marcaría como ruido el 82% de los ataques
reales del CRS). Esos criterios se descartaron a propósito.

Lo que SÍ se quita:
1. VACÍAS: sin contenido clasificable (method+path+query+body con < 3 chars
   útiles). Son peticiones que no aportan señal.
2. CASI-DUPLICADAS: se conserva una fila por grupo de casi-duplicados
   (mismo `group_id`, ver text.skeleton). CSIC reenvía la misma plantilla
   cientos de veces cambiando sólo números; esas copias inflan el dataset y
   sesgan el entrenamiento hacia unas pocas plantillas. Se prioriza conservar
   el representante más informativo (el de texto más largo del grupo) y, ante
   empate, el primero.

Ruido de etiqueta de CSIC (peticiones ilegales pero no maliciosas marcadas
como "anómalas"): NO se elimina automáticamente, porque no hay forma fiable
de separarlo de los ataques sin una regex de firmas que ya demostró ser
inexacta. Se documenta como limitación conocida.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .schema import validate
from .text import group_id

MIN_CONTENT = 3


def _content_len(df: pd.DataFrame) -> pd.Series:
    return (df["path"].str.len() + df["query"].str.len() + df["body"].str.len())


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = validate(df).copy()
    stats = {"input_rows": int(len(df))}

    # 1. Vacías / sin contenido clasificable.
    empties = _content_len(df) < MIN_CONTENT
    stats["dropped_empty"] = int(empties.sum())
    df = df[~empties].copy()

    # 2. Colapsar casi-duplicados: una fila por group_id, la más larga.
    df["group_id"] = [
        group_id(m, p, q, b) for m, p, q, b in zip(df["method"], df["path"], df["query"], df["body"])
    ]
    df["_len"] = _content_len(df)
    df = (
        df.sort_values("_len", ascending=False, kind="stable")
        .drop_duplicates("group_id", keep="first")
        .drop(columns="_len")
        .sort_index()
        .reset_index(drop=True)
    )
    stats["collapsed_near_duplicates"] = stats["input_rows"] - stats["dropped_empty"] - len(df)

    stats["output_rows"] = int(len(df))
    stats["by_source"] = {
        s: {"normal": int((g.label == 0).sum()), "attack": int((g.label == 1).sum())}
        for s, g in df.groupby("source")
    }
    stats["by_attack_type"] = {k: int(v) for k, v in df["attack_type"].value_counts().items()}
    return df, stats


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    args = ap.parse_args(argv)

    df = pd.read_parquet(args.processed_dir / "http_requests.parquet")
    clean_df, stats = clean(df)
    clean_df.to_parquet(args.processed_dir / "http_clean.parquet", index=False)
    (args.processed_dir / "clean_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    print(f"{stats['input_rows']:,} -> {stats['output_rows']:,} filas")
    print(f"  vacías eliminadas:        {stats['dropped_empty']:,}")
    print(f"  casi-duplicadas colapsadas: {stats['collapsed_near_duplicates']:,}")
    atk = sum(v["attack"] for v in stats["by_source"].values())
    print(f"  resultado: {atk:,} ataques, {stats['output_rows'] - atk:,} normales")
    print(f"  -> {args.processed_dir / 'http_clean.parquet'}")


if __name__ == "__main__":
    main()
