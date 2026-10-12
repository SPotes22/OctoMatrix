"""Construye el dataset unificado del experto HTTP (y opcionalmente el de flujos).

Uso:
    python -m octomatrix.data.build                       # CSIC + ECML + CRS
    python -m octomatrix.data.build --own-logs logs/pereira:pereira
    python -m octomatrix.data.build --cicids               # además, flujos

Entradas en data/raw/ (ver scripts/download_data.py). Salidas en data/processed/:
    http_requests.parquet   esquema de octomatrix/data/schema.py
    http_stats.json         conteos y lo que se descartó en la deduplicación
    cicids2018/*.parquet    flujos, esquema propio (sólo con --cicids)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .loaders import access_log, cicids2018, crs_regression, csic2010, ecml_pkdd2007
from .schema import REQUEST_KEY, to_frame, validate

HTTP_SOURCES = {
    "csic2010": lambda raw: csic2010.load(raw / "csic2010"),
    "ecml_pkdd2007": lambda raw: ecml_pkdd2007.load(raw / "ecml_pkdd2007"),
    "crs_regression": lambda raw: crs_regression.load(raw / "coreruleset"),
}


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Quita duplicados exactos y elimina peticiones con etiquetas en conflicto.

    - Duplicado exacto (misma petición y misma etiqueta): se queda la primera
      aparición, así que el orden de las fuentes importa.
    - Conflicto (misma petición, etiquetas distintas): se eliminan TODAS las
      copias, porque no hay forma de saber cuál es la buena.
    """
    stats: dict = {}
    n_labels = df.groupby(REQUEST_KEY, sort=False)["label"].transform("nunique")
    conflict = n_labels > 1
    stats["conflicting_rows_dropped"] = df[conflict]["source"].value_counts().to_dict()
    df = df[~conflict]

    dup = df.duplicated(subset=REQUEST_KEY, keep="first")
    stats["duplicate_rows_dropped"] = df[dup]["source"].value_counts().to_dict()
    df = df[~dup].reset_index(drop=True)
    return df, stats


def summarize(df: pd.DataFrame) -> dict:
    by_source = (
        df.groupby(["source", "label"]).size().unstack(fill_value=0).rename(columns={0: "normal", 1: "attack"})
    )
    return {
        "rows": int(len(df)),
        "attack": int(df["label"].sum()),
        "normal": int((df["label"] == 0).sum()),
        "by_source": {s: {k: int(v) for k, v in row.items()} for s, row in by_source.iterrows()},
        "by_attack_type": {k: int(v) for k, v in df["attack_type"].value_counts().items()},
        "by_source_split": {
            f"{s}/{sp}": int(n) for (s, sp), n in df.groupby(["source", "origin_split"]).size().items()
        },
    }


def build_http(raw_dir: Path, sources, own_logs) -> tuple[pd.DataFrame, dict]:
    frames, loaded = [], {}
    for name in sources:
        df = to_frame(HTTP_SOURCES[name](raw_dir))
        loaded[name] = int(len(df))
        frames.append(df)
        print(f"  {name}: {len(df):,} filas")
    for spec in own_logs:
        path, _, src = spec.partition(":")
        df = to_frame(access_log.load(path, source=src or Path(path).stem))
        loaded[df["source"].iat[0] if len(df) else spec] = int(len(df))
        frames.append(df)
        print(f"  {spec}: {len(df):,} filas")
    df = validate(pd.concat(frames, ignore_index=True))
    df, dedup_stats = deduplicate(df)
    stats = {"loaded": loaded, "dedup": dedup_stats, **summarize(df)}
    return df, stats


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--sources", default=",".join(HTTP_SOURCES), help="fuentes HTTP separadas por coma")
    ap.add_argument(
        "--own-logs", action="append", default=[], metavar="RUTA[:NOMBRE]", help="access log propio (repetible)"
    )
    ap.add_argument("--cicids", action="store_true", help="procesar también CIC-IDS2018 (flujos)")
    ap.add_argument("--cicids-max-rows", type=int, default=None, help="tope de filas por día (para pruebas)")
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    unknown = set(sources) - set(HTTP_SOURCES)
    if unknown:
        ap.error(f"fuentes desconocidas: {sorted(unknown)}")

    if sources or args.own_logs:
        print("Experto HTTP:")
        df, stats = build_http(args.raw_dir, sources, args.own_logs)
        df.to_parquet(args.out_dir / "http_requests.parquet", index=False)
        (args.out_dir / "http_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
        print(f"  -> {len(df):,} filas únicas ({stats['attack']:,} ataques, {stats['normal']:,} normales)")
        print(f"  -> {args.out_dir / 'http_requests.parquet'}")

    if args.cicids:
        print("Experto de flujos (CIC-IDS2018):")
        stats = cicids2018.build(
            args.raw_dir / "cicids2018", args.out_dir / "cicids2018", max_rows_per_day=args.cicids_max_rows
        )
        if not stats:
            print("  no hay CSV en", args.raw_dir / "cicids2018")
        (args.out_dir / "cicids2018_stats.json").write_text(json.dumps(stats, indent=2))
        for day, s in stats.items():
            print(f"  {day}: {s['rows']:,} flujos")


if __name__ == "__main__":
    main()
