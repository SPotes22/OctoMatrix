"""Splits train/val/test por grupos de casi-duplicados y escenarios entre fuentes.

Reglas:
1. Nunca se parte fila a fila. Todas las peticiones con el mismo `group_id`
   (ver text.skeleton) van al mismo split.
2. En los escenarios entre fuentes, cualquier fila de test cuyo grupo
   aparezca en train/val se excluye (fuga entre fuentes).
3. La augmentación (augment.py) sólo se aplica a train, dentro de load_scenario.

Escenarios:
    in_source           cada fuente partida 70/15/15 por grupos (línea base optimista)
    csic_to_crs         entrena CSIC  -> prueba CRS (ataques + negativos)
    csic_to_ecml        entrena CSIC  -> prueba ECML
    csic_ecml_to_crs    entrena CSIC+ECML -> prueba CRS
    ecml_crs_to_csic    entrena ECML+CRS  -> prueba CSIC (tráfico más realista)

Uso:
    python -m octomatrix.data.splits            # -> data/processed/splits.parquet + splits_stats.json
    from octomatrix.data.splits import load_scenario
    train, val, test = load_scenario("csic_to_crs", augment=True)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .text import group_id

TRAIN, VAL, TEST, UNUSED, LEAK = "train", "val", "test", "", "excluded_leak"

CRS = ["crs_regression", "crs_regression_negative"]
CSIC = ["csic2010"]
ECML = ["ecml_pkdd2007"]

# escenario -> (fuentes de entrenamiento, fuentes de test)
CROSS_SCENARIOS = {
    "csic_to_crs": (CSIC, CRS),
    "csic_to_ecml": (CSIC, ECML),
    "csic_ecml_to_crs": (CSIC + ECML, CRS),
    "ecml_crs_to_csic": (ECML + CRS, CSIC),
}
SCENARIOS = ["in_source", *CROSS_SCENARIOS]


def add_groups(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["group_id"] = [
        group_id(m, p, q, b) for m, p, q, b in zip(df["method"], df["path"], df["query"], df["body"])
    ]
    return df


def group_split(df: pd.DataFrame, fractions: dict[str, float], seed: int) -> pd.Series:
    """Asigna cada grupo a un split, estratificando por fuente y por si el grupo contiene ataques.

    Las fracciones se respetan en número de filas (no de grupos), aproximadamente.
    """
    rng = np.random.default_rng(seed)
    out = pd.Series(UNUSED, index=df.index, dtype=object)
    # Estrato = (fuente principal del grupo, si contiene ataques). Un grupo que
    # aparece en varias fuentes se asigna una sola vez, así nunca se parte.
    groups = df.groupby("group_id").agg(n=("label", "size"), attack=("label", "max"), source=("source", "first"))
    names = list(fractions)
    bounds = np.cumsum([fractions[k] for k in names])
    bounds = bounds / bounds[-1]
    for _, stratum in groups.groupby(["source", "attack"]):
        order = stratum.index.to_numpy()[rng.permutation(len(stratum))]
        cum = np.cumsum(stratum.loc[order, "n"].to_numpy()) / stratum["n"].sum()
        # el grupo va al split donde cae el punto medio de su tramo acumulado
        mid = cum - stratum.loc[order, "n"].to_numpy() / (2 * stratum["n"].sum())
        idx = np.searchsorted(bounds, mid, side="right").clip(max=len(names) - 1)
        assignment = dict(zip(order, (names[i] for i in idx)))
        mask = df["group_id"].isin(stratum.index)
        out[mask] = df.loc[mask, "group_id"].map(assignment)
    return out


def make_splits(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Devuelve un DataFrame alineado con df: group_id + una columna por escenario."""
    if "group_id" not in df.columns:
        df = add_groups(df)
    splits = pd.DataFrame({"group_id": df["group_id"]}, index=df.index)

    splits["in_source"] = group_split(df, {TRAIN: 0.70, VAL: 0.15, TEST: 0.15}, seed)

    for name, (train_src, test_src) in CROSS_SCENARIOS.items():
        col = pd.Series(UNUSED, index=df.index, dtype=object)
        fit = df[df["source"].isin(train_src)]
        col[fit.index] = group_split(fit, {TRAIN: 0.85, VAL: 0.15}, seed)
        test_mask = df["source"].isin(test_src)
        seen = set(df.loc[col.isin([TRAIN, VAL]), "group_id"])
        leak = test_mask & df["group_id"].isin(seen)
        col[test_mask] = TEST
        col[leak] = LEAK
        splits[name] = col
    check_no_group_leak(splits)
    return splits


def check_no_group_leak(splits: pd.DataFrame) -> None:
    """Falla si algún grupo aparece en test y también en train/val en cualquier escenario."""
    for name in SCENARIOS:
        col = splits[name]
        fit_groups = set(splits.loc[col.isin([TRAIN, VAL]), "group_id"])
        test_groups = set(splits.loc[col == TEST, "group_id"])
        shared = fit_groups & test_groups
        if shared:
            raise AssertionError(f"{name}: {len(shared)} grupos en train/val y test a la vez")
        train_groups = set(splits.loc[col == TRAIN, "group_id"])
        val_groups = set(splits.loc[col == VAL, "group_id"])
        if train_groups & val_groups:
            raise AssertionError(f"{name}: grupos compartidos entre train y val")


def summarize(df: pd.DataFrame, splits: pd.DataFrame) -> dict:
    stats = {"groups": int(splits["group_id"].nunique()), "rows": int(len(df)), "scenarios": {}}
    for name in SCENARIOS:
        tab = (
            pd.DataFrame({"split": splits[name], "source": df["source"], "label": df["label"]})
            .query("split != ''")
            .groupby(["split", "source", "label"])
            .size()
        )
        sc: dict = {}
        for (split, source, label), n in tab.items():
            sc.setdefault(split, {}).setdefault(source, {})["attack" if label else "normal"] = int(n)
        stats["scenarios"][name] = sc
    return stats


def load_scenario(
    name: str,
    processed_dir: str | Path = "data/processed",
    augment: bool = False,
    seed: int = 42,
    **augment_kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Devuelve (train, val, test). Si augment=True, se aumenta SÓLO train."""
    if name not in SCENARIOS:
        raise KeyError(f"escenario desconocido: {name}. Opciones: {SCENARIOS}")
    processed_dir = Path(processed_dir)
    df = pd.read_parquet(processed_dir / "http_requests.parquet")
    splits = pd.read_parquet(processed_dir / "splits.parquet")
    if len(splits) != len(df):
        raise ValueError("splits.parquet no corresponde a http_requests.parquet: vuelve a correr splits")
    df["group_id"] = splits["group_id"].to_numpy()
    col = splits[name].to_numpy()
    train = df[col == TRAIN].reset_index(drop=True)
    val = df[col == VAL].reset_index(drop=True)
    test = df[col == TEST].reset_index(drop=True)
    for part in (train, val, test):
        part["augmented"] = False
        part["aug"] = ""
    if augment:
        from .augment import augment_train

        train = augment_train(train, seed=seed, **augment_kwargs)
    return train, val, test


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    df = pd.read_parquet(args.processed_dir / "http_requests.parquet")
    splits = make_splits(df, seed=args.seed)
    splits.reset_index(drop=True).to_parquet(args.processed_dir / "splits.parquet", index=False)
    stats = summarize(df, splits)
    stats["seed"] = args.seed
    (args.processed_dir / "splits_stats.json").write_text(json.dumps(stats, indent=2))

    print(f"{stats['rows']:,} filas en {stats['groups']:,} grupos de casi-duplicados")
    for name, sc in stats["scenarios"].items():
        parts = []
        for split in (TRAIN, VAL, TEST, LEAK):
            if split in sc:
                n = sum(sum(v.values()) for v in sc[split].values())
                a = sum(v.get("attack", 0) for v in sc[split].values())
                parts.append(f"{split}={n:,} ({a:,} atq)")
        print(f"  {name:18s} " + "  ".join(parts))


if __name__ == "__main__":
    main()
