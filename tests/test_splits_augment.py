import numpy as np
import pandas as pd
import pytest

from octomatrix.data import augment, splits
from octomatrix.data.schema import validate
from octomatrix.data.text import group_id, skeleton


def _row(**kw):
    base = dict(method="GET", path="/p", query="", body="", headers="", label=0, attack_type="normal",
                source="csic2010", origin_split="-", sample_id="")
    base.update(kw)
    return base


def _toy():
    rows = []
    for src in ("csic2010", "ecml_pkdd2007", "crs_regression", "crs_regression_negative"):
        is_crs = src == "crs_regression"
        for i in range(60):
            attack = is_crs or (src not in ("crs_regression_negative",) and i % 3 == 0)
            rows.append(_row(
                source=src,
                path=f"/{src}/{chr(97 + i % 20)}",  # 20 plantillas por fuente
                query=f"id={i}" + ("&q=%27%20or%201=1--" if attack else ""),  # id distinto, mismo grupo
                label=int(attack),
                attack_type="sqli" if attack else "normal",
                sample_id=str(i),
            ))
    return validate(pd.DataFrame(rows))


def test_skeleton_groups_near_duplicates_but_keeps_double_encoding_apart():
    assert skeleton("GET", "/a", "id=1&q=%27%20OR", "") == skeleton("GET", "/a", "id=27&q=' or", "")
    assert group_id("GET", "/a", "B1=Confirmar%2B", "") != group_id("GET", "/a", "B1=Confirmar%252B", "")


def test_group_split_never_splits_a_group_and_respects_fractions():
    df = splits.add_groups(_toy())
    col = splits.group_split(df, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=0)
    assert (df.assign(s=col).groupby("group_id")["s"].nunique() == 1).all()
    frac = col.value_counts(normalize=True)
    assert abs(frac["train"] - 0.7) < 0.15


def test_make_splits_scenarios_have_no_group_leak():
    df = _toy()
    sp = splits.make_splits(df, seed=1)
    assert set(splits.SCENARIOS) <= set(sp.columns)
    # csic_to_crs: train/val sólo CSIC, test sólo CRS
    used = df.assign(s=sp["csic_to_crs"])
    assert set(used[used.s.isin(["train", "val"])].source) == {"csic2010"}
    assert set(used[used.s == "test"].source) <= {"crs_regression", "crs_regression_negative"}
    assert set(used[used.source == "ecml_pkdd2007"].s) == {""}


def test_cross_source_leak_is_excluded():
    df = validate(pd.DataFrame([
        _row(source="csic2010", query="x=1", sample_id="a"),
        _row(source="csic2010", query="x=2", sample_id="b"),
        _row(source="crs_regression_negative", query="x=3", sample_id="c"),  # mismo grupo que a/b
        _row(source="crs_regression_negative", query="y=3", sample_id="d"),
    ]))
    sp = splits.make_splits(df, seed=0)
    assert list(sp["csic_to_crs"].iloc[2:]) == [splits.LEAK, splits.TEST]


def test_check_no_group_leak_detects_leak():
    sp = pd.DataFrame({"group_id": ["g", "g"], **{s: ["train", "test"] for s in splits.SCENARIOS}})
    with pytest.raises(AssertionError):
        splits.check_no_group_leak(sp)


def test_augment_transforms():
    rng = np.random.default_rng(0)
    assert augment.TRANSFORMS["url_encode"]("' or 1=1", rng) == "%27%20or%201%3D1"
    assert augment.TRANSFORMS["double_encode"]("'", rng) == "%2527"
    assert augment.TRANSFORMS["url_encode"]("Jam%F3n", rng) == "Jam%F3n"           # bytes latin-1 intactos
    assert augment.TRANSFORMS["url_encode"]("<ſcript>", rng) == "%3C%C5%BFcript%3E"  # fuera de latin-1 -> utf-8
    assert augment.TRANSFORMS["sql_comment"]("1 union select", rng) == "1/**/union/**/select"
    assert augment._map_values("a=1 2&b=&c", lambda v: v.upper()) == "a=1 2&b=&c"
    assert augment._map_values("a=x y&b=z", lambda v: v.upper()) == "a=X Y&b=Z"


def test_applicable_rules():
    csic_normal = _row()
    assert augment.applicable(csic_normal) == ["url_encode"]
    ecml_normal = _row(source="ecml_pkdd2007")
    assert "case_swap" not in augment.applicable(ecml_normal)
    sqli = _row(label=1, attack_type="sqli", query="id=1 union select")
    assert {"sql_comment", "sql_whitespace", "case_swap"} <= set(augment.applicable(sqli))
    xss = _row(label=1, attack_type="xss", query="q=<script>")
    assert "sql_comment" not in augment.applicable(xss)


def test_augment_train_keeps_labels_groups_and_originals():
    df = splits.add_groups(_toy())
    out = augment.augment_train(df, frac=1.0, n_variants=2, seed=0)
    orig = out[~out.augmented]
    aug = out[out.augmented]
    assert len(orig) == len(df) and len(aug) > 0
    parents = aug.sample_id.str.split("#").str[0]
    merged = aug.assign(parent=parents).merge(
        df[["source", "sample_id", "label", "group_id"]], left_on=["source", "parent"],
        right_on=["source", "sample_id"], suffixes=("", "_p"))
    assert len(merged) == len(aug)
    assert (merged.label == merged.label_p).all()
    assert (merged.group_id == merged.group_id_p).all()
    assert not out.duplicated(["method", "path", "query", "body", "headers"]).any()
    with pytest.raises(ValueError):
        augment.augment_train(out)


def test_load_scenario_only_augments_train(tmp_path):
    df = _toy()
    df.to_parquet(tmp_path / "http_requests.parquet")
    splits.make_splits(df, seed=0).reset_index(drop=True).to_parquet(tmp_path / "splits.parquet")
    train, val, test = splits.load_scenario("in_source", tmp_path, augment=True, frac=1.0)
    assert train.augmented.any()
    assert not val.augmented.any() and not test.augmented.any()
    assert not set(train.group_id) & set(test.group_id)
    with pytest.raises(KeyError):
        splits.load_scenario("nope", tmp_path)
