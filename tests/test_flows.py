import numpy as np
import pandas as pd
import pytest

from octomatrix.flows import FlowModel, NON_FEATURES, feature_columns


def _toy_flows(n=400, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        attack = i % 2
        # ataque: flujos cortos y muy repetitivos (tipo DoS/bruteforce);
        # normal: flujos variados.
        if attack:
            dur = rng.normal(50, 5); pkts = rng.normal(2, 0.5); byts = rng.normal(80, 10)
        else:
            dur = rng.normal(5000, 1500); pkts = rng.normal(40, 15); byts = rng.normal(4000, 1500)
        rows.append({
            "Flow Duration": dur, "Tot Fwd Pkts": pkts, "TotLen Fwd Pkts": byts,
            "Flow Byts/s": byts / max(dur, 1) * 1000,
            "timestamp": pd.Timestamp("2018-02-14") + pd.Timedelta(seconds=i),
            "attack_type": "dos_hulk" if attack else "normal",
            "label": attack, "source": "cicids2018", "day_file": "d.csv",
        })
    return pd.DataFrame(rows)


def test_feature_columns_excludes_metadata():
    df = _toy_flows(10)
    feats = feature_columns(df)
    assert set(feats) == {"Flow Duration", "Tot Fwd Pkts", "TotLen Fwd Pkts", "Flow Byts/s"}
    assert not (set(feats) & NON_FEATURES)


def test_flowmodel_fit_predict_threshold():
    df = _toy_flows(400)
    tr, te = df.iloc[:300], df.iloc[300:]
    m = FlowModel.fit(tr, calibrate=False, n_estimators=40)
    s = m.scores(te)
    assert s.shape == (len(te),) and s.min() >= 0 and s.max() <= 1
    thr = m.set_threshold_for_fpr(tr, 0.05)
    assert thr.measured_fpr <= 0.1
    # separa ataque de normal en los datos de juguete
    assert m.scores(te[te.label == 1]).mean() > m.scores(te[te.label == 0]).mean()


def test_flowmodel_handles_nan_via_imputer():
    df = _toy_flows(200)
    df.loc[df.index[:10], "Flow Byts/s"] = np.nan  # el loader deja NaN en inf
    m = FlowModel.fit(df, calibrate=False, n_estimators=20)
    assert np.isfinite(m.scores(df)).all()


def test_flowmodel_missing_feature_raises():
    df = _toy_flows(120)
    m = FlowModel.fit(df, calibrate=False, n_estimators=20)
    with pytest.raises(ValueError, match="faltan columnas"):
        m.scores(df.drop(columns=["Flow Duration"]))


def test_flowmodel_save_load_roundtrip_and_hash(tmp_path):
    df = _toy_flows(200)
    m = FlowModel.fit(df, calibrate=False, n_estimators=25)
    m.set_threshold_for_fpr(df, 0.05)
    path = m.save(tmp_path / "f.joblib")
    assert path.with_suffix(".joblib.sha256").exists()
    loaded = FlowModel.load(path)
    assert loaded.features == m.features
    assert np.allclose(m.scores(df), loaded.scores(df))
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="sha256"):
        FlowModel.load(path)
