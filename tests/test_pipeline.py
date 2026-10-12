import numpy as np
import pytest

from octomatrix.features import FEATURE_NAMES, SecurityFeatures, extract_one, shannon_entropy
from octomatrix.pipeline import OctoMatrixModel
from octomatrix.postfilter import RegexPostFilter


def test_shannon_entropy():
    assert shannon_entropy("") == 0.0
    assert shannon_entropy("aaaa") == 0.0
    assert shannon_entropy("ab") == pytest.approx(1.0)
    assert shannon_entropy("abcd") == pytest.approx(2.0)


def test_security_features_shape_and_signals():
    X = SecurityFeatures().fit_transform(["GET /a", "GET /x?q=' OR 1=1 UNION SELECT * FROM users--"])
    assert X.shape == (2, len(FEATURE_NAMES))
    f = dict(zip(FEATURE_NAMES, X[1]))
    assert f["re_sql_keywords"] >= 2  # UNION, SELECT
    assert f["re_sql_tautology"] >= 1
    assert f["re_sql_comment"] >= 1
    assert dict(zip(FEATURE_NAMES, X[0]))["re_sql_keywords"] == 0


def test_security_features_ratios_and_maxrun():
    f = dict(zip(FEATURE_NAMES, extract_one("AAAAbb12 ")))
    assert f["length"] == 9
    assert f["max_run"] == 4
    assert f["digit_ratio"] == pytest.approx(2 / 9)
    assert f["whitespace_ratio"] == pytest.approx(1 / 9)


def test_postfilter_rescues_static_but_not_when_attack_present():
    pf = RegexPostFilter()
    assert pf.is_allowlisted("GET /static/js/app.js")
    assert pf.is_allowlisted("GET /favicon.ico")
    assert not pf.is_allowlisted("GET /static/app.js?x=<script>alert(1)</script>")
    assert not pf.is_allowlisted("GET /api/users?id=1 OR 1=1")
    scores = pf.apply(["GET /static/a.css", "GET /x?q=' OR 1=1"], [0.9, 0.9])
    assert scores[0] == 0.0 and scores[1] == 0.9


def _toy_data(n=120):
    rng = np.random.default_rng(0)
    texts, labels = [], []
    for i in range(n):
        if i % 2:
            texts.append(f"GET /search?q=' OR {i}={i} UNION SELECT pass FROM users--")
            labels.append(1)
        else:
            texts.append(f"GET /api/items/{i}?page={i % 5}")
            labels.append(0)
    return texts, np.array(labels), rng


def test_model_fit_predict_threshold_confidence():
    texts, labels, _ = _toy_data()
    model = OctoMatrixModel.fit(texts, labels, calibrate=False, n_estimators=40, max_features=2000)
    scores = model.scores(texts)
    assert scores.shape == (len(texts),)
    assert scores.min() >= 0 and scores.max() <= 1
    thr = model.set_threshold_for_fpr(texts, labels, target_fpr=0.05)
    assert thr.measured_fpr <= 0.1
    conf = model.confidence(texts)
    assert (conf >= 0.5).all()  # confianza siempre del lado de la decisión
    # separa bien en datos de juguete
    assert model.scores(["GET /x?q=' OR 1=1 UNION SELECT a FROM b--"])[0] > model.scores(["GET /api/items/3"])[0]


def test_model_save_load_roundtrip_and_hash(tmp_path):
    texts, labels, _ = _toy_data()
    model = OctoMatrixModel.fit(texts, labels, calibrate=False, n_estimators=30, max_features=1500)
    model.set_threshold_for_fpr(texts, labels, 0.05)
    path = model.save(tmp_path / "m.joblib")
    assert path.with_suffix(".joblib.sha256").exists()
    loaded = OctoMatrixModel.load(path)
    assert np.allclose(model.scores(texts), loaded.scores(texts))
    assert loaded.threshold.value == model.threshold.value
    # artefacto alterado -> falla la verificación
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="sha256"):
        OctoMatrixModel.load(path)


def test_calibration_runs_small():
    texts, labels, _ = _toy_data(80)
    model = OctoMatrixModel.fit(texts, labels, calibrate=True, cv=2, n_estimators=20, max_features=1000)
    assert model.metadata["calibrated"] is True
    assert model.scores(texts).shape == (80,)
