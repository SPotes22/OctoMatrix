import numpy as np
import pandas as pd

from octomatrix.data.adversarial import ATTACKS, AdversarialGenerator
from octomatrix.data.schema import ATTACK_TYPES, validate
from octomatrix.owasp import ATTACK_TYPE_TO_OWASP, OWASP_2021, owasp_category, owasp_label
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "waf_report", Path(__file__).resolve().parents[1] / "scripts" / "waf_report.py"
)
waf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(waf)


def test_every_attack_type_maps_to_a_known_owasp_bucket():
    for at in ATTACK_TYPES:
        cat = owasp_category(at)
        if at == "normal":
            assert cat is None
        else:
            assert cat in OWASP_2021, f"{at} -> {cat} no está en OWASP_2021"
    # y al revés: el mapa no tiene tipos fantasma
    assert set(ATTACK_TYPE_TO_OWASP) == ATTACK_TYPES


def test_owasp_label_format():
    assert owasp_label("sqli").startswith("A03")
    assert owasp_label("ssrf").startswith("A10")
    assert owasp_label("normal") is None


def test_adversarial_generator_is_deterministic_and_valid():
    a = AdversarialGenerator(seed=7).build(per_category=20)
    b = AdversarialGenerator(seed=7).build(per_category=20)
    validate(a)  # cumple el esquema
    assert a.equals(b)  # reproducible con misma semilla
    assert set(a[a.label == 1].attack_type) == set(ATTACKS)
    # balanceado y con payload presente en query o body
    assert a.label.sum() == (a.label == 0).sum()
    atk = a[a.label == 1].iloc[0]
    assert atk.query or atk.body


def test_adversarial_covers_the_requested_gaps():
    for needed in ["rce", "ssrf", "xxe", "buffer_overflow", "privilege_escalation", "crlf", "deserialization"]:
        assert needed in ATTACKS


def test_recall_at_fpr_respects_budget():
    y = np.array([0] * 100 + [1] * 100)
    scores = np.concatenate([np.linspace(0, 0.6, 100), np.linspace(0.4, 1.0, 100)])
    recall, thr = waf.recall_at_fpr(y, scores, 0.01)
    pred = scores >= thr
    assert pred[y == 0].mean() <= 0.02
    assert 0 <= recall <= 1


def test_group_recall_counts():
    df = pd.DataFrame({
        "label": [1, 1, 1, 0],
        "attack_type": ["sqli", "sqli", "xss", "normal"],
    })
    scores = np.array([0.9, 0.2, 0.9, 0.1])
    out = waf.group_recall(df, scores, thr=0.5, key="attack_type")
    assert out["sqli"] == {"n": 2, "recall": 0.5}
    assert out["xss"] == {"n": 1, "recall": 1.0}
    assert "normal" not in out  # sólo ataques
