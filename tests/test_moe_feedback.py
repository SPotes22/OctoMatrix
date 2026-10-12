import numpy as np
import pandas as pd

from octomatrix import feedback
from octomatrix.data.adversarial import ATTACKS, AdversarialGenerator, split_payloads
from octomatrix.moe import Expert, ScoreUnionEnsemble
from octomatrix.pipeline import OctoMatrixModel


# --- split de payloads (train/holdout disjuntos) ---
def test_split_payloads_disjoint_and_covers():
    for payloads in ATTACKS.values():
        tr = split_payloads(payloads, "train")
        ho = split_payloads(payloads, "holdout")
        assert set(tr).isdisjoint(ho)
        assert set(tr) | set(ho) == set(payloads)
        assert tr  # train nunca vacío


def test_generator_train_holdout_use_different_payloads():
    tr = AdversarialGenerator(seed=1).build(per_category=30, payload_split="train")
    ho = AdversarialGenerator(seed=1).build(per_category=30, payload_split="holdout")
    # payloads de ataque de train y holdout no se solapan (para sqli, que tiene varios)
    def sqli_vals(df):
        d = df[df.attack_type == "sqli"]
        return set((d["query"] + d.body).str.replace("^[^=]*=", "", regex=True))
    assert sqli_vals(tr).isdisjoint(sqli_vals(ho))


# --- ensemble OR ---
class _FakeModel:
    def __init__(self, scores, thr):
        self._scores = np.asarray(scores, float)
        from octomatrix.pipeline import Threshold
        self.threshold = Threshold(value=thr)

    def scores(self, texts):
        return self._scores[: len(list(texts))]


def _ensemble(scores_a, thr_a, scores_b, thr_b):
    return ScoreUnionEnsemble([
        Expert("a", _FakeModel(scores_a, thr_a)),
        Expert("b", _FakeModel(scores_b, thr_b)),
    ])


def test_ensemble_or_and_attributions():
    ens = _ensemble([0.9, 0.1, 0.1], 0.5, [0.1, 0.9, 0.1], 0.5)
    texts = ["x", "y", "z"]
    assert list(ens.predict(texts)) == [1, 1, 0]
    attr = ens.attributions(texts)
    assert attr[0] == ["a"] and attr[1] == ["b"] and attr[2] == []


def test_ensemble_margin_is_max_exceedance():
    ens = _ensemble([0.6, 0.2], 0.5, [0.3, 0.45], 0.5)
    m = ens.margin(["x", "y"])
    assert np.isclose(m[0], 0.1)     # a: 0.6-0.5
    assert np.isclose(m[1], -0.05)   # ninguno alerta; el más alto (b) quedó a -0.05


def _toy_attacks(n):
    t, y = [], []
    for i in range(n):
        if i % 2:
            t.append(f"GET /x?q=' OR {i}=1 UNION SELECT a FROM b--"); y.append(1)
        else:
            t.append(f"GET /api/items/{i}"); y.append(0)
    return t, np.array(y)


def test_ensemble_save_load_roundtrip(tmp_path):
    t, y = _toy_attacks(80)
    m = OctoMatrixModel.fit(t, y, calibrate=False, n_estimators=25, max_features=800)
    m.set_threshold_for_fpr(t, y, 0.1)
    ens = ScoreUnionEnsemble([Expert("real", m)])
    ens.save(tmp_path / "moe")
    loaded = ScoreUnionEnsemble.load(tmp_path / "moe")
    assert loaded.names == ["real"]
    assert np.array_equal(ens.predict(t), loaded.predict(t))


# --- feedback / post-test ---
def test_select_for_review_prefers_small_margin_and_excludes_seen():
    preds = [
        {"request_hash": "a", "margin": 0.02, "text": "t1"},
        {"request_hash": "b", "margin": 0.5, "text": "t2"},   # fuera de banda
        {"request_hash": "c", "margin": -0.03, "text": "t3"},
        {"request_hash": "a", "margin": 0.02, "text": "t1"},   # duplicado
    ]
    out = feedback.select_for_review(preds, band=0.1, max_items=10, exclude_hashes={"c"})
    assert [p["request_hash"] for p in out] == ["a"]  # b fuera de banda, c excluido, a una vez
    assert out[0]["label"] == "" and out[0]["reviewer"] == ""


def test_feedback_full_cycle(tmp_path):
    t, y = _toy_attacks(40)
    m = OctoMatrixModel.fit(t, y, calibrate=False, n_estimators=20, max_features=500)
    m.set_threshold_for_fpr(t, y, 0.1)
    ens = ScoreUnionEnsemble([Expert("real", m)])

    pred_log = tmp_path / "pred.jsonl"
    assert feedback.log_predictions(ens, t, pred_log) == len(t)
    preds = feedback.read_jsonl(pred_log)
    assert preds and "margin" in preds[0] and "fired_experts" in preds[0]

    queue = tmp_path / "queue.jsonl"
    items = feedback.select_for_review(preds, band=1.0, max_items=5)
    feedback.write_review_queue(items, queue)

    # una persona etiqueta las dos primeras, deja el resto en blanco
    rows = feedback.read_jsonl(queue)
    rows[0]["label"], rows[0]["reviewer"] = "1", "ana"
    rows[1]["label"], rows[1]["reviewer"] = "0", "ana"
    queue.write_text("\n".join(__import__("json").dumps(r) for r in rows))
    store = tmp_path / "store.jsonl"
    stats = feedback.ingest_reviewed(queue, store)
    assert stats["added_to_store"] == 2
    assert stats["skipped_unlabeled"] == len(rows) - 2
    # idempotente: reingesta no duplica
    assert feedback.ingest_reviewed(queue, store)["added_to_store"] == 0
