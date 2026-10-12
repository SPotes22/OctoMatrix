"""El detector OctoMatrix como UN solo artefacto (Fase 0 + Fase 3).

Arquitectura (la que describe el README, implementada de verdad):

    petición ──► FeatureUnion
                 ├─ TfidfVectorizer(char n-grams)   contexto de tokens
                 └─ SecurityFeatures(regex+entropía) toxicidad estructural
                        │
                        ▼
                 RandomForest  ──►  CalibratedClassifierCV (Fase 3)
                        │                 probabilidad calibrada
                        ▼
                 RegexPostFilter (allowlist)  ──►  % de confianza
                        │
                        ▼
                 umbral elegido para un FPR objetivo

Todo el pre-procesado vive en un único sklearn.Pipeline, así que no hay
desajuste de features entre scripts (el bug que tenía matrix_mechanic).
El artefacto se guarda con joblib + un sha256 para integridad, no con pickle
suelto.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline

from .features import SecurityFeatures
from .postfilter import RegexPostFilter

ARTIFACT_VERSION = "2.0.0"


def build_pipeline(
    max_features: int = 3_000,
    ngram_range: tuple[int, int] = (1, 3),
    n_estimators: int = 150,
    max_depth: int | None = 40,
    random_state: int = 42,
) -> Pipeline:
    """El Pipeline SIN calibrar (RF crudo). La calibración la añade OctoMatrixModel.fit."""
    features = FeatureUnion(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=ngram_range,
                    max_features=max_features,
                    sublinear_tf=True,
                    lowercase=False,
                    dtype=np.float32,
                ),
            ),
            ("security", SecurityFeatures()),
        ]
    )
    forest = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    return Pipeline([("features", features), ("forest", forest)])


@dataclass
class Threshold:
    value: float = 0.5
    target_fpr: float | None = None
    measured_fpr: float | None = None
    measured_recall: float | None = None


class OctoMatrixModel:
    """Envuelve el Pipeline calibrado + post-filtro + umbral. Un solo artefacto."""

    def __init__(self, pipeline: Pipeline, postfilter: RegexPostFilter | None = None):
        self.pipeline = pipeline
        self.postfilter = postfilter or RegexPostFilter()
        self.threshold = Threshold()
        self.metadata: dict = {}

    # --- entrenamiento ---
    @classmethod
    def fit(
        cls,
        texts,
        labels,
        *,
        calibrate: bool = True,
        calibration_method: str = "isotonic",
        cv: int = 3,
        postfilter: RegexPostFilter | None = None,
        **pipeline_kwargs,
    ) -> "OctoMatrixModel":
        base = build_pipeline(**pipeline_kwargs)
        if calibrate:
            # CalibratedClassifierCV envuelve el Pipeline completo: calibra sobre
            # folds de train, sin tocar val/test.
            estimator = CalibratedClassifierCV(base, method=calibration_method, cv=cv)
            estimator.fit(texts, labels)
            pipeline = estimator
        else:
            base.fit(texts, labels)
            pipeline = base
        model = cls(pipeline, postfilter)
        model.metadata = {
            "version": ARTIFACT_VERSION,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "n_train": int(len(labels)),
            "n_attacks": int(np.sum(labels)),
            "calibrated": calibrate,
            "calibration_method": calibration_method if calibrate else None,
            "pipeline_kwargs": pipeline_kwargs,
        }
        return model

    # --- inferencia ---
    def scores(self, texts) -> np.ndarray:
        """Probabilidad de ataque (0..1) tras calibración y post-filtro."""
        texts = list(texts)
        proba = self.pipeline.predict_proba(texts)[:, 1]
        return self.postfilter.apply(texts, proba)

    def predict(self, texts) -> np.ndarray:
        return (self.scores(texts) >= self.threshold.value).astype(int)

    def confidence(self, texts) -> np.ndarray:
        """% de confianza en la decisión tomada (no la prob. cruda de ataque)."""
        s = self.scores(texts)
        return np.where(s >= self.threshold.value, s, 1 - s)

    # --- umbral ---
    def set_threshold_for_fpr(self, val_texts, val_labels, target_fpr: float) -> Threshold:
        """Fija el umbral en validación para no superar `target_fpr` de falsos positivos."""
        val_labels = np.asarray(val_labels)
        s = self.scores(val_texts)
        neg = np.sort(s[val_labels == 0])
        if len(neg):
            k = min(len(neg) - 1, int(np.ceil((1 - target_fpr) * len(neg))))
            thr = float(np.nextafter(neg[k], 1.0))
        else:
            thr = 0.5
        pred = s >= thr
        self.threshold = Threshold(
            value=thr,
            target_fpr=target_fpr,
            measured_fpr=float(pred[val_labels == 0].mean()) if (val_labels == 0).any() else None,
            measured_recall=float(pred[val_labels == 1].mean()) if (val_labels == 1).any() else None,
        )
        return self.threshold

    # --- persistencia (un solo archivo) ---
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pipeline": self.pipeline,
            "postfilter_allow": self.postfilter.allow,
            "postfilter_rescue_to": self.postfilter.rescue_to,
            "threshold": asdict(self.threshold),
            "metadata": self.metadata,
        }
        joblib.dump(payload, path, compress=3)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n")
        (path.parent / (path.stem + "_meta.json")).write_text(
            json.dumps({**self.metadata, "threshold": asdict(self.threshold), "sha256": digest}, indent=2)
        )
        return path

    @classmethod
    def load(cls, path: str | Path, verify: bool = True) -> "OctoMatrixModel":
        path = Path(path)
        if verify:
            sig = path.with_suffix(path.suffix + ".sha256")
            if sig.exists():
                expected = sig.read_text().split()[0]
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                if expected != actual:
                    raise ValueError(f"sha256 no coincide para {path}: artefacto alterado")
        payload = joblib.load(path)
        model = cls(
            payload["pipeline"],
            RegexPostFilter(allow=payload["postfilter_allow"], rescue_to=payload["postfilter_rescue_to"]),
        )
        model.threshold = Threshold(**payload["threshold"])
        model.metadata = payload["metadata"]
        return model
