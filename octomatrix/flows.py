"""Experto de FLUJOS (CIC-IDS2018): el tercer experto del MoE.

NO es texto. Cada muestra es un flujo de red descrito por ~78 estadísticas de
CICFlowMeter (duración, paquetes, bytes, flags, tiempos entre llegadas). Por
eso es un modelo tabular aparte, no el pipeline de char n-grams.

Su razón de ser en el MoE: detecta ataques de RED que el experto de texto HTTP
no puede ver por definición — DoS/DDoS, fuerza bruta (FTP/SSH), botnet,
infiltración. Son invisibles al payload de una sola petición; sólo se ven en la
forma del tráfico.

Artefacto análogo a OctoMatrixModel: RF calibrado + umbral por FPR, guardado con
joblib + sha256. La diferencia es la entrada (matriz de features, no texto) y
que lleva dentro el imputador y el escalado.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .pipeline import Threshold  # reutiliza la misma dataclass de umbral

ARTIFACT_VERSION = "1.0.0"
# Columnas que NO son features (metadatos añadidos por el loader).
NON_FEATURES = {"timestamp", "attack_type", "label", "source", "day_file"}


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURES]


def build_flow_pipeline(n_estimators: int = 150, max_depth: int | None = 30, random_state: int = 42) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),   # los inf/NaN del loader
        ("forest", RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state,
            n_jobs=-1, class_weight="balanced_subsample")),
    ])


class FlowModel:
    def __init__(self, pipeline: Pipeline, features: list[str]):
        self.pipeline = pipeline
        self.features = features
        self.threshold = Threshold()
        self.metadata: dict = {}
        self._set_inference_njobs(1)

    def _set_inference_njobs(self, n: int) -> None:
        est = self.pipeline
        if hasattr(est, "calibrated_classifiers_"):
            for cc in est.calibrated_classifiers_:
                f = getattr(cc.estimator, "named_steps", {}).get("forest")
                if f is not None:
                    f.n_jobs = n
        else:
            f = getattr(est, "named_steps", {}).get("forest")
            if f is not None:
                f.n_jobs = n

    def _matrix(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.features if c not in df.columns]
        if missing:
            raise ValueError(f"faltan columnas de features: {missing[:5]}...")
        return df[self.features].to_numpy(dtype=np.float32)

    @classmethod
    def fit(cls, df: pd.DataFrame, *, calibrate: bool = True, cv: int = 3, **kw) -> "FlowModel":
        features = feature_columns(df)
        X = df[features].to_numpy(dtype=np.float32)
        y = df["label"].to_numpy()
        base = build_flow_pipeline(**kw)
        if calibrate:
            est = CalibratedClassifierCV(base, method="isotonic", cv=cv)
            est.fit(X, y)
            pipeline = est
        else:
            base.fit(X, y)
            pipeline = base
        model = cls(pipeline, features)
        model.metadata = {
            "version": ARTIFACT_VERSION, "expert": "flows",
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "n_train": int(len(y)), "n_attacks": int(y.sum()),
            "n_features": len(features), "calibrated": calibrate,
        }
        return model

    def scores(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(self._matrix(df))[:, 1]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return (self.scores(df) >= self.threshold.value).astype(int)

    def set_threshold_for_fpr(self, df_val: pd.DataFrame, target_fpr: float) -> Threshold:
        y = df_val["label"].to_numpy()
        s = self.scores(df_val)
        neg = np.sort(s[y == 0])
        thr = float(np.nextafter(neg[min(len(neg) - 1, int(np.ceil((1 - target_fpr) * len(neg))))], 1.0)) if len(neg) else 0.5
        pred = s >= thr
        self.threshold = Threshold(
            value=thr, target_fpr=target_fpr,
            measured_fpr=float(pred[y == 0].mean()) if (y == 0).any() else None,
            measured_recall=float(pred[y == 1].mean()) if (y == 1).any() else None,
        )
        return self.threshold

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "features": self.features,
                     "threshold": asdict(self.threshold), "metadata": self.metadata}, path, compress=3)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path, verify: bool = True) -> "FlowModel":
        path = Path(path)
        if verify:
            sig = path.with_suffix(path.suffix + ".sha256")
            if sig.exists() and sig.read_text().split()[0] != hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError(f"sha256 no coincide para {path}: artefacto alterado")
        payload = joblib.load(path)
        model = cls(payload["pipeline"], payload["features"])
        model.threshold = Threshold(**payload["threshold"])
        model.metadata = payload["metadata"]
        return model
