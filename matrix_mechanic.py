# Copyright (c) 2025 Santiago Potes Giraldo - BlackByte
# Licencia de Paz Digital v1.0 (LPD). Ver LICENSE.
"""
MECHANIC: matriz de confusión + análisis de errores de un artefacto OctoMatrix.

Reescrito en la Fase 0: antes cargaba el .pkl viejo (modelo + vectorizer por
separado) y se caía con "X has 407 features, but RF is expecting 415" porque
reconstruía las features a mano y no cuadraban. Ahora carga el artefacto único
(octomatrix.pipeline.OctoMatrixModel), que lleva TODO el pre-procesado dentro,
así que no hay forma de que las features se desajusten.

    python matrix_mechanic.py --model models/octomatrix.joblib --scenario in_source
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from octomatrix.data.splits import SCENARIOS, load_scenario
from octomatrix.data.text import request_text
from octomatrix.pipeline import OctoMatrixModel


def _texts(df):
    return [request_text(*r) for r in zip(df.method, df.path, df["query"], df.body, df.headers)]


class MatrixMechanic:
    def __init__(self, model_path: str, scenario: str = "in_source", processed_dir: str = "data/processed"):
        self.model = OctoMatrixModel.load(model_path)
        _, _, self.test = load_scenario(scenario, processed_dir)
        self.texts = _texts(self.test)
        self.y_true = self.test.label.to_numpy()
        self.scores = self.model.scores(self.texts)
        self.y_pred = (self.scores >= self.model.threshold.value).astype(int)
        self.confusion_data: dict = {}

    def generate_confusion_matrix(self) -> dict:
        cm = confusion_matrix(self.y_true, self.y_pred, labels=[0, 1])
        report = classification_report(
            self.y_true, self.y_pred, labels=[0, 1], target_names=["Normal", "Ataque"], output_dict=True
        )
        self.confusion_data = {
            "matrix": cm.tolist(),
            "true_negatives": int(cm[0, 0]), "false_positives": int(cm[0, 1]),
            "false_negatives": int(cm[1, 0]), "true_positives": int(cm[1, 1]),
            "accuracy": report["accuracy"],
            "precision_malicious": report["Ataque"]["precision"],
            "recall_malicious": report["Ataque"]["recall"],
            "threshold": self.model.threshold.value,
            "timestamp": datetime.now().isoformat(),
        }
        return self.confusion_data

    def visualize_confusion(self, save_path: str = "output/confusion_matrix.png") -> str | None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            print("⚠️  matplotlib/seaborn no instalados; se omite el PNG")
            return None
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        cm = np.array(self.confusion_data["matrix"])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Normal", "Ataque"], yticklabels=["Normal", "Ataque"])
        plt.title("Matriz de Confusión - OctoMatrix")
        plt.xlabel("Predicción"); plt.ylabel("Real")
        plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
        print(f"📊 Matriz guardada en: {save_path}")
        return save_path

    def analyze_errors(self) -> dict:
        errors = {"false_positives": [], "false_negatives": []}
        for i in np.where((self.y_true == 0) & (self.y_pred == 1))[0]:
            errors["false_positives"].append(self._err_row(i))
        for i in np.where((self.y_true == 1) & (self.y_pred == 0))[0]:
            errors["false_negatives"].append(self._err_row(i))
        print(f"🔍 FP={len(errors['false_positives'])}  FN={len(errors['false_negatives'])}")
        return errors

    def _err_row(self, i: int) -> dict:
        r = self.test.iloc[i]
        return {
            "source": r.source, "attack_type": r.attack_type,
            "score": float(self.scores[i]), "sample_id": r.sample_id,
        }

    def export(self, output_path: str = "output/mechanic_report.json") -> str:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        report = {
            "model_metadata": self.model.metadata,
            "confusion": self.confusion_data,
            "errors": self.analyze_errors(),
        }
        Path(output_path).write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"📦 Reporte: {output_path}")
        return output_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/octomatrix.joblib")
    ap.add_argument("--scenario", choices=SCENARIOS, default="in_source")
    ap.add_argument("--processed-dir", default="data/processed")
    args = ap.parse_args()

    mech = MatrixMechanic(args.model, args.scenario, args.processed_dir)
    cm = mech.generate_confusion_matrix()
    mech.visualize_confusion()
    mech.export()
    print("\n📈 RESUMEN MECHANIC")
    print(f"   Accuracy:          {cm['accuracy']:.3f}")
    print(f"   Precisión ataques: {cm['precision_malicious']:.3f}")
    print(f"   Recall ataques:    {cm['recall_malicious']:.3f}")
    print(f"   FP={cm['false_positives']}  FN={cm['false_negatives']}")


if __name__ == "__main__":
    main()
