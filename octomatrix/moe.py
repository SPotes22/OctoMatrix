"""Mixture-of-Experts por procedencia de datos (combinador OR, Fase 5).

Cada experto es un OctoMatrixModel independiente, entrenado y calibrado sobre
SU propia procedencia de datos, con su propio umbral:

- experto "real":     CSIC + ECML + CRS (datos reales).
- experto "generado": payloads generados de categorías OWASP sin ejemplos
                       públicos (SSRF, XXE, RCE, CRLF, privesc...).
- [futuro] experto "flujos": CIC-IDS2018 (va por otro camino, no es texto).

Combinador: OR por score. La petición es ataque si CUALQUIER experto supera su
propio umbral. Esto mantiene el aislamiento que evita el data poisoning: el
experto generado NO reentrena ni altera al real; sólo puede AÑADIR detecciones.
El coste es que su FPR se suma al del real, y por eso se mide y se controla.

No hay meta-modelo entrenado sobre ambos: eso reacoplaría los expertos y es
justo lo que se quiere evitar.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .pipeline import OctoMatrixModel


@dataclass
class Expert:
    name: str
    model: OctoMatrixModel

    def fired(self, texts) -> np.ndarray:
        return self.model.scores(texts) >= self.model.threshold.value


class ScoreUnionEnsemble:
    """OR por score sobre expertos independientes."""

    def __init__(self, experts: list[Expert]):
        if not experts:
            raise ValueError("se necesita al menos un experto")
        self.experts = experts

    @property
    def names(self) -> list[str]:
        return [e.name for e in self.experts]

    def per_expert_fired(self, texts) -> dict[str, np.ndarray]:
        texts = list(texts)
        return {e.name: e.fired(texts) for e in self.experts}

    def predict(self, texts) -> np.ndarray:
        fired = self.per_expert_fired(texts)
        out = np.zeros(len(list(texts)) if not isinstance(texts, list) else len(texts), dtype=bool)
        for arr in fired.values():
            out |= arr
        return out.astype(int)

    def attributions(self, texts) -> list[list[str]]:
        """Qué expertos dispararon por petición (para explicar la alerta)."""
        fired = self.per_expert_fired(texts)
        n = len(next(iter(fired.values())))
        return [[name for name, arr in fired.items() if arr[i]] for i in range(n)]

    def margin(self, texts) -> np.ndarray:
        """Score - umbral del experto más 'seguro' por petición (para incertidumbre).

        Positivo = algún experto alerta con holgura; cercano a 0 = decisión
        dudosa; negativo = ninguno alerta y el más alto quedó cerca."""
        texts = list(texts)
        margins = np.column_stack([e.model.scores(texts) - e.model.threshold.value for e in self.experts])
        return margins.max(axis=1)

    # --- persistencia: cada experto es su propio artefacto .joblib ---
    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        manifest = {"experts": []}
        for e in self.experts:
            path = directory / f"expert_{e.name}.joblib"
            e.model.save(path)
            manifest["experts"].append({"name": e.name, "file": path.name,
                                        "threshold": e.model.threshold.value})
        (directory / "ensemble.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        return directory / "ensemble.json"

    @classmethod
    def load(cls, directory: str | Path) -> "ScoreUnionEnsemble":
        directory = Path(directory)
        manifest = json.loads((directory / "ensemble.json").read_text())
        experts = [
            Expert(name=e["name"], model=OctoMatrixModel.load(directory / e["file"]))
            for e in manifest["experts"]
        ]
        return cls(experts)
