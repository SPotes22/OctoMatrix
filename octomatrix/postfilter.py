"""Post-filtro regex (Fase 3): reduce falsos positivos con una allowlist.

Va DESPUÉS del modelo: el RF da una probabilidad calibrada y, si una petición
cae en la allowlist (tráfico claramente benigno) y además no dispara ninguna
firma de ataque fuerte, se rebaja su score. Nunca sube el score: sólo rescata
falsos positivos, no inventa detecciones.

La allowlist es explícita y auditable. Por defecto cubre recursos estáticos y
rutas de API/salud típicas. Es configurable por proyecto.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .features import PATTERNS

# Firmas fuertes: si alguna dispara, el post-filtro NO rescata aunque la ruta
# parezca benigna (una petición a /static/x.css con `<script>` sigue siendo ataque).
STRONG_SIGNATURES = ["sql_tautology", "xss_tag", "xss_handler", "path_traversal", "cmd_injection", "ssi_xxe", "sql_keywords"]

DEFAULT_ALLOW = [
    r"\.(?:css|js|png|jpe?g|gif|svg|ico|woff2?|ttf|map|webp|mp4|pdf)(?:\?|$)",
    r"^/(?:static|assets|images|img|js|css|fonts|downloads|media)/",
    r"^/(?:health|healthz|status|metrics|ping|favicon\.ico|robots\.txt)\b",
]


@dataclass
class RegexPostFilter:
    allow: list[str] = field(default_factory=lambda: list(DEFAULT_ALLOW))
    rescue_to: float = 0.0  # score al que se rebaja una petición rescatada

    def __post_init__(self):
        self._allow = [re.compile(p, re.I) for p in self.allow]
        self._strong = [PATTERNS[k] for k in STRONG_SIGNATURES]

    def is_allowlisted(self, text: str) -> bool:
        text = text or ""
        if any(p.search(text) for p in self._strong):
            return False
        return any(p.search(text) for p in self._allow)

    def apply(self, texts, scores) -> np.ndarray:
        """Devuelve los scores con los falsos positivos de la allowlist rebajados."""
        scores = np.asarray(scores, dtype=float).copy()
        for i, text in enumerate(texts):
            if self.is_allowlisted(text):
                scores[i] = min(scores[i], self.rescue_to)
        return scores
