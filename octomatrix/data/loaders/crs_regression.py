"""Tests de regresión del OWASP Core Rule Set (formato go-ftw).

Fuente: https://github.com/coreruleset/coreruleset, carpeta
tests/regression/tests. Sustituye a SpiderLabs/OWASP-CRS-regressions,
que está archivado.

Cada test es una petición y la regla CRS que debe dispararse (expect_ids) o
no dispararse (no_expect_ids):
- expect_ids no vacío → ataque, con attack_type según la familia de la regla.
- sólo no_expect_ids  → regresión de falso positivo: petición que el CRS NO
  debe bloquear. Se etiqueta 0 pero con source "crs_regression_negative" para
  poder filtrarla: el test garantiza que esa regla no salta, no que la
  petición sea inocua para todas las demás.

Sólo se usan familias de ataque a aplicación. Se excluyen 901/911/920/922/949/
999 (inicialización, métodos, conformidad de protocolo, multipart y
evaluación) y todas las RESPONSE-*, porque no son payloads de petición.
"""

from __future__ import annotations

import base64
import binascii
from pathlib import Path
from typing import Iterator

import yaml

from ..http_parse import clean_headers, parse_raw_request, split_uri
from ..schema import NORMAL, HttpSample

SOURCE = "crs_regression"
SOURCE_NEGATIVE = "crs_regression_negative"
TESTS_SUBDIR = Path("tests/regression/tests")

FAMILY_MAP = {
    "913": "scanner",
    "921": "protocol_attack",
    "930": "lfi",
    "931": "rfi",
    "932": "rce",
    "933": "php_injection",
    "934": "generic_injection",
    "941": "xss",
    "942": "sqli",
    "943": "session_fixation",
    "944": "java_injection",
}


def _tests_dir(root: Path) -> Path:
    for cand in (root / TESTS_SUBDIR, root):
        if any(cand.glob("REQUEST-9*")):
            return cand
    raise FileNotFoundError(f"no encontré carpetas REQUEST-9xx en {root} ni en {root / TESTS_SUBDIR}")


def _family(dirname: str) -> str | None:
    # "REQUEST-942-APPLICATION-ATTACK-SQLI" -> "942"
    parts = dirname.split("-")
    if len(parts) < 2 or parts[0] != "REQUEST":
        return None
    return parts[1] if parts[1] in FAMILY_MAP else None


def _request_from_input(inp: dict) -> dict | None:
    if inp.get("encoded_request"):
        try:
            raw = base64.b64decode(inp["encoded_request"]).decode("latin-1")
        except (binascii.Error, ValueError):
            return None
        return parse_raw_request(raw)
    path, query = split_uri(str(inp.get("uri") or "/"))
    headers = inp.get("headers") or {}
    data = inp.get("data")
    return {
        "method": str(inp.get("method") or "GET").upper(),
        "path": path,
        "query": query,
        "body": "" if data is None else str(data),
        "headers": clean_headers((str(k), str(v)) for k, v in headers.items()),
    }


def load(root: str | Path) -> Iterator[HttpSample]:
    tests_dir = _tests_dir(Path(root))
    for yaml_path in sorted(tests_dir.glob("REQUEST-*/*.yaml")):
        family = _family(yaml_path.parent.name)
        if family is None:
            continue
        doc = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        rule_id = doc.get("rule_id", yaml_path.stem)
        for test in doc.get("tests", []):
            for n, stage in enumerate(test.get("stages", [])):
                stage = stage.get("stage", stage)  # formato viejo anidaba "stage:"
                log = (stage.get("output") or {}).get("log") or {}
                if log.get("expect_ids"):
                    label, attack_type, source = 1, FAMILY_MAP[family], SOURCE
                elif log.get("no_expect_ids"):
                    label, attack_type, source = 0, NORMAL, SOURCE_NEGATIVE
                else:
                    continue  # tests de status/regex: no dicen si es ataque
                req = _request_from_input(stage.get("input") or {})
                if req is None:
                    continue
                yield HttpSample(
                    **req,
                    label=label,
                    attack_type=attack_type,
                    source=source,
                    origin_split="-",
                    sample_id=f"{rule_id}:{test.get('test_id')}:{n}",
                )
