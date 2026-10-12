"""ECML/PKDD 2007 Discovery Challenge (análisis de tráfico web).

Trae el tipo de ataque por petición: XSS, SqlInjection, LdapInjection,
XPathInjection, PathTransversal, OsCommanding, SSI.

Usa `dataset_ecml_pkdd_train_test.tar.gz` (o los .txt ya extraídos) del mirror
https://github.com/msudol/Web-Application-Attack-Datasets: el train tiene
24.504 válidas y el test 10.502 válidas + 15.110 ataques.

Limitación: las peticiones son sintéticas, con nombres de parámetros y
cabeceras aleatorias. Sirve para variedad de tipos de ataque, pero no
representa tráfico real; conviene evaluar aparte por fuente.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path
from typing import Iterator

from ..http_parse import parse_raw_request
from ..schema import NORMAL, HttpSample
from ._delimited import iter_records

SOURCE = "ecml_pkdd2007"
TARBALL = "dataset_ecml_pkdd_train_test.tar.gz"
FILES = {"xml_train.txt": "train", "xml_test.txt": "test"}

CLASS_MAP = {
    "valid": NORMAL,
    "xss": "xss",
    "sqlinjection": "sqli",
    "ldapinjection": "ldap_injection",
    "xpathinjection": "xpath_injection",
    "pathtransversal": "path_traversal",
    "oscommanding": "command_injection",
    "ssi": "ssi_injection",
}


def _open_members(root: Path) -> Iterator[tuple[str, io.TextIOBase]]:
    # Preferir los .txt extraídos (el mirror los trae) y si no, el tarball.
    candidates = [root, root / "dataset_ecml_pkdd_train_test"]
    for base in candidates:
        if all((base / n).exists() for n in FILES):
            for name in FILES:
                with open(base / name, encoding="latin-1") as fh:
                    yield name, fh
            return
    tar_path = root / TARBALL
    if tar_path.exists():
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar.getmembers():
                name = Path(member.name).name
                if name in FILES and member.isfile():
                    yield name, io.TextIOWrapper(tar.extractfile(member), encoding="latin-1")
        return
    raise FileNotFoundError(f"no encontré {list(FILES)} ni {TARBALL} en {root}")


def load(root: str | Path) -> Iterator[HttpSample]:
    for name, fh in _open_members(Path(root)):
        for rec_id, cls, raw in iter_records(fh):
            attack_type = CLASS_MAP.get(cls.lower())
            if attack_type is None:
                raise ValueError(f"clase ECML desconocida: {cls!r} ({name}:{rec_id})")
            parsed = parse_raw_request(raw)
            if parsed is None:
                continue
            yield HttpSample(
                **parsed,
                label=int(attack_type != NORMAL),
                attack_type=attack_type,
                source=SOURCE,
                origin_split=FILES[name],
                sample_id=f"{name}:{rec_id}",
            )
