"""HTTP DATASET CSIC 2010 (Instituto de Seguridad de la Información, CSIC).

Tráfico generado contra una tienda web (Java/JSP): 36.000 peticiones normales
de entrenamiento, 36.000 normales de test y 25.065 anómalas de test.

Usa la versión formateada `dataset_cisc_train_test.tar.gz` del mirror
https://github.com/msudol/Web-Application-Attack-Datasets (el sitio original
www.isi.csic.es lleva tiempo caído).

Limitaciones a tener en cuenta en el modelado:
- No hay tipo de ataque por petición: attack_type = "unspecified".
- "Anómalo" incluye peticiones ilegales no maliciosas (p. ej. parámetros que
  no existen o valores fuera de formato), así que hay ruido de etiqueta.
- Un solo sitio y una sola app: normales y ataques comparten rutas, lo que es
  bueno (no hay fuga por forma), pero limita la generalización.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path
from typing import Iterator

from ..http_parse import parse_raw_request
from ..schema import NORMAL, HttpSample
from ._delimited import iter_records

SOURCE = "csic2010"
TARBALL = "dataset_cisc_train_test.tar.gz"
FILES = {
    "cisc_normalTraffic_train.txt": "train",
    "cisc_normalTraffic_test.txt": "test",
    "cisc_anomalousTraffic_test.txt": "test",
}


def _open_members(root: Path) -> Iterator[tuple[str, io.TextIOBase]]:
    tar_path = root / TARBALL
    if tar_path.exists():
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar.getmembers():
                name = Path(member.name).name
                if name in FILES and member.isfile():
                    fh = tar.extractfile(member)
                    yield name, io.TextIOWrapper(fh, encoding="latin-1")
        return
    for name in FILES:
        p = root / name
        if p.exists():
            with open(p, encoding="latin-1") as fh:
                yield name, fh


def load(root: str | Path) -> Iterator[HttpSample]:
    root = Path(root)
    found = False
    for name, fh in _open_members(root):
        found = True
        split = FILES[name]
        for rec_id, cls, raw in iter_records(fh):
            parsed = parse_raw_request(raw)
            if parsed is None:
                continue
            is_attack = cls.lower() != "valid"
            yield HttpSample(
                **parsed,
                label=int(is_attack),
                attack_type="unspecified" if is_attack else NORMAL,
                source=SOURCE,
                origin_split=split,
                sample_id=f"{name}:{rec_id}",
            )
    if not found:
        raise FileNotFoundError(f"no encontré {TARBALL} ni {list(FILES)} en {root}")
