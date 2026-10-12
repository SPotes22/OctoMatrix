#!/usr/bin/env python3
"""Descarga las fuentes crudas en data/raw/ y deja un MANIFEST.json con versiones.

    python scripts/download_data.py                 # csic2010, ecml_pkdd2007, crs
    python scripts/download_data.py --cicids web    # + los 2 días con ataques web (~0,8 GB)
    python scripts/download_data.py --cicids all    # + los 10 días (~6,9 GB)

Sólo usa git y HTTPS (sin cuenta de Kaggle ni AWS CLI).
Los datos NO se suben al repo: data/ está en .gitignore.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from octomatrix.data.loaders import cicids2018  # noqa: E402

# Mirror de CSIC 2010 y ECML/PKDD 2007 (el sitio oficial del CSIC está caído).
WAAD_REPO = "https://github.com/msudol/Web-Application-Attack-Datasets"
WAAD_REV = "424c6e1c1951570473d9d6c0227eaaac6fdd6d02"
WAAD_FILES = {
    "csic2010": "OriginalDataSets/csic_2010/dataset_cisc_train_test.tar.gz",
    "ecml_pkdd2007": "OriginalDataSets/ecml_pkdd/dataset_ecml_pkdd_train_test.tar.gz",
}

CRS_REPO = "https://github.com/coreruleset/coreruleset"
CRS_REV = "af76ccfcd1ad947e38ef9998bdd464f1a39fef86"
CRS_PATH = "tests/regression/tests"


def git(*args, cwd=None):
    subprocess.run(["git", *args], cwd=cwd, check=True)


def sparse_clone(repo: str, rev: str, paths: list[str], dest: Path, no_cone: bool = False) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    git("clone", "--quiet", "--filter=blob:none", "--no-checkout", repo, str(dest))
    git("sparse-checkout", "init", *(["--no-cone"] if no_cone else ["--cone"]), cwd=dest)
    git("sparse-checkout", "set", *paths, cwd=dest)
    git("checkout", "--quiet", rev, cwd=dest)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch_waad(raw: Path, manifest: dict) -> None:
    tmp = raw / ".waad"
    sparse_clone(WAAD_REPO, WAAD_REV, list(WAAD_FILES.values()), tmp, no_cone=True)
    for name, rel in WAAD_FILES.items():
        dst = raw / name / Path(rel).name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tmp / rel, dst)
        manifest[name] = {"repo": WAAD_REPO, "rev": WAAD_REV, "file": rel, "sha256": sha256(dst)}
        print(f"  {name}: {dst}")
    shutil.rmtree(tmp)


def fetch_crs(raw: Path, manifest: dict) -> None:
    dest = raw / "coreruleset"
    sparse_clone(CRS_REPO, CRS_REV, [CRS_PATH], dest)
    manifest["crs_regression"] = {"repo": CRS_REPO, "rev": CRS_REV, "path": CRS_PATH}
    print(f"  crs_regression: {dest / CRS_PATH}")


def fetch_cicids(raw: Path, manifest: dict, which: str) -> None:
    days = cicids2018.WEB_ATTACK_DAYS if which == "web" else cicids2018.DAY_FILES
    dest = raw / "cicids2018"
    dest.mkdir(parents=True, exist_ok=True)
    for day in days:
        out = dest / day
        url = f"{cicids2018.S3_BASE}/{urllib.parse.quote(cicids2018.S3_PREFIX + day)}"
        if out.exists():
            print(f"  ya existe {out.name}")
        else:
            print(f"  bajando {day} ...")
            tmp = out.with_suffix(".part")
            with urllib.request.urlopen(url) as resp, open(tmp, "wb") as fh:
                shutil.copyfileobj(resp, fh, length=1 << 20)
            tmp.rename(out)
        manifest.setdefault("cicids2018", {"url": cicids2018.S3_BASE, "files": {}})["files"][day] = {
            "bytes": out.stat().st_size
        }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--skip-http", action="store_true", help="no bajar CSIC/ECML/CRS")
    ap.add_argument("--cicids", choices=["none", "web", "all"], default="none")
    args = ap.parse_args()

    raw = args.raw_dir
    raw.mkdir(parents=True, exist_ok=True)
    manifest_path = raw / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    if not args.skip_http:
        print("CSIC 2010 + ECML/PKDD 2007:")
        fetch_waad(raw, manifest)
        print("OWASP CRS regression tests:")
        fetch_crs(raw, manifest)
    if args.cicids != "none":
        print("CSE-CIC-IDS2018:")
        fetch_cicids(raw, manifest, args.cicids)

    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
