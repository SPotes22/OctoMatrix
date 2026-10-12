"""Mapa de attack_type -> categoría OWASP Top 10 (2021), para reportes por categoría.

Varias familias caen en A03 (Injection), pero conviene verlas por separado en
el reporte; por eso el reporte agrupa por OWASP y también por attack_type.
Categorías que no son del Top 10 web (recon, protocolo, memoria) se marcan con
un prefijo "X_" para que no se confundan con las oficiales.
"""

from __future__ import annotations

OWASP_2021 = {
    "A01": "Broken Access Control",
    "A02": "Cryptographic Failures",
    "A03": "Injection",
    "A04": "Insecure Design",
    "A05": "Security Misconfiguration",
    "A06": "Vulnerable Components",
    "A07": "Identification & Authentication Failures",
    "A08": "Software & Data Integrity Failures",
    "A09": "Security Logging & Monitoring Failures",
    "A10": "Server-Side Request Forgery",
    "X_recon": "Reconnaissance / scanners",
    "X_protocol": "Protocol attacks",
    "X_memory": "Memory safety (buffer overflow)",
    "X_unspecified": "Anomalía sin categoría",
}

ATTACK_TYPE_TO_OWASP = {
    "normal": None,
    # A01 Broken Access Control
    "path_traversal": "A01",
    "lfi": "A01",
    "idor": "A01",
    "privilege_escalation": "A01",
    "open_redirect": "A01",
    # A03 Injection
    "sqli": "A03",
    "nosqli": "A03",
    "xss": "A03",
    "command_injection": "A03",
    "rce": "A03",
    "ldap_injection": "A03",
    "xpath_injection": "A03",
    "ssi_injection": "A03",
    "ssti": "A03",
    "php_injection": "A03",
    "java_injection": "A03",
    "generic_injection": "A03",
    "crlf": "A03",
    "rfi": "A03",
    # A05 Security Misconfiguration (XXE se reclasificó aquí en 2021)
    "xxe": "A05",
    # A07 Auth failures
    "session_fixation": "A07",
    # A08 Integrity failures
    "deserialization": "A08",
    # A10 SSRF
    "ssrf": "A10",
    # fuera del Top 10 web
    "scanner": "X_recon",
    "protocol_attack": "X_protocol",
    "buffer_overflow": "X_memory",
    "unspecified": "X_unspecified",
}


def owasp_category(attack_type: str) -> str | None:
    return ATTACK_TYPE_TO_OWASP.get(attack_type)


def owasp_label(attack_type: str) -> str | None:
    cat = owasp_category(attack_type)
    return f"{cat} {OWASP_2021[cat]}" if cat else None
