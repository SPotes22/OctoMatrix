"""Genera docs/papers/informe_laboratorio.pdf — informe de laboratorio + replicación."""

from __future__ import annotations

from pathlib import Path

from _common import Paragraph, Spacer, cm, single_column_doc, styles, table

OUT = Path(__file__).parent / "informe_laboratorio.pdf"


def build() -> Path:
    S = styles()
    doc = single_column_doc(str(OUT))
    E = []

    def P(txt, st="body"):
        E.append(Paragraph(txt, S[st]))

    def code(txt):
        E.append(Paragraph(txt.replace("\n", "<br/>"), S["code"]))

    P("UNIVERSIDAD TECNOLÓGICA DE PEREIRA", "title")
    P("Laboratorio de Ciberseguridad · Semillero BlackByte", "subtitle")
    P("OctoMatrix — Detección de ataques OWASP en tráfico HTTP (Informe reproducible)", "subtitle")
    P("Docente: Biviana Yulieth Aguirre Arias &nbsp;·&nbsp; Integrante: Santiago Potes Giraldo", "subtitle")
    E.append(Spacer(1, 0.4 * cm))

    P("Resumen", "h1")
    P("OctoMatrix es un detector de anomalías OWASP sobre peticiones HTTP. Esta versión "
      "corrige el prototipo inicial, que se entrenaba con <b>72 ejemplos escritos a mano</b> y "
      "reportaba un 83% de precisión sobre un test de 15 muestras. Aquí se entrena con "
      "<b>~75.500 peticiones reales</b> (CSIC 2010, ECML/PKDD 2007 y los tests de regresión del "
      "OWASP CRS) y se evalúa con métricas propias de un WAF. El sistema es un <b>Mixture-of-Experts "
      "por procedencia de datos</b>: un experto sobre datos reales, uno sobre payloads generados "
      "(cobertura OWASP) y uno sobre flujos de red. Este informe prioriza la <b>reproducibilidad</b>: "
      "la sección 6 replica todo el experimento en una máquina limpia con un solo comando.")

    P("1. Objetivo", "h1")
    P("Clasificar tráfico HTTP como <i>normal</i> o <i>ataque</i> con un umbral calibrado a una tasa "
      "de falsos positivos (FPR) objetivo, sirviendo como capa heurística para un WAF o un monitor de "
      "tráfico, y medir honestamente su capacidad de generalización.")

    P("2. Materiales", "h1")
    P("Python 3.13, scikit-learn, pandas, numpy, pyarrow. Sin GPU. Fuentes de datos públicas, "
      "descargadas bajo demanda (no se redistribuyen):")
    E.append(table([
        ["Fuente", "Rol", "Aporte"],
        ["CSIC 2010", "tráfico real de una tienda web", "normal + anómalo (sin tipo)"],
        ["ECML/PKDD 2007", "tráfico sintético etiquetado", "7 tipos de ataque"],
        ["OWASP CRS (coreruleset)", "tests de regresión de reglas", "SQLi, XSS, RCE, LFI, ..."],
        ["CSE-CIC-IDS2018", "flujos de red (CICFlowMeter)", "DoS, fuerza bruta (experto de flujos)"],
    ], col_widths=[4.2 * cm, 5.5 * cm, 6.3 * cm]))
    E.append(Paragraph("Tabla 1. Fuentes de datos.", S["cap"]))

    P("3. Metodología", "h1")
    P("Pipeline por etapas: <b>Generar/Recolectar → Limpiar → Particionar → Entrenar → Probar → "
      "Exportar → Post-test</b>. Un único <i>sklearn.Pipeline</i> por experto: TF-IDF de n-gramas de "
      "caracteres + características de seguridad (regex + entropía de Shannon) + Random Forest, con "
      "probabilidad <b>calibrada</b> (CalibratedClassifierCV), un <b>post-filtro regex</b> que reduce "
      "falsos positivos, y un umbral elegido en validación para un FPR objetivo. El artefacto se "
      "serializa con joblib + verificación sha256 (no pickle suelto).")
    P("Partición sin fugas: las peticiones casi-duplicadas se agrupan y van completas al mismo lado; "
      "además se evalúa <b>entre fuentes</b> (entrenar con unas, probar con otra) para medir "
      "generalización real, no memorización.")

    P("4. Resultados reales", "h1")
    E.append(table([
        ["Escenario", "Qué mide", "ROC-AUC", "recall@FPR 1%", "recall@FPR 0,1%"],
        ["in_source", "dentro de las fuentes vistas", "0,989", "89,6%", "59,1%"],
        ["cross-source", "generalización a fuente NUEVA", "0,53", "—", "—"],
        ["adversarial (MoE)", "cobertura 19 cat. OWASP", "—", "~100%", "—"],
        ["flujos (CIC-IDS2018)", "DoS / fuerza bruta", "1,00", "100%", "100%"],
    ], col_widths=[3.6 * cm, 5.0 * cm, 2.2 * cm, 2.6 * cm, 2.6 * cm]))
    E.append(Paragraph("Tabla 2. Métricas honestas por escenario (modelo real, umbral por FPR).", S["cap"]))
    P("Lectura honesta: dentro de las fuentes conocidas el detector es fuerte (recall 89,6% molestando "
      "a 1 de cada 100 usuarios legítimos). Ante un <b>sitio nuevo</b>, ROC-AUC 0,53 (azar): aún no "
      "generaliza, por falta de diversidad de tráfico normal. El 1,00 de flujos refleja que "
      "CIC-IDS2018 es trivialmente separable, no que esté resuelto en producción. El experto generado "
      "sube la cobertura OWASP a ~100% <b>sin</b> alterar al experto real (aislamiento verificado: FPR "
      "extra +0,00%).")

    P("5. Latencia", "h1")
    P("Experto de texto: ~39 ms por petición suelta, ~2 ms/petición en lote. Experto de flujos: "
      "~0,05 ms/flujo en lote. Apto para un consumidor Kafka que procesa por lotes.")

    P("6. Replicación desde cero (un comando)", "h1")
    P("En una máquina limpia (incluida una instancia cloud), todo el experimento se reproduce copiando "
      "estas líneas en una terminal:")
    code("git clone https://github.com/SPotes22/OctoMatrix\n"
         "cd OctoMatrix\n"
         "bash reproducir.sh                 # texto (experto real + generado)\n"
         "WITH_FLOWS=1 bash reproducir.sh    # además, experto de flujos (~0,8 GB)")
    P("El script <i>reproducir.sh</i> crea el entorno, descarga las fuentes, unifica/limpia/particiona, "
      "entrena, ensambla el MoE y evalúa. Es idempotente: si un insumo ya existe, no lo rehace. Los "
      "datos crudos no se versionan; un MANIFEST.json fija el commit/sha256 exacto de cada fuente.")

    P("7. Limitaciones conocidas", "h1")
    for lim in [
        "No generaliza todavía a sitios no vistos: necesita tráfico normal variado (logs propios vía --own-logs).",
        "El set adversarial es sólo evaluación: entrenar con él inflaría el resultado (memorización).",
        "CSIC marca como anómalas peticiones ilegales pero no maliciosas: hay ruido de etiqueta.",
        "IDOR y escalada de privilegios son poco detectables sólo por el payload.",
        "No reemplaza validaciones determinísticas ni otros controles de seguridad; es una capa heurística.",
    ]:
        P("• " + lim)

    P("8. Referencias", "h1")
    for r in [
        "[1] HTTP DATASET CSIC 2010. Information Security Institute, CSIC.",
        "[2] ECML/PKDD 2007 Discovery Challenge — Analyzing Web Traffic.",
        "[3] OWASP Core Rule Set — regression tests. github.com/coreruleset/coreruleset",
        "[4] CSE-CIC-IDS2018. Canadian Institute for Cybersecurity, UNB.",
    ]:
        P(r, "small")

    doc.build(E)
    return OUT


if __name__ == "__main__":
    print(build())
