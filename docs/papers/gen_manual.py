"""Genera docs/papers/Manual_de_implementacion.pdf — pasos DESPUÉS de exportar el modelo."""

from __future__ import annotations

from pathlib import Path

from _common import Paragraph, Spacer, cm, single_column_doc, styles, table

OUT = Path(__file__).parent / "Manual_de_implementacion.pdf"


def build() -> Path:
    S = styles()
    doc = single_column_doc(str(OUT))
    E = []

    def P(txt, st="body"):
        E.append(Paragraph(txt, S[st]))

    def code(txt):
        E.append(Paragraph(txt.replace("\n", "<br/>").replace(" ", "&nbsp;"), S["code"]))

    P("Manual de Implementación — OctoMatrix", "title")
    P("Qué hacer DESPUÉS de tener el modelo exportado", "subtitle")
    P("Semillero BlackByte · Universidad Tecnológica de Pereira", "subtitle")
    P("Docente: Biviana Yulieth Aguirre Arias · Integrante: Santiago Potes Giraldo", "subtitle")
    E.append(Spacer(1, 0.4 * cm))

    P("0. Punto de partida", "h1")
    P("Este manual asume que ya corriste el entrenamiento (ver <i>informe_laboratorio.pdf</i> / "
      "<i>reproducir.sh</i>) y tienes en <b>models/</b> los artefactos:")
    E.append(table([
        ["Artefacto", "Qué es"],
        ["models/octomatrix.joblib", "experto de texto (real) + su .sha256 y _meta.json"],
        ["models/moe/", "MoE: experto real + experto generado (ensemble.json)"],
        ["models/flows.joblib", "experto de flujos de red (opcional)"],
    ], col_widths=[6.2 * cm, 9.8 * cm]))
    P("OctoMatrix es una <b>capa de validación</b>: no reemplaza otros controles de seguridad; "
      "emite una evaluación (normal/sospechoso) y la aplicación decide la acción.")

    P("1. Cargar el modelo", "h1")
    P("Se carga una sola vez por proceso. La carga verifica el sha256 del artefacto antes de "
      "deserializar (integridad).")
    code('from octomatrix.moe import ScoreUnionEnsemble\n'
         'from octomatrix.data.text import request_text\n\n'
         'moe = ScoreUnionEnsemble.load("models/moe")   # una vez al iniciar')

    P("2. Evaluar una petición", "h1")
    P("La aplicación normaliza el evento (método, ruta, query, cuerpo) y lo pasa como texto. El "
      "ensemble devuelve la decisión (OR de los expertos) y qué experto disparó.")
    code('texto = request_text("POST", "/login", "",\n'
         '                     "user=admin&pass=admin\' OR \'1\'=\'1")\n'
         'alerta = bool(moe.predict([texto])[0])          # True = ataque\n'
         'quien  = moe.attributions([texto])[0]           # ["real"], ["generated"]...\n'
         'margen = float(moe.margin([texto])[0])          # cercanía a la frontera')
    P("Decisión según resultado: la aplicación registra, bloquea o alerta. OctoMatrix no decide la "
      "acción final.")

    P("3. Integración como middleware (productor/consumidor)", "h1")
    P("En una arquitectura con Kafka u otra cola, OctoMatrix va entre el productor de eventos y la "
      "lógica de negocio:")
    P("1. La aplicación (p. ej. Parchate Pereira) produce el evento HTTP.<br/>"
      "2. Un consumidor lo normaliza con <i>request_text</i>.<br/>"
      "3. Llama a <i>moe.predict</i> (en lote para rendimiento: ~2 ms/petición).<br/>"
      "4. Publica la decisión; la aplicación actúa.")
    P("<b>Importante para Parchate Pereira:</b> reemplazar el <i>_sanitize_input</i> actual (una "
      "blacklist que borra &lt;, &gt; y 'script', evadible con <i>&lt;scr&lt;script&gt;ipt&gt;</i>) por "
      "esta evaluación del modelo. La blacklist da falsa sensación de seguridad; el detector mide "
      "probabilidad de ataque sobre el payload completo.")

    P("4. Rendimiento y operación", "h1")
    for t in [
        "Procesar en LOTE siempre que se pueda: ~2 ms/petición vs ~39 ms suelta.",
        "Cargar el modelo una sola vez por proceso (no por petición).",
        "No modificar el .joblib en caliente; si cambia, reiniciar el proceso.",
        "Registrar todas las decisiones (entra el lazo de post-entrenamiento, sección 6).",
        "Un umbral más estricto (FPR 0,1%) detecta menos pero molesta menos; decidir con negocio.",
    ]:
        P("• " + t)

    P("5. Experto de flujos (si se usa)", "h1")
    P("El experto de flujos (DoS, fuerza bruta) consume <b>flujos de red</b>, no peticiones HTTP. Va "
      "en paralelo, sobre la telemetría de red (CICFlowMeter), y compone a nivel de alerta con el "
      "experto de texto: hay alerta si cualquiera dispara. No se combinan por-petición porque son "
      "telemetrías de distinta granularidad.")

    P("6. Post-entrenamiento (lazo de feedback)", "h1")
    P("El modelo NO se reentrena solo. El tráfico de producción se registra, se seleccionan los casos "
      "dudosos (margen pequeño) para revisión humana, y lo revisado alimenta un reentrenamiento "
      "<b>manual</b>. El lazo se rompe a propósito en la revisión humana: ahí es donde entraría el "
      "envenenamiento de datos si fuera automático.")
    code('python scripts/postprod.py log   --model models/moe --traffic trafico.jsonl\n'
         'python scripts/postprod.py queue --band 0.1 --max 300\n'
         '# una persona revisa data/feedback/review_queue.jsonl (label 0/1 + reviewer)\n'
         'python scripts/postprod.py ingest')
    P("Para cerrar la brecha de generalización, alimentar el experto real con tráfico benigno de tus "
      "propias apps (access logs) vía <i>--own-logs</i> (ver docs/BENIGN.md).")

    P("7. Limitaciones conocidas", "h1")
    for t in [
        "Depende de la calidad del entrenamiento; no detecta ataques completamente nuevos.",
        "No generaliza aún a sitios no vistos: necesita benigno propio.",
        "No reemplaza validaciones determinísticas ni un WAF de reglas.",
        "IDOR y escalada de privilegios no se detectan bien sólo por el payload.",
        "La verificación sha256 protege contra corrupción, no contra un atacante que controle ambos archivos.",
    ]:
        P("• " + t)

    doc.build(E)
    return OUT


if __name__ == "__main__":
    print(build())
