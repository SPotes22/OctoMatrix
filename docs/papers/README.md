# Documentación (PDF)

Los PDF se generan con reportlab (no hay LaTeX en el entorno). El **fuente** es
el código Python de aquí; los `.pdf` son artefactos que se regeneran:

```bash
pip install reportlab
cd docs/papers
python gen_informe.py   # informe_laboratorio.pdf  (informe + replicación)
python gen_paper.py     # Paper_Octomatrix.pdf     (IEEE dos columnas)
python gen_manual.py    # Manual_de_implementacion.pdf (post-exportación)
```

Números y contenido salen de los resultados reales del proyecto
(docs/DATA.md, docs/MODEL.md, docs/METRICS.md, docs/MOE.md).
