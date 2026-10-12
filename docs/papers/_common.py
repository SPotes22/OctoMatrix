"""Utilidades compartidas para generar los PDF de documentación con reportlab.

No hay LaTeX en el entorno, así que los PDF se construyen con reportlab
(Platypus). Este módulo centraliza estilos, cabeceras y helpers de tabla.
"""

from __future__ import annotations

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table, TableStyle

ACCENT = colors.HexColor("#1f3b66")
GREY = colors.HexColor("#555555")


def styles() -> dict:
    s = getSampleStyleSheet()
    out = {
        "title": ParagraphStyle("t", parent=s["Title"], fontSize=16, leading=19, textColor=ACCENT, alignment=TA_CENTER),
        "subtitle": ParagraphStyle("st", parent=s["Normal"], fontSize=10, leading=13, alignment=TA_CENTER, textColor=GREY),
        "h1": ParagraphStyle("h1", parent=s["Heading1"], fontSize=12, leading=15, textColor=ACCENT, spaceBefore=10, spaceAfter=4),
        "h2": ParagraphStyle("h2", parent=s["Heading2"], fontSize=10.5, leading=13, textColor=ACCENT, spaceBefore=6, spaceAfter=2),
        "body": ParagraphStyle("b", parent=s["BodyText"], fontSize=9.5, leading=13, alignment=TA_JUSTIFY, spaceAfter=4),
        "small": ParagraphStyle("sm", parent=s["BodyText"], fontSize=8, leading=10, textColor=GREY),
        "code": ParagraphStyle("c", parent=s["Code"], fontSize=8, leading=10, backColor=colors.HexColor("#f2f2f2"),
                               borderPadding=4, leftIndent=2, spaceBefore=3, spaceAfter=6),
        "cap": ParagraphStyle("cap", parent=s["Normal"], fontSize=8, leading=10, alignment=TA_CENTER, textColor=GREY, spaceAfter=8),
        "abstract": ParagraphStyle("ab", parent=s["BodyText"], fontSize=9, leading=12, alignment=TA_JUSTIFY),
    }
    return out


def table(data, col_widths=None, header=True, font=8) -> Table:
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    style = [
        ("FONTSIZE", (0, 0), (-1, -1), font),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), ACCENT), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                  ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    t.setStyle(TableStyle(style))
    return t


def single_column_doc(path, **margins):
    m = {"leftMargin": 2.2 * cm, "rightMargin": 2.2 * cm, "topMargin": 2 * cm, "bottomMargin": 2 * cm}
    m.update(margins)
    doc = BaseDocTemplate(path, pagesize=letter, **m)
    w, h = letter
    frame = Frame(m["leftMargin"], m["bottomMargin"], w - m["leftMargin"] - m["rightMargin"],
                  h - m["topMargin"] - m["bottomMargin"], id="single")
    doc.addPageTemplates([PageTemplate(id="single", frames=[frame])])
    return doc


def two_column_doc(path):
    lm = rm = 1.6 * cm
    tm = 1.6 * cm
    bm = 1.6 * cm
    gap = 0.7 * cm
    doc = BaseDocTemplate(path, pagesize=letter, leftMargin=lm, rightMargin=rm, topMargin=tm, bottomMargin=bm)
    w, h = letter
    cw = (w - lm - rm - gap) / 2
    # banda superior a todo el ancho para título (primera página)
    top = h - tm - 4.2 * cm
    full = Frame(lm, top, w - lm - rm, 4.2 * cm, id="banner")
    c1 = Frame(lm, bm, cw, top - bm, id="c1")
    c2 = Frame(lm + cw + gap, bm, cw, top - bm, id="c2")
    col1 = Frame(lm, bm, cw, h - tm - bm, id="col1")
    col2 = Frame(lm + cw + gap, bm, cw, h - tm - bm, id="col2")
    doc.addPageTemplates([
        PageTemplate(id="first", frames=[full, c1, c2]),
        PageTemplate(id="rest", frames=[col1, col2]),
    ])
    return doc


__all__ = ["styles", "table", "single_column_doc", "two_column_doc", "Paragraph", "Spacer", "cm", "colors"]
