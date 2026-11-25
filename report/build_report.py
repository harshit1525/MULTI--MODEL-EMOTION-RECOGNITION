"""Build report/Project_Report.docx from real report/ assets, tables, formulas, and results."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence

import docx
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Cm, Inches, Pt, RGBColor

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "report"
FIG_DIR = REPORT_DIR / "figures"
TAB_DIR = REPORT_DIR / "tables"
FORM_DIR = REPORT_DIR / "formulas"

TEXT_WIDTH_IN = 5.75  # A4 (8.268 in) - left 1.5 in - right 1.0 in = 5.768 in


# ---------------------------------------------------------------------------
# XML & STYLING HELPERS
# ---------------------------------------------------------------------------
def configure_a4_margins(section) -> None:
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.left_margin = Inches(1.5)
    section.right_margin = Inches(1.0)
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)


def set_page_number_format(section, fmt: str = "decimal", start: int | None = None) -> None:
    sectPr = section._sectPr
    for existing in sectPr.findall(qn("w:pgNumType")):
        sectPr.remove(existing)
    pg = OxmlElement("w:pgNumType")
    pg.set(qn("w:fmt"), fmt)
    if start is not None:
        pg.set(qn("w:start"), str(start))
    sectPr.append(pg)


def add_field(paragraph, field_text: str, font_size: float = 11.0, bold: bool = False) -> None:
    run_begin = paragraph.add_run()
    run_begin.font.name = "Times New Roman"
    run_begin.font.size = Pt(font_size)
    fldChar1 = parse_xml(r'<w:fldChar %s w:fldCharType="begin"/>' % nsdecls("w"))
    run_begin._r.append(fldChar1)

    run_instr = paragraph.add_run()
    run_instr.font.name = "Times New Roman"
    run_instr.font.size = Pt(font_size)
    instrText = parse_xml(r'<w:instrText %s xml:space="preserve"> %s </w:instrText>' % (nsdecls("w"), field_text))
    run_instr._r.append(instrText)

    run_sep = paragraph.add_run()
    fldChar2 = parse_xml(r'<w:fldChar %s w:fldCharType="separate"/>' % nsdecls("w"))
    run_sep._r.append(fldChar2)

    run_val = paragraph.add_run("1")
    run_val.font.name = "Times New Roman"
    run_val.font.size = Pt(font_size)
    run_val.bold = bold

    run_end = paragraph.add_run()
    fldChar3 = parse_xml(r'<w:fldChar %s w:fldCharType="end"/>' % nsdecls("w"))
    run_end._r.append(fldChar3)


def set_bottom_center_page_number(section, fmt: str = "decimal", start: int | None = None) -> None:
    section.different_first_page_header_footer = False
    section.footer.is_linked_to_previous = False
    set_page_number_format(section, fmt=fmt, start=start)
    footer = section.footer
    p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    p.text = ""
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    add_field(p, "PAGE", font_size=11.0)


def setup_styles(doc: Document) -> None:
    # Enable automatic field updating on open
    settings_el = doc.settings.element
    if settings_el.find(qn("w:updateFields")) is None:
        settings_el.append(parse_xml(r'<w:updateFields %s w:val="true"/>' % nsdecls("w")))

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)
    normal.font.color.rgb = RGBColor(0, 0, 0)
    normal.paragraph_format.line_spacing = 1.5
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.space_before = Pt(0)

    for lvl, sz, sb, sa, pb in [
        ("Heading 1", 16, 10, 10, True),
        ("Heading 2", 14, 12, 6, False),
        ("Heading 3", 12, 8, 4, False),
    ]:
        st = styles[lvl]
        st.font.name = "Times New Roman"
        st.font.size = Pt(sz)
        st.font.bold = True
        st.font.color.rgb = RGBColor(0, 0, 0)
        st.paragraph_format.line_spacing = 1.5
        st.paragraph_format.space_before = Pt(sb)
        st.paragraph_format.space_after = Pt(sa)
        st.paragraph_format.keep_with_next = True
        st.paragraph_format.page_break_before = pb

    for cname in ("FigureCaption", "TableCaption"):
        if cname not in [s.name for s in styles]:
            cs = styles.add_style(cname, docx.enum.style.WD_STYLE_TYPE.PARAGRAPH)
            cs.base_style = normal
            cs.font.name = "Times New Roman"
            cs.font.size = Pt(10.5)
            cs.font.bold = True
            cs.font.color.rgb = RGBColor(20, 20, 20)
            cs.paragraph_format.line_spacing = 1.15
            cs.paragraph_format.space_before = Pt(4)
            cs.paragraph_format.space_after = Pt(8)
            if cname == "TableCaption":
                cs.paragraph_format.keep_with_next = True

    if "CodeBlock" not in [s.name for s in styles]:
        cb = styles.add_style("CodeBlock", docx.enum.style.WD_STYLE_TYPE.PARAGRAPH)
        cb.font.name = "Consolas"
        cb.font.size = Pt(9)
        cb.font.color.rgb = RGBColor(20, 20, 20)
        cb.paragraph_format.line_spacing = 1.05
        cb.paragraph_format.space_before = Pt(2)
        cb.paragraph_format.space_after = Pt(4)
        cb.paragraph_format.left_indent = Inches(0.15)


def add_para(doc: Document, text: str, align=WD_ALIGN_PARAGRAPH.JUSTIFY, bold: bool = False,
             italic: bool = False, size: float = 12.0, space_after: float = 6.0) -> docx.text.paragraph.Paragraph:
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_after = Pt(space_after)
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(size)
    r.bold = bold
    r.italic = italic
    return p


def add_bullet(doc: Document, lead: str, body: str) -> None:
    p = doc.add_paragraph(style="List Bullet")
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_after = Pt(4)
    if lead:
        r1 = p.add_run(lead)
        r1.bold = True
        r1.font.name = "Times New Roman"
        r1.font.size = Pt(12)
    r2 = p.add_run(body)
    r2.font.name = "Times New Roman"
    r2.font.size = Pt(12)


def add_chapter_heading(doc: Document, num: str, title: str, page_break: bool = True) -> None:
    p = doc.add_paragraph(style="Heading 1")
    p.paragraph_format.page_break_before = page_break
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    r = p.add_run(f"CHAPTER {num}: {title.upper()}" if num.isdigit() else title.upper())
    r.font.name = "Times New Roman"
    r.font.size = Pt(16)
    r.bold = True


def add_h2(doc: Document, title: str) -> None:
    p = doc.add_paragraph(style="Heading 2")
    r = p.add_run(title)
    r.font.name = "Times New Roman"
    r.font.size = Pt(14)
    r.bold = True


def add_h3(doc: Document, title: str) -> None:
    p = doc.add_paragraph(style="Heading 3")
    r = p.add_run(title)
    r.font.name = "Times New Roman"
    r.font.size = Pt(12)
    r.bold = True


def add_mapped_pos(doc: Document, pos_line: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(6)
    r1 = p.add_run("Mapped Program Outcomes (POs): ")
    r1.bold = True
    r1.font.name = "Times New Roman"
    r1.font.size = Pt(11.5)
    r2 = p.add_run(pos_line)
    r2.italic = True
    r2.font.name = "Times New Roman"
    r2.font.size = Pt(11.5)


def add_figure(doc: Document, fig_rel_path: str, fig_num: str, caption: str, width_in: float = 5.5) -> None:
    img_path = REPO_ROOT / fig_rel_path
    assert img_path.is_file(), f"Missing figure file: {img_path}"
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(6)
    p_img.paragraph_format.space_after = Pt(3)
    p_img.paragraph_format.keep_with_next = True
    p_img.add_run().add_picture(str(img_path), width=Inches(min(width_in, 5.75)))

    p_cap = doc.add_paragraph(style="FigureCaption")
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p_cap.add_run(f"Figure {fig_num}: {caption}")
    r.font.name = "Times New Roman"
    r.font.size = Pt(10.5)
    r.bold = True


def set_table_borders_and_style(table, col_widths: Sequence[float], font_pt: float = 9.0, header_fill: str = "E8EEF5") -> None:
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        '<w:top w:val="single" w:sz="6" w:space="0" w:color="333333"/>'
        '<w:bottom w:val="single" w:sz="6" w:space="0" w:color="333333"/>'
        '<w:left w:val="single" w:sz="4" w:space="0" w:color="555555"/>'
        '<w:right w:val="single" w:sz="4" w:space="0" w:color="555555"/>'
        '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="777777"/>'
        '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="777777"/>'
        "</w:tblBorders>"
    )
    for old in tblPr.findall(qn("w:tblBorders")):
        tblPr.remove(old)
    tblPr.append(borders)

    for r_idx, row in enumerate(table.rows):
        trPr = row._tr.get_or_add_trPr()
        if r_idx == 0:
            trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
        trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
        for c_idx, cell in enumerate(row.cells):
            cell.width = Inches(col_widths[c_idx])
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            tcPr = cell._tc.get_or_add_tcPr()
            tcMar = parse_xml(
                f'<w:tcMar {nsdecls("w")}>'
                '<w:top w:w="55" w:type="dxa"/><w:bottom w:w="55" w:type="dxa"/>'
                '<w:left w:w="80" w:type="dxa"/><w:right w:w="80" w:type="dxa"/>'
                "</w:tcMar>"
            )
            for old_m in tcPr.findall(qn("w:tcMar")):
                tcPr.remove(old_m)
            tcPr.append(tcMar)
            if r_idx == 0:
                tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{header_fill}"/>'))
            for p in cell.paragraphs:
                p.paragraph_format.line_spacing = 1.1
                p.paragraph_format.space_before = Pt(1)
                p.paragraph_format.space_after = Pt(1)
                for run in p.runs:
                    run.font.name = "Times New Roman"
                    run.font.size = Pt(font_pt)
                    if r_idx == 0:
                        run.bold = True


def add_styled_table(doc: Document, tab_num: str, caption: str, headers: Sequence[str],
                     rows: Sequence[Sequence[str]], col_widths: Sequence[float],
                     font_pt: float = 9.0) -> None:
    p_cap = doc.add_paragraph(style="TableCaption")
    p_cap.alignment = WD_ALIGN_PARAGRAPH.LEFT
    r = p_cap.add_run(f"Table {tab_num}: {caption}")
    r.font.name = "Times New Roman"
    r.font.size = Pt(10.5)
    r.bold = True

    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    for j, h in enumerate(headers):
        table.cell(0, j).text = str(h)
    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            table.cell(i, j).text = str(val)
    set_table_borders_and_style(table, col_widths, font_pt=font_pt)
    sp = doc.add_paragraph()
    sp.paragraph_format.space_before = Pt(0)
    sp.paragraph_format.space_after = Pt(4)


def add_csv_table(doc: Document, tab_num: str, caption: str, csv_rel_path: str,
                  col_widths: Sequence[float], font_pt: float = 8.5,
                  rename_headers: Sequence[str] | None = None,
                  filter_fn=None, col_indices: Sequence[int] | None = None) -> None:
    p = REPO_ROOT / csv_rel_path
    with open(p, newline="", encoding="utf-8") as f:
        reader = list(csv.reader(f))
    raw_headers = reader[0]
    if col_indices is not None:
        raw_headers = [raw_headers[i] for i in col_indices]
    headers = rename_headers if rename_headers else raw_headers
    data_rows = []
    for r in reader[1:]:
        if filter_fn and not filter_fn(r):
            continue
        data_rows.append([r[i] for i in col_indices] if col_indices is not None else r)
    add_styled_table(doc, tab_num, caption, headers, data_rows, col_widths, font_pt=font_pt)


def add_formula_eq(doc: Document, eq_num: str, form_rel_path: str, symbol_note: str,
                   img_width_in: float = 4.4) -> None:
    img_path = REPO_ROOT / form_rel_path
    assert img_path.is_file(), f"Missing formula image: {img_path}"
    tbl = doc.add_table(rows=1, cols=3)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    tblPr = tbl._tbl.tblPr
    tblPr.append(parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        '<w:top w:val="none"/><w:bottom w:val="none"/>'
        '<w:left w:val="none"/><w:right w:val="none"/>'
        '<w:insideH w:val="none"/><w:insideV w:val="none"/>'
        "</w:tblBorders>"
    ))
    widths = [0.45, 4.65, 0.65]
    for idx, w in enumerate(widths):
        cell = tbl.cell(0, idx)
        cell.width = Inches(w)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER

    p_mid = tbl.cell(0, 1).paragraphs[0]
    p_mid.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_mid.paragraph_format.space_before = Pt(2)
    p_mid.paragraph_format.space_after = Pt(2)
    p_mid.paragraph_format.keep_with_next = True
    p_mid.add_run().add_picture(str(img_path), width=Inches(min(img_width_in, 4.55)))

    p_num = tbl.cell(0, 2).paragraphs[0]
    p_num.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r_num = p_num.add_run(f"({eq_num})")
    r_num.font.name = "Times New Roman"
    r_num.font.size = Pt(11.5)

    p_note = doc.add_paragraph()
    p_note.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_note.paragraph_format.line_spacing = 1.35
    p_note.paragraph_format.space_before = Pt(1)
    p_note.paragraph_format.space_after = Pt(6)
    r_w = p_note.add_run("where ")
    r_w.italic = True
    r_w.font.name = "Times New Roman"
    r_w.font.size = Pt(11.0)
    r_s = p_note.add_run(symbol_note)
    r_s.font.name = "Times New Roman"
    r_s.font.size = Pt(11.0)


def add_pseudocode_block(doc: Document, title: str, lines: Sequence[str]) -> None:
    p_t = doc.add_paragraph()
    p_t.paragraph_format.space_before = Pt(6)
    p_t.paragraph_format.space_after = Pt(2)
    p_t.paragraph_format.keep_with_next = True
    rt = p_t.add_run(title)
    rt.bold = True
    rt.font.name = "Times New Roman"
    rt.font.size = Pt(11.5)

    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    cell = tbl.cell(0, 0)
    cell.width = Inches(5.75)
    tcPr = cell._tc.get_or_add_tcPr()
    tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="F6F8FA"/>'))
    tbl._tbl.tblPr.append(parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        '<w:top w:val="single" w:sz="6" w:space="0" w:color="2B4C7E"/>'
        '<w:bottom w:val="single" w:sz="6" w:space="0" w:color="2B4C7E"/>'
        '<w:left w:val="single" w:sz="4" w:space="0" w:color="BBBBBB"/>'
        '<w:right w:val="single" w:sz="4" w:space="0" w:color="BBBBBB"/>'
        "</w:tblBorders>"
    ))
    p = cell.paragraphs[0]
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run("\n".join(lines))
    r.font.name = "Consolas"
    r.font.size = Pt(9.0)


def add_code_snippet(doc: Document, title: str, code_lines: Sequence[str]) -> None:
    assert len(code_lines) <= 40, f"Code snippet exceeds 40 lines ({len(code_lines)}): {title}"
    add_pseudocode_block(doc, title, code_lines)


# ---------------------------------------------------------------------------
# PRELIMINARY PAGES
# ---------------------------------------------------------------------------
def build_prelims(doc: Document) -> None:
    sec0 = doc.sections[0]
    configure_a4_margins(sec0)
    sec0.different_first_page_header_footer = True

    # 1. Cover Page
    p = add_para(doc, "UNIFIED MULTIMODAL AND REAL-TIME WEBCAM EMOTION RECOGNITION SYSTEM",
                 align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=16.0, space_after=8.0)
    add_para(doc, "A Major Project Report", align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=13.0, space_after=6.0)
    add_para(doc, "Submitted in partial fulfillment of the requirements for the degree of",
             align=WD_ALIGN_PARAGRAPH.CENTER, size=12.0, space_after=4.0)
    add_para(doc, "[TO BE FILLED: Degree Name, e.g., Bachelor of Technology in Computer Science and Engineering]",
             align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=12.0, space_after=14.0)

    add_para(doc, "Submitted by:", align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=12.0, space_after=4.0)
    add_para(doc, "[TO BE FILLED: Student Name 1] (Enrollment No.: [TO BE FILLED: Enrollment No. 1])\n"
                  "[TO BE FILLED: Student Name 2] (Enrollment No.: [TO BE FILLED: Enrollment No. 2])\n"
                  "[TO BE FILLED: Student Name 3] (Enrollment No.: [TO BE FILLED: Enrollment No. 3])",
             align=WD_ALIGN_PARAGRAPH.CENTER, size=12.0, space_after=14.0)

    add_para(doc, "Under the Supervision of:", align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=12.0, space_after=4.0)
    add_para(doc, "[TO BE FILLED: Supervisor Name and Designation]",
             align=WD_ALIGN_PARAGRAPH.CENTER, size=12.0, space_after=18.0)

    logo_p = REPO_ROOT / "Project_Report" / "Figures" / "jiit_logo.png"
    add_para(doc, "[TO BE FILLED: University Logo]", align=WD_ALIGN_PARAGRAPH.CENTER, italic=True, size=11.0, space_after=18.0)

    add_para(doc, "[TO BE FILLED: Department Name]\n[TO BE FILLED: Institute / University Name]\n[TO BE FILLED: Month and Year]",
             align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=12.5, space_after=6.0)

    # Section 1: Preliminary Pages (lowerRoman starting at i)
    sec1 = doc.add_section(WD_SECTION.NEW_PAGE)
    configure_a4_margins(sec1)
    set_bottom_center_page_number(sec1, fmt="lowerRoman", start=1)

    # 2. Certificate
    add_para(doc, "CERTIFICATE", align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=16.0, space_after=14.0)
    add_para(doc,
             "This is to certify that the project report entitled \"Unified Multimodal and Real-Time Webcam Emotion Recognition System\" "
             "submitted by [TO BE FILLED: Student Name 1] ([TO BE FILLED: Enrollment No. 1]), [TO BE FILLED: Student Name 2] "
             "([TO BE FILLED: Enrollment No. 2]), and [TO BE FILLED: Student Name 3] ([TO BE FILLED: Enrollment No. 3]) in partial "
             "fulfillment of the requirements for the award of the degree of [TO BE FILLED: Degree Name] at [TO BE FILLED: Institute Name] "
             "is an authentic record of the work carried out under my supervision and guidance.")
    add_para(doc,
             "The technical contents, software artifacts, experimental evaluations, and empirical results embodied in this report have not "
             "been submitted in part or in full to any other university or institution for the award of any degree or diploma.",
             space_after=36.0)

    sig_tbl = doc.add_table(rows=2, cols=2)
    sig_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    sig_tbl.autofit = False
    for r in sig_tbl.rows:
        for c in r.cells:
            c.width = Inches(2.85)
    sig_tbl.cell(0, 0).text = "______________________________\n[TO BE FILLED: Supervisor Name]\nProject Supervisor"
    sig_tbl.cell(0, 1).text = "______________________________\n[TO BE FILLED: Associate Head / HOD Name]\nAssociate Head / Head of Department"
    sig_tbl.cell(1, 0).text = "Date: [TO BE FILLED: Date]"
    sig_tbl.cell(1, 1).text = "Place: [TO BE FILLED: City]"
    for row in sig_tbl.rows:
        for cell in row.cells:
            for par in cell.paragraphs:
                par.paragraph_format.space_after = Pt(4)
                for run in par.runs:
                    run.font.name = "Times New Roman"
                    run.font.size = Pt(11.5)

    # 3. Declaration of Originality (PO8 Ethics)
    doc.add_page_break()
    add_para(doc, "DECLARATION OF ORIGINALITY", align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=16.0, space_after=14.0)
    add_para(doc,
             "We hereby declare that the project work presented in this report entitled \"Unified Multimodal and Real-Time Webcam Emotion "
             "Recognition System\" is our own original work carried out at [TO BE FILLED: Department Name], [TO BE FILLED: Institute Name]. "
             "In accordance with Program Outcome PO8 (Ethics), we affirm that this report is free from plagiarism, fabrication, or falsification "
             "of experimental data.")
    add_para(doc,
             "All third-party datasets (MELD, RAF-DB, and RAVDESS), open-source libraries (PyTorch, TorchVision, OpenCV, Transformers, "
             "FastAPI, Librosa, and scikit-learn), and pretrained models (ResNet-18, ResNet-50, EfficientNet-B0, YuNet, Wav2Vec 2.0, and "
             "all-MiniLM-L6-v2) have been appropriately cited in IEEE format. All reported metrics, confusion matrices, latency measurements, "
             "and failure analyses accurately reflect real execution logs and held-out evaluations, including honest disclosure of limitations "
             "and evaluation constraints.", space_after=24.0)
    add_para(doc,
             "1. [TO BE FILLED: Student Name 1] — Enrollment No.: [TO BE FILLED: Enrollment No. 1]   Signature: ______________\n"
             "2. [TO BE FILLED: Student Name 2] — Enrollment No.: [TO BE FILLED: Enrollment No. 2]   Signature: ______________\n"
             "3. [TO BE FILLED: Student Name 3] — Enrollment No.: [TO BE FILLED: Enrollment No. 3]   Signature: ______________\n\n"
             "Date: [TO BE FILLED: Date]\nPlace: [TO BE FILLED: City]",
             align=WD_ALIGN_PARAGRAPH.LEFT, size=12.0)

    # 4. Acknowledgement
    doc.add_page_break()
    add_para(doc, "ACKNOWLEDGEMENT", align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=16.0, space_after=14.0)
    add_para(doc,
             "We express our sincere gratitude to our project supervisor, [TO BE FILLED: Supervisor Name], [TO BE FILLED: Department Name], "
             "[TO BE FILLED: Institute Name], for continuous technical guidance, critical evaluation of experimental baselines, and constructive "
             "feedback throughout the development and empirical auditing of this project.")
    add_para(doc,
             "We extend our appreciation to [TO BE FILLED: Associate Head / HOD Name] and the faculty evaluation committee for their thoughtful "
             "review of system architecture, ethics compliance, and benchmarking methodology. We also acknowledge the creators and maintainers of "
             "the academic MELD, RAF-DB, and RAVDESS datasets, as well as the open-source PyTorch, OpenCV, Hugging Face, and FastAPI communities "
             "whose tools enabled strictly local reproducible experimentation.")
    add_para(doc,
             "Finally, we thank our families and peers for their encouragement and support throughout the duration of this academic project.")

    # 5. Abstract (1 page)
    doc.add_page_break()
    add_para(doc, "ABSTRACT", align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=16.0, space_after=12.0)
    add_para(doc,
             "Automated human emotion recognition requires modeling both spontaneous real-time facial expressions from live camera streams and "
             "multi-channel conversational cues spanning speech prosody, spoken text, and visual context. Existing standalone classifiers often "
             "suffer from train-serve preprocessing mismatches, severe domain shift between studio or television benchmarks and consumer webcams, "
             "uncalibrated softmax confidence, and fragility under class imbalance or missing modalities.")
    add_para(doc,
             "This project presents a unified local-inference Emotion Recognition platform comprising two integrated subsystems evaluated across "
             "a shared seven-class discrete emotion schema (anger, disgust, fear, joy, neutral, sadness, and surprise): (1) a real-time webcam "
             "facial expression engine powered by OpenCV YuNet face detection, eye-line affine alignment, Laplacian blur quality gating, and three "
             "selectable CNN backbones (EfficientNet-B0 at 96x96 pixels, RAF-DB ResNet-18 at 128x128 pixels, and a mixed RAF-DB + RAVDESS "
             "fine-tuned ResNet-18 checkpoint); and (2) a MELD multimodal conversational pipeline integrating Wav2Vec 2.0 Base acoustic embeddings "
             "with a balanced StrongerAudioClassifier MLP, contextual all-MiniLM-L6-v2 sentence embeddings, ResNet-50 temporal frame pooling, and an "
             "AdvancedCrossModalAttentionFusionNet with Shannon-entropy reliability gating and post-hoc temperature scaling.")
    add_para(doc,
             "Empirical evaluation demonstrates concrete progress alongside documented limitations. On 9 personal webcam evaluation clips (916 "
             "frames used for both backbone selection and reporting), unifying live and offline inference pipelines, removing hard-coded action-unit "
             "rules (which improved MELD validation macro-F1 from 0.2872 to 0.3868), and mixed fine-tuning on RAF-DB plus 3,970 aligned RAVDESS "
             "speech crops raised pooled frame accuracy from 15.6% (1/9 clips, efficientnet_current) to 44.0% (4/9 clips, resnet18_raf) and 57.3% "
             "(5/9 clips, resnet18_ravdess), while improving RAF-DB test macro-F1 from 0.6721 to 0.7077 and achieving 0.5354 macro-F1 on held-out "
             "RAVDESS actors. On a unified 200-frame CPU benchmark, all three backbones achieved ~31.5-33.6 ms median (p50) and ~33.3-35.8 ms p95 "
             "latency (~30 fps). On the 105-sample balanced MELD fusion test partition, cross-modal attention fusion with temperature scaling "
             "achieved 36.19% accuracy (0.3574 macro-F1) and reduced Expected Calibration Error (ECE) from 0.2130 to 0.0734. Automated verification "
             "passed 5/5 unit suites and 11/11 FastAPI integration tests.")
    add_para(doc,
             "From a societal and ethical standpoint, all webcam frames and microphone buffers execute strictly in volatile local memory without "
             "disk persistence, accompanied by explicit scientific disclaimers noting that apparent facial expressions do not equate to internal "
             "affective states.")
    add_para(doc, "Keywords: Multimodal Emotion Recognition, Facial Expression Recognition, MELD, RAF-DB, RAVDESS, Cross-Modal Attention, "
                  "Temperature Scaling, Domain Shift, FastAPI, Real-Time Computer Vision.", bold=False, italic=True, size=11.5)

    # 6. Table of Contents (Automatic Field)
    doc.add_page_break()
    add_para(doc, "TABLE OF CONTENTS", align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=16.0, space_after=10.0)
    p_toc = doc.add_paragraph()
    add_field(p_toc, 'TOC \\o "1-3" \\h \\z \\u', font_size=11.5)

    # 7. List of Figures (Automatic Field collecting FigureCaption style)
    doc.add_page_break()
    add_para(doc, "LIST OF FIGURES", align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=16.0, space_after=10.0)
    p_lof = doc.add_paragraph()
    add_field(p_lof, 'TOC \\h \\z \\t "FigureCaption,1"', font_size=11.0)

    # 8. List of Tables (Automatic Field collecting TableCaption style)
    doc.add_page_break()
    add_para(doc, "LIST OF TABLES", align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=16.0, space_after=10.0)
    p_lot = doc.add_paragraph()
    add_field(p_lot, 'TOC \\h \\z \\t "TableCaption,1"', font_size=11.0)

    # 9. List of Abbreviations
    doc.add_page_break()
    add_para(doc, "LIST OF ABBREVIATIONS", align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, size=16.0, space_after=10.0)
    abbr_rows = [
        ("API", "Application Programming Interface"),
        ("ASR", "Automatic Speech Recognition"),
        ("AU", "Facial Action Unit"),
        ("BGR / RGB", "Blue-Green-Red / Red-Green-Blue Color Channels"),
        ("BN", "Batch Normalization"),
        ("CE", "Cross-Entropy Loss"),
        ("CNN", "Convolutional Neural Network"),
        ("CPU", "Central Processing Unit"),
        ("ECE", "Expected Calibration Error"),
        ("EMA", "Exponential Moving Average"),
        ("ERC", "Emotion Recognition in Conversations"),
        ("FER", "Facial Expression Recognition"),
        ("FPS", "Frames Per Second"),
        ("GELU / ReLU / SiLU", "Gaussian / Rectified / Sigmoid Linear Unit Activations"),
        ("HTTP / WS", "Hypertext Transfer Protocol / WebSocket"),
        ("JSON", "JavaScript Object Notation"),
        ("LN", "Layer Normalization"),
        ("MELD", "Multimodal EmotionLines Dataset"),
        ("MLP", "Multi-Layer Perceptron"),
        ("NBA", "National Board of Accreditation"),
        ("NLL", "Negative Log-Likelihood"),
        ("ONNX", "Open Neural Network Exchange"),
        ("PO / PSO", "Program Outcome / Program Specific Outcome"),
        ("RAF-DB", "Real-world Affective Faces Database"),
        ("RAM", "Random Access Memory"),
        ("RAVDESS", "Ryerson Audio-Visual Database of Emotional Speech and Song"),
        ("REST", "Representational State Transfer"),
        ("SER", "Speech Emotion Recognition"),
        ("SPA", "Single-Page Application"),
        ("TF-IDF", "Term Frequency - Inverse Document Frequency"),
        ("WBS", "Work Breakdown Structure"),
    ]
    add_styled_table(doc, "0.1", "List of Abbreviations Used in the Report",
                     ["Abbreviation", "Full Technical Expansion"], abbr_rows, [1.55, 4.20], font_pt=10.0)


# ---------------------------------------------------------------------------
# CHAPTER 1: INTRODUCTION
# ---------------------------------------------------------------------------
def build_chapter1(doc: Document) -> None:
    sec2 = doc.add_section(WD_SECTION.NEW_PAGE)
    configure_a4_margins(sec2)
    set_bottom_center_page_number(sec2, fmt="decimal", start=1)

    add_chapter_heading(doc, "1", "Introduction", page_break=False)

    add_h2(doc, "1.1 Background of the Problem")
    add_para(doc,
             "Human affective communication operates simultaneously across visual, acoustic, and linguistic channels. Facial muscle "
             "configurations convey immediate apparent expressions, vocal prosody encodes arousal and pitch dynamics, and spoken words "
             "carry conversational intent and sentiment. Designing software systems capable of interpreting these cues reliably under "
             "real-world constraints requires addressing both single-modality computer vision challenges—such as face alignment, lighting "
             "variation, and sensor noise—and multimodal fusion challenges, including conflicting cues, class imbalance, and probability "
             "miscalibration.")
    add_para(doc,
             "Standard academic benchmarks isolate distinct aspects of this problem. The Multimodal EmotionLines Dataset (MELD) captures "
             "multi-party conversational utterances from television dialogue across synchronized audio, video, and text streams [1], while "
             "the Real-world Affective Faces Database (RAF-DB) offers unconstrained static facial photographs [2] and the Ryerson Audio-Visual "
             "Database of Emotional Speech and Song (RAVDESS) supplies controlled spoken emotional video clips [3]. However, models trained on "
             "one benchmark frequently degrade sharply when deployed on live consumer webcam streams due to domain shift and preprocessing "
             "divergence.")

    add_h2(doc, "1.2 Motivation and Need")
    add_para(doc,
             "Empirical auditing of the project baseline revealed four concrete failure modes that motivated a unified architectural rebuild. "
             "First, train-serve pipeline mismatch occurred when live REST/WebSocket webcam routes, evaluation scripts, and training loaders "
             "used inconsistent image crops, mirror transformations, or hand-crafted Action Unit (AU) heuristic overrides, degrading validation "
             "macro-F1 from 0.3868 down to 0.2872 [see results/webcam_fix_log.md]. Second, domain shift caused a MELD-trained EfficientNet-B0 "
             "webcam model (efficientnet_current) to achieve only 15.6% pooled frame accuracy (1/9 clips correct) on personal webcam clips "
             "and 20.4% accuracy on RAF-DB test images.")
    add_para(doc,
             "Third, severe class imbalance in MELD—where 47.1% of training utterances are neutral while disgust and fear each represent "
             "under 2.8%—caused baseline acoustic classifiers to collapse toward majority-class predictions (0.1771 macro-F1). Fourth, "
             "uncalibrated baseline multimodal fusion exhibited an Expected Calibration Error (ECE) of 0.2130, outputting overconfident "
             "probabilities on conflicting or noisy inputs. Addressing these bottlenecks necessitated a cohesive, locally deployable system "
             "combining shared preprocessing, mixed-domain fine-tuning, learned cross-modal attention fusion, and temperature calibration.")

    add_h2(doc, "1.3 Problem Statement")
    add_para(doc,
             "The objective of this project is to design, implement, and empirically validate a unified local-inference emotion recognition "
             "system covering both (a) a real-time webcam facial expression subsystem with unified preprocessing, quality gating, and selectable "
             "deep vision backbones, and (b) a MELD multimodal conversational analysis subsystem fusing acoustic (Wav2Vec 2.0), linguistic "
             "(MiniLM / TF-IDF), and visual (ResNet-50) streams across a common seven-emotion taxonomy (anger, disgust, fear, joy, neutral, "
             "sadness, and surprise), served through a FastAPI backend and browser-based studio interface.")

    add_h2(doc, "1.4 Project Objectives")
    add_para(doc, "To resolve the identified engineering gaps, five clear and measurable objectives were established:")
    add_bullet(doc, "Objective 1 — Pipeline Parity & Quality Gating: ",
               "Unify all webcam endpoints (/frame, /frame-json, /ws, /ws/webcam, /multimodal) and evaluation scripts on a single "
               "WebcamService.process_frame engine achieving 100% frame-level parity, unmirrored inference, eye-line affine alignment, "
               "and Laplacian blur gating (threshold 30.0).")
    add_bullet(doc, "Objective 2 — Cross-Domain Webcam Generalization: ",
               "Improve personal webcam facial expression accuracy beyond the MELD-trained EfficientNet-B0 baseline (15.6% pooled, 1/9 "
               "clips) and RAF-DB ResNet-18 baseline (44.0% pooled, 4/9 clips) via mixed RAF-DB + aligned RAVDESS fine-tuning while "
               "preserving RAF-DB macro-F1 (>= 0.6721) and maintaining CPU p95 inference latency <= 150 ms.")
    add_bullet(doc, "Objective 3 — Balanced MELD Unimodal Modeling: ",
               "Mitigate MELD minority-class collapse in acoustic speech emotion recognition (Wav2Vec 2.0 768-D) and evaluate contextual "
               "sentence embeddings (all-MiniLM-L6-v2) against TF-IDF baselines on all 2,610 held-out MELD test utterances.")
    add_bullet(doc, "Objective 4 — Calibrated Multimodal Attention Fusion: ",
               "Construct an attention-gated neural fusion architecture supporting missing-modality masking and post-hoc temperature "
               "calibration to reduce ECE below 0.10 on the held-out fusion test benchmark.")
    add_bullet(doc, "Objective 5 — Privacy-Preserving Full-Stack Verification: ",
               "Deploy the unified system through FastAPI REST/WebSocket endpoints and a 7-tab Web Studio with strictly volatile RAM "
               "processing for live streams and 100% pass verification across automated unit and integration tests.")

    add_h2(doc, "1.5 Social and Environmental Relevance")
    add_para(doc,
             "Affective interfaces have practical relevance in accessible human-computer interaction, pedagogical feedback tools, customer "
             "support triage, and non-clinical communication aids. At the same time, affective computing carries serious privacy and misuse "
             "risks because facial video and voice recordings constitute sensitive biometric personal data. Processing all live webcam "
             "frames and microphone audio strictly in volatile local RAM—without cloud uploads or persistent disk logging—directly mitigates "
             "surveillance and data-breach exposure, while explicit UI disclaimers clarify that visual outputs represent apparent expressions "
             "rather than internal psychological states.")
    add_para(doc,
             "From an environmental sustainability perspective, retraining massive vision-language models from scratch incurs substantial "
             "compute energy. This project minimizes energy consumption by reusing frozen pretrained encoders (Wav2Vec 2.0, MiniLM, ResNet-50), "
             "freezing early ResNet-18 stages during mixed fine-tuning (--freeze-until layer3), running lightweight 96 px and 128 px vision "
             "backbones directly on laptop CPU hardware without dedicated GPU clusters, and gating redundant frame processing.")

    add_h2(doc, "1.6 Project Plan")
    add_h3(doc, "1.6.1 Work Breakdown Structure (WBS)")
    add_para(doc,
             "Engineering execution was decomposed into six cohesive work packages covering data auditing, MELD unimodal modeling, cross-modal "
             "attention fusion, real-time webcam vision adaptation, FastAPI/UI integration, and automated QA verification, as illustrated in "
             "Figure 1.1.")
    add_figure(doc, "report/figures/fig_diag_wbs.png", "1.1",
               "Project Work Breakdown Structure (WBS) across six technical work packages.", width_in=5.65)

    add_h3(doc, "1.6.2 Project Timeline (Gantt Chart)")
    add_para(doc,
             "Engineering execution across the MELD multimodal and real-time webcam subsystems is organized chronologically into ten sequential "
             "and overlapping development phases measured in relative semester progression weeks (without fixed calendar dates). Figure 1.2 "
             "presents the project work breakdown timeline covering literature survey, dataset curation, unimodal and attention fusion modeling, "
             "webcam parity fixes and RAF-DB + RAVDESS ResNet-18 fine-tuning, FastAPI/Web Studio integration, automated verification, and reporting.")
    add_figure(doc, "report/figures/fig_gantt_git_timeline.png", "1.2",
               "Major Project Work Breakdown Timeline (Gantt chart across relative semester progression weeks, without calendar dates).", width_in=5.65)

    add_h3(doc, "1.6.3 Roles and Responsibilities")
    add_para(doc,
             "In alignment with collaborative engineering standards (PO9), project tasks were partitioned across team roles (detailed "
             "administrative assignments are provided in Annexure D with placeholders):")
    add_bullet(doc, "[TO BE FILLED: Student A — Multimodal & Backend Lead]: ",
               "Multimodal fusion architecture (AdvancedCrossModalAttentionFusionNet), temperature calibration, FastAPI routers, and test suite.")
    add_bullet(doc, "[TO BE FILLED: Student B — Webcam Vision & Domain Adaptation Lead]: ",
               "YuNet face tracking, affine alignment, RAVDESS crop preparation, mixed RAF-DB + RAVDESS fine-tuning, and parity verification.")
    add_bullet(doc, "[TO BE FILLED: Student C — Speech, NLP & Frontend Studio Lead]: ",
               "Wav2Vec 2.0 audio imbalance experiments, MiniLM text service, transcription indexing, Web Studio SPA, and reporting assets.")

    add_h3(doc, "1.6.4 Challenges and Risk Management")
    add_para(doc,
             "Throughout development, concrete technical risks were observed and logged in results/webcam_fix_log.md and system benchmarks. "
             "Table 1.1 summarizes the five primary project risks, their observed likelihood and impact, empirical evidence, and implemented "
             "mitigations.")
    add_csv_table(
        doc, "1.1",
        "Project Risk Register Derived from Engineering Logs and System Benchmarks",
        "report/tables/risk_register.csv",
        col_widths=[0.42, 0.95, 0.72, 0.65, 0.56, 1.20, 1.25],
        font_pt=7.8,
        rename_headers=["ID", "Risk", "Category", "Likelihood", "Impact", "Observed Evidence", "Mitigation Implemented"],
    )

    add_h2(doc, "1.7 Scope and Limitations")
    add_para(doc,
             "The system covers seven discrete emotion categories (anger, disgust, fear, joy, neutral, sadness, and surprise) across real-time "
             "webcam video frames and uploaded MELD-compatible audio, text, and video clips running locally on CPU hardware. Key limitations "
             "are acknowledged upfront: the 9 personal webcam clips feature a single subject performing acted expressions with highly correlated "
             "consecutive frames and served as both the backbone selection set and evaluation set; anger and sadness remain unsolved on those "
             "clips (0.0% accuracy); MELD multimodal ablation results reflect a balanced 105-sample test subset (15 per class); and no "
             "multi-speaker conversational context window is modeled across sequential MELD dialogue turns.")
    add_mapped_pos(doc, "PO1 (Engineering Knowledge), PO2 (Problem Analysis), PO6 (The Engineer and The World), "
                        "PO9 (Individual and Teamwork), PO10 (Project Management and Finance).")


# ---------------------------------------------------------------------------
# CHAPTER 2: LITERATURE SURVEY
# ---------------------------------------------------------------------------
def build_chapter2(doc: Document) -> None:
    add_chapter_heading(doc, "2", "Literature Survey", page_break=True)

    add_h2(doc, "2.1 Evolution of Facial Expression Recognition (FER)")
    add_para(doc,
             "Early facial expression recognition relied on posed laboratory corpora under controlled illumination, failing to generalize to "
             "unconstrained scenes. Li et al. [2] addressed this bottleneck by introducing RAF-DB, a crowdsourced in-the-wild facial expression "
             "benchmark filtered with Expectation-Maximization label refinement alongside Deep Locality-Preserving CNNs. Convolutional residual "
             "backbones such as ResNet [4] and compound-scaled lightweight architectures such as EfficientNet [5] subsequently became standard "
             "feature extractors for 7-class FER.")
    add_para(doc,
             "Recent 2021–2024 FER literature targets label noise, multi-region attention, and real-time edge inference. Wen et al. [6] "
             "proposed the Distract Your Attention Network (DAN), employing multi-head cross-attention across non-overlapping facial patches to "
             "achieve 89.70% accuracy on RAF-DB. Zhang et al. [7] introduced Erasing Attention Consistency (EAC) with ResNet-50 to suppress "
             "memorization of noisy labels, reaching 89.99%–90.35% on RAF-DB. Further accuracy gains came from landmark-guided transformer "
             "fusion in POSTER [8] and its streamlined successor POSTER++ [9], which achieved 92.21% on RAF-DB, as well as the lightweight "
             "Dual-Direction Attention Mixed Feature Network (DDAMFN) by Zhang et al. [10] reporting 91.35% on RAF-DB. For resource-constrained "
             "deployment, Savchenko [11], [12] developed HSEmotion and EmotiEffNets using multi-task pretraining on facial identity and attributes "
             "for fast edge feature extraction, paired with real-time anchor-free detectors such as YuNet (75.8K parameters, 5 facial landmarks) [13].")
    add_para(doc,
             "Critically, however, high accuracy on static RAF-DB test splits does not automatically transfer to live video or unseen cameras. "
             "Chen et al. [14] formalized Cross-Domain FER (CD-FER), demonstrating substantial accuracy degradation across dataset boundaries "
             "caused by camera optics, head pose, framing, and class-prior shifts. Similarly, dynamic speech corpora such as RAVDESS [3] expose "
             "neutral-motion and speaking-mouth ambiguities absent in static web-scraped smile photographs.")

    add_h2(doc, "2.2 Speech and Conversational Multimodal Emotion Recognition")
    add_para(doc,
             "In spoken and conversational settings, Poria et al. [1] established MELD (derived from EmotionLines and the television series "
             "Friends) as a primary multi-party trimodal benchmark featuring text transcripts, acoustic tracks, and video clips across seven "
             "emotions. For acoustic representation learning, self-supervised masked latent transformers—led by wav2vec 2.0 [15]—replaced "
             "hand-crafted prosodic descriptors. Pepino et al. [16] showed that layer-combined wav2vec 2.0 embeddings paired with shallow "
             "neural classifiers outperform traditional eGeMAPS features on RAVDESS and IEMOCAP, while Wagner et al. [17] proved in a large "
             "IEEE TPAMI study that fine-tuned speech transformers help close the historical acoustic valence gap.")
    add_para(doc,
             "In text and multimodal fusion on MELD, Transformer self-attention [18] and compact distilled sentence encoders such as MiniLM [19] "
             "and Sentence-BERT [20] provide dense semantic representations. Recent multimodal ERC architectures explicitly model cross-channel "
             "context and modality disparity: Hu et al. [21] proposed MM-DFN with dynamic graph fusion to curb inter-layer redundancy (62.49% "
             "accuracy, 59.46% weighted-F1 on MELD); Li et al. [22] introduced EmoCaps using emotion capsules (64.00% weighted-F1); Hu et al. [23] "
             "unified sentiment and emotion via UniMSE (65.51% accuracy, 65.91% weighted-F1); and Yun et al. [24] developed TelME, applying "
             "text-teacher cross-modal knowledge distillation to boost weak acoustic and visual streams to ~66.36%–67.37%.")

    add_h2(doc, "2.3 Uncertainty Calibration and Imbalanced Learning")
    add_para(doc,
             "Modern deep classifiers frequently exhibit severe overconfidence under distribution mismatch or class imbalance. Guo et al. [25] "
             "demonstrated that post-hoc single-parameter temperature scaling sharply reduces Expected Calibration Error (ECE) without altering "
             "decision boundaries, while Müller et al. [26] showed that label smoothing improves generalization and implicit calibration. To "
             "address extreme class skew (such as 47% neutral utterances in MELD), Lin et al. [27] proposed Focal Loss to down-weight well-"
             "classified majority samples, complementing inverse-frequency class-balanced sampling and decoupled weight decay optimization (AdamW) [28].")

    add_h2(doc, "2.4 Comparative Summary of Prior Works")
    add_para(doc,
             "Table 2.1 synthesizes key peer-reviewed works across FER, SER, multimodal ERC, and calibration, comparing methodology, datasets, "
             "and verified published metrics.")
    lit_rows = [
        ("Li et al. (2017) [2]", "DLP-CNN + EM crowdsourcing", "RAF-DB", "Foundational 7/11-class FER dataset", "Static in-the-wild only; no live stream"),
        ("Wen et al. (2023) [6]", "DAN (Multi-head cross-attn)", "RAF-DB", "89.70% Accuracy", "Static images; no cross-domain webcam eval"),
        ("Zhang et al. (2022) [7]", "EAC (ResNet-50 noisy-label reg.)", "RAF-DB", "89.99%–90.35% Accuracy", "Heavy ResNet-50; no temporal speech adaptation"),
        ("Zhang et al. (2023) [10]", "DDAMFN (MixConv + dual attn)", "RAF-DB", "91.35% Accuracy", "Static benchmark focus; no MELD multimodal fusion"),
        ("Mao et al. (2024) [9]", "POSTER++ (Landmark-image attn)", "RAF-DB", "92.21% Accuracy", "Dual-branch landmark transformer overhead"),
        ("Savchenko (2022–23) [11][12]", "HSEmotion / EmotiEffNets", "AffectNet, RAF-DB, ABAW", "~90% FER; edge real-time", "Pretrained weights; needs domain tuning for custom cam"),
        ("Chen et al. (2022) [14]", "AGRA (Cross-domain graph adapt.)", "CD-FER benchmarks", "Standardized CD-FER protocol", "Assumes target-domain adaptation data"),
        ("Wu et al. (2023) [13]", "YuNet anchor-free tiny detector", "WIDER FACE", "81.1% hard mAP (~1.6 ms CPU)", "Detection + 5 landmarks only (no expression head)"),
        ("Pepino et al. (2021) [16]", "Frozen Wav2Vec 2.0 + shallow net", "IEMOCAP, RAVDESS", "Outperformed eGeMAPS UAR", "Acoustic only; no text or visual fusion"),
        ("Wagner et al. (2023) [17]", "Fine-tuned Wav2Vec2 / HuBERT", "MSP-Podcast, IEMOCAP", "Closed acoustic valence gap", "Requires fine-tuned large speech transformer"),
        ("Poria et al. (2019) [1]", "MELD dataset + DialogueRNN base", "MELD (13k+ utt.)", "Multi-party A+T+V benchmark", "Severe 47% neutral skew; weak raw audio/video"),
        ("Hu et al. (2022) [21]", "MM-DFN (Dynamic graph fusion)", "MELD, IEMOCAP", "62.49% Acc / 59.46% W-F1 (MELD)", "Offline multi-turn dialogue graph; no live webcam"),
        ("Li et al. (2022) [22]", "EmoCaps (Emotion capsules)", "MELD, IEMOCAP", "64.00% W-F1 (MELD)", "Requires full conversational history; no calibration"),
        ("Hu et al. (2022) [23]", "UniMSE (Unified multimodal)", "MELD, MOSI, MOSEI", "65.51% Acc / 65.91% W-F1 (MELD)", "Heavy T5/multimodal pretraining; not CPU real-time"),
        ("Yun et al. (2024) [24]", "TelME (Text-teacher distill.)", "MELD, IEMOCAP", "~66.36%–67.37% (MELD)", "Offline conversation context; no webcam studio"),
        ("Guo et al. (2017) [25]", "Post-hoc Temperature Scaling", "Vision/NLP benchmarks", "Sharp ECE reduction", "Calibration only; requires held-out validation split"),
    ]
    add_styled_table(
        doc, "2.1", "Critical Comparison of Literature Works Across Methodology, Datasets, and Verified Benchmarks",
        ["Study", "Methodology", "Datasets", "Verified Reported Metric", "Identified Gap vs. Project Need"],
        lit_rows, [1.05, 1.25, 0.90, 1.20, 1.35], font_pt=7.8,
    )

    add_h2(doc, "2.5 Identification of Research Gaps")
    add_para(doc,
             "Synthesizing the reviewed literature reveals four critical gaps addressed by this project:")
    add_bullet(doc, "Gap 1 — Disconnect Between Offline Benchmarks and Live Webcam Inference: ",
               "High-scoring static RAF-DB transformers and offline MELD dialogue graphs rarely validate strict train-serve parity across "
               "face alignment, JPEG transport compression, mirror coordinates, and real-time CPU latency constraints.")
    add_bullet(doc, "Gap 2 — Cross-Domain Gap Between Static In-the-Wild Faces and Speaking Faces: ",
               "Models trained solely on static photographs (RAF-DB) or television clips (MELD) struggle on speaking, blinking, or "
               "resting-neutral webcam users unless adapted with aligned dynamic speech crops (such as RAVDESS) and sensor-realistic augmentations.")
    add_bullet(doc, "Gap 3 — Calibration and Modality-Absence Resilience in Lightweight Multimodal Fusion: ",
               "Many MELD systems assume all three modalities are always present and report uncalibrated softmax scores without entropy "
               "gating or conflict diagnostics.")
    add_bullet(doc, "Gap 4 — Privacy-First Local Execution: ",
               "Cloud-hosted emotion APIs transmit raw facial and vocal biometrics externally; an end-to-end local CPU architecture combining "
               "multimodal analysis with zero-disk-storage live webcam tracking addresses practical deployment constraints.")
    add_mapped_pos(doc, "PO2 (Problem Analysis), PO10 (Project Management and Finance), PO11 (Life-Long Learning).")


# ---------------------------------------------------------------------------
# CHAPTER 3: SYSTEM DESIGN AND METHODOLOGY
# ---------------------------------------------------------------------------
def build_chapter3(doc: Document) -> None:
    add_chapter_heading(doc, "3", "System Design and Methodology", page_break=True)

    add_h2(doc, "3.1 Engineering Design Constraints")
    add_para(doc,
             "Architectural decisions across both subsystems—the real-time webcam emotion engine and the MELD multimodal conversational pipeline—"
             "were governed by seven practical engineering constraints aligned with accreditation standards (PO3, PO5). Specifically, the solution had "
             "to operate on consumer laptop CPU hardware without discrete GPU acceleration, respect strict local biometric privacy, preserve full "
             "train-serve preprocessing parity, and avoid proprietary software dependencies. Table 3.1 summarizes how each constraint category "
             "shaped the design.")
    constraint_rows = [
        ("Economic", "Zero cloud API fees or paid GPU compute; open-source stack", "100% local CPU execution on existing Intel i7-12650H + 16 GB RAM"),
        ("Environmental & Energy", "Avoid training massive foundation models from scratch", "Frozen feature extractors (Wav2Vec2, MiniLM, ResNet-50) and partial ResNet-18 freeze (--freeze-until layer3)"),
        ("Social & Accessibility", "Readable browser UI with clear uncertainty and warnings", "Confidence tiers (Robust/Moderate/Subtle), explicit uncertain flagging (p<0.38 or margin<0.08)"),
        ("Political & Licensing", "Respect academic-only terms of MELD, RAF-DB, and RAVDESS", "Non-commercial research framing; no redistribution of raw third-party media in Git"),
        ("Ethical & Scientific", "Prevent overclaiming internal affect from external cues", "Apparent-expression terminology throughout UI and calibration diagnostics"),
        ("Health, Safety & Privacy", "Protect sensitive webcam video and voice streams", "Volatile RAM processing only on live path (/frame, /ws); explicit opt-in for /debug-capture"),
        ("Sustainability & Parity", "Eliminate divergence between offline scripts and live APIs", "Single shared WebcamService engine imported by REST, WebSocket, and eval_clips.py"),
    ]
    add_styled_table(
        doc, "3.1", "System Engineering Design Constraints and Architectural Responses",
        ["Constraint Dimension", "Design Requirement", "Implemented Architectural Decision"],
        constraint_rows, [1.20, 1.95, 2.60], font_pt=8.2,
    )

    add_h2(doc, "3.2 Overall System and Subsystem Architectures")
    add_para(doc,
             "Figure 3.1 depicts the top-level platform architecture hosted inside backend/main.py. HTTP and WebSocket traffic enters the FastAPI "
             "server on local loopback (127.0.0.1:8000), which mounts five service modules sharing a unified seven-class taxonomy (anger, disgust, "
             "fear, joy, neutral, sadness, surprise): AudioEmotionService, TextEmotionService, VideoEmotionService, MultimodalFusionService, and "
             "WebcamService.")
    add_figure(doc, "report/figures/fig_diag_system_architecture.png", "3.1",
               "Overall system architecture integrating MELD multimodal pipelines, WebcamService, FastAPI, and Web Studio UI.", width_in=5.65)

    add_para(doc,
             "Real-time webcam analysis follows the single-source pipeline in Figure 3.2. Incoming BGR frames pass through OpenCV YuNet face "
             "detection (face_detection_yunet_2023mar.onnx) to locate the primary bounding box and five landmarks (left eye, right eye, nose tip, "
             "and mouth corners). If landmarks satisfy confidence and minimum-size bounds, eye-line affine alignment centers and squares the crop "
             "before Laplacian blur gating rejects motion-blurred frames. Selectable CNN backbones (ResNet-18 RAVDESS, ResNet-18 RAF-DB, or "
             "EfficientNet-B0) produce raw logits refined by optional baseline calibration, confidence-adaptive EMA temporal smoothing, hysteresis "
             "stability filtering, and uncertainty checking prior to overlay.")
    add_figure(doc, "report/figures/fig_diag_live_inference.png", "3.2",
               "Live webcam inference pipeline from capture through YuNet alignment, quality gate, backbone, EMA, and uncertainty overlay.", width_in=5.65)

    add_para(doc,
             "To bridge the domain gap between static RAF-DB facial photographs [2] and live speaking faces without catastrophic forgetting of "
             "RAF-DB features, train_mixed.py implements the dual-source fine-tuning pipeline illustrated in Figure 3.3. Each training mini-batch "
             "blends balanced RAF-DB crops with aligned RAVDESS speech crops at ratio extra_share = 0.2 (26 RAF-DB + 6 RAVDESS frames per 32-sample "
             "batch) while freezing early residual stages through layer3 and training layer4 plus the linear classification head under differential "
             "AdamW learning rates.")
    add_figure(doc, "report/figures/fig_diag_training_pipeline.png", "3.3",
               "Mixed RAF-DB and RAVDESS ResNet-18 fine-tuning pipeline (extra_share=0.2, freeze-until layer3, best epoch 7).", width_in=5.60)

    add_para(doc,
             "Prior to mixed training, raw RAVDESS [3] videos pass through the automated preprocessing pipeline shown in Figure 3.4 (prepare_ravdess.py). "
             "Filename metadata strictly retains full audio-visual speech clips (modality=01, channel=01), drops calm (code 02) outside the 7-class "
             "taxonomy, samples the middle 30%–70% temporal window at 5 fps to avoid neutral onset/offset tails, and enforces the same YuNet aligned-"
             "crop and Laplacian blur quality gate used at live inference time.")
    add_figure(doc, "report/figures/fig_diag_ravdess_prep.png", "3.4",
               "RAVDESS speech video preparation flow (modality=01, channel=01, calm dropped, middle 30-70% at 5 fps, quality gate).", width_in=5.60)

    add_para(doc,
             "For MELD conversational multimodal analysis [1], Figure 3.5 shows the two-stage fusion architecture implemented in MultimodalFusionService "
             "and AdvancedCrossModalAttentionFusionNet. Each unimodal service outputs a 16-dimensional feature vector concatenating seven softmax "
             "probabilities (dims 0:7), seven raw logits (dims 7:14), a normalized Shannon-entropy certainty score (dim 14), and a modality quality score "
             "(dim 15). Modality projections feed a 2-layer Transformer encoder with missing-modality masking, followed by entropy-aware gating, weighted "
             "consensus logit interpolation, and learned post-hoc temperature scaling (T* = 0.5953).")
    add_figure(doc, "report/figures/fig_diag_meld_fusion.png", "3.5",
               "MELD AdvancedCrossModalAttentionFusionNet and MultimodalFusionService entropy gating and temperature calibration.", width_in=5.65)

    add_para(doc,
             "Finally, Figure 3.6 documents the REST and WebSocket deployment boundary exposed by FastAPI across backend/main.py and backend/api/*.py, "
             "connecting client browser sessions to unimodal endpoints (/predict/audio, /predict/text, /predict/video), multimodal upload routes "
             "(/predict/multimodal), and real-time webcam endpoints (/predict/webcam/frame, /frame-json, /ws, /backbone, /calibrate).")
    add_figure(doc, "report/figures/fig_diag_api_deployment.png", "3.6",
               "FastAPI deployment and REST/WebSocket endpoint architecture hosted locally on 127.0.0.1:8000.", width_in=5.65)

    add_h2(doc, "3.3 Core Algorithms (Pseudo-Code)")
    add_para(doc,
             "The three critical algorithms underlying live webcam parity, RAVDESS crop extraction, and mixed-domain training are specified below in "
             "structured pseudo-code corresponding directly to backend/services/webcam_service.py, prepare_ravdess.py, and train_mixed.py.")

    add_pseudocode_block(doc, "Algorithm 3.1: Unified Webcam Frame Processing & Uncertainty Gating (WebcamService.process_frame)", [
        "Input: BGR frame I, client_id c, mirror flag m, smooth flag s, calibrate flag cal",
        "1.  If m is True (legacy preview only; default m = False): flip I horizontally.",
        "2.  Run YuNet face detector on I -> select largest/highest-confidence face F = (x,y,w,h, re, le, nose, rm, lm, conf).",
        "3.  If F is empty or conf < 0.6 or min(w,h) < 40: return status = NO_FACE.",
        "4.  If alignment active: compute eye angle theta = atan2(ley - rey, lex - rex); if |theta| <= 45 deg,",
        "    rotate I around eye midpoint (cx, cy) and extract BORDER_REPLICATE square crop C of side max(w,h)*pad.",
        "5.  Evaluate Laplacian variance v = Var(Laplacian(Gray(C))); flag blurry warning if v < 30.0.",
        "6.  Resize C to active backbone resolution (128x128 for ResNet-18, 96x96 for EfficientNet-B0), RGB + ImageNet norm.",
        "7.  Forward active backbone -> logits z in R^7; update/subtract per-client neutral baseline z_base if cal active.",
        "8.  Compute softmax p_raw = softmax(z); apply adaptive EMA p_ema = alpha * p_raw + (1 - alpha) * p_prev if s is True.",
        "9.  Update hysteresis label tracker; check p_top1 < 0.38 or (p_top1 - p_top2) < 0.08 -> is_uncertain flag.",
        "10. Return JSON payload (emotion, tier, confidence, is_uncertain, probabilities, face_box, quality_metrics).",
    ])

    add_pseudocode_block(doc, "Algorithm 3.2: RAVDESS Dynamic Speech Crop Preparation & Quality Filtering (prepare_ravdess.py)", [
        "Input: Directory of RAVDESS MP4 files (1,200 total), target_fps = 5.0, blur_threshold = 30.0, actors val = {09, 10}",
        "1.  Parse 7-part RAVDESS filename mm-cc-ee-ii-ss-rr-aa.mp4.",
        "2.  Reject unless modality mm == '01' (full-AV) and vocal channel cc == '01' (speech); skip calm (ee == '02').",
        "3.  Map remaining 7 emotions -> canonical 7-class index; assign train (aa <= 08) or val (aa in {09, 10}).",
        "4.  Open video clip -> frame_count N, fps f; compute middle window [floor(0.30*N), ceil(0.70*N)] stride round(f/5.0).",
        "5.  For each sampled frame index t:",
        "    a. Detect YuNet face; reject if conf < 0.6 or w < 64 or h < 64.",
        "    b. Extract eye-aligned square crop via aligned_square_crop(..., pad=1.10/1.25, min_side=64).",
        "    c. Reject if Var(Laplacian(Gray(crop))) < 30.0.",
        "    d. Save PNG crop and append (crop_path, emotion_id, actor_id) row to train/val manifest CSV.",
        "Output: 518 retained clips -> 3,970 aligned crops (3,184 train Actors 01-08, 786 val Actors 09-10).",
    ])

    add_pseudocode_block(doc, "Algorithm 3.3: Mixed RAF-DB + RAVDESS Differential Fine-Tuning (train_mixed.py)", [
        "Input: ResNet-18 RAF checkpoint, RAF-DB train (10,428), RAVDESS train (3,184), extra_share = 0.2, batch_size = 32",
        "1.  Freeze ResNet-18 conv1, bn1, layer1, layer2, layer3; keep layer4 and linear fc trainable; freeze early BN stats.",
        "2.  Configure AdamW (wd = 1e-4) with differential LRs: backbone_lr = 2e-5 for layer4, head_lr = 2e-4 for fc.",
        "3.  Set per-batch mix quotas: n_extra = round(32 * 0.2) = 6 RAVDESS crops, n_raf = 32 - 6 = 26 RAF-DB crops.",
        "4.  Construct inverse-frequency class-balanced WeightedRandomSampler separately for raf_loader and extra_loader.",
        "5.  For epoch = 1 .. max_epochs:",
        "    a. Iterate paired mini-batches, concatenate (26 + 6 = 32), compute CrossEntropyLoss(label_smoothing = 0.1).",
        "    b. Backpropagate, step AdamW, evaluate Macro-F1 independently on RAF-DB val (1,841) and RAVDESS val (786).",
        "    c. Compute composite selection score = 0.5 * (macro_f1_raf_val + macro_f1_ravdess_val); checkpoint if best (ep 7).",
        "Output: best_resnet18_ravdess.pt + embedded preprocessing metadata (128 px, RGB ImageNet normalization).",
    ])

    add_h2(doc, "3.4 Mathematical Modelling")
    add_para(doc,
             "Every mathematical formulation implemented across training, preprocessing, multimodal fusion, calibration, and evaluation in this "
             "project is presented below in standardized equation numbering (3.1)–(3.22), alongside its physical symbol definitions and code location.")

    add_formula_eq(doc, "3.1", "report/formulas/F1_softmax.png",
                   "z in R^7 denotes raw class logits, c indexing the 7 emotions, producing calibrated probabilities sum_k p_k = 1 "
                   "(webcam_service.py:765, audio_service.py:142, video_service.py:141).")
    add_formula_eq(doc, "3.2", "report/formulas/F2_cross_entropy_label_smoothing.png",
                   "y is the ground-truth label index, C=7, eps=0.1 is the label-smoothing factor (0.05 in MELD fusion) regularizing target "
                   "distributions against noisy labels [26] (train_mixed.py:686,768).")
    add_formula_eq(doc, "3.3", "report/formulas/F3_focal_loss.png",
                   "alpha_y weights inverse class frequency, p_y is predicted true-class probability, and gamma=2.0 down-weights easy majority "
                   "samples under MELD/webcam imbalance [27] (audio_model.py:41-59, train_webcam_phase3.py:163-192).")
    add_formula_eq(doc, "3.4", "report/formulas/F6_class_balanced_weight.png",
                   "N is dataset size, C=7, N_c is sample count of class c, yielding sampling weights w_c inversely proportional to class rarity "
                   "(train_mixed.py:424-431).")
    add_formula_eq(doc, "3.5", "report/formulas/F7_input_normalization.png",
                   "x_RGB in [0,1] are RGB channels normalized by ImageNet mean mu=[0.485,0.456,0.406] and std sigma=[0.229,0.224,0.225] "
                   "(train_mixed.py:383, live_resnet18_demo.py:44, webcam_service.py:137).")
    add_formula_eq(doc, "3.6", "report/formulas/F9_conv_relu.png",
                   "W and b denote 2D convolutional filter kernels and biases applied over spatial receptive fields followed by ReLU nonlinearity "
                   "(torchvision ResNet-18, audio_model.py:13).")
    add_formula_eq(doc, "3.7", "report/formulas/F10_batch_norm.png",
                   "mu_B and sigma_B^2 are mini-batch feature moments regularized by learnable affine parameters gamma, beta (frozen up to layer3 "
                   "during mixed ResNet-18 fine-tuning; active in StrongerAudioClassifier; train_mixed.py:465-474).")
    add_formula_eq(doc, "3.8", "report/formulas/F8_resnet_residual_block.png",
                   "F(x, {W_i}) denotes the two-layer residual mapping and W_s an optional 1x1 projection shortcut for identity skip-connection learning [4] "
                   "(train_mixed.py:29, video_service.py:33).")
    add_formula_eq(doc, "3.9", "report/formulas/F18_efficientnet_compound_scaling.png",
                   "phi uniformly scales network depth d, width w, and input resolution r under resource factor alpha * beta^2 * gamma^2 approx 2 [5] "
                   "for efficientnet_current (96x96 px, webcam_service.py:140-144).")
    add_formula_eq(doc, "3.10", "report/formulas/F11_adamw_update.png",
                   "m_t, v_t are first/second gradient moments, eta is differential learning rate (2e-5 backbone / 2e-4 head), and lambda=1e-4 is "
                   "decoupled weight decay [28] (train_mixed.py:477-489).")
    add_formula_eq(doc, "3.11", "report/formulas/F12_mixed_batch_composition.png",
                   "B=32 total mini-batch size and rho=0.2 (extra_share) enforce n_extra=6 RAVDESS and n_raf=26 RAF-DB samples per step "
                   "(train_mixed.py:618-621).")
    add_formula_eq(doc, "3.12", "report/formulas/F13_aligned_square_crop.png",
                   "(rex, rey) and (lex, ley) are YuNet eye coordinates defining rotation angle theta and padded crop side s=max(w,h)*pad "
                   "(live_resnet18_demo.py:49-81, webcam_service.py:585-586).")
    add_formula_eq(doc, "3.13", "report/formulas/F14_blur_quality_gate.png",
                   "nabla^2 I_gray is the 2D Laplacian operator on grayscale crop C whose spatial variance must meet threshold tau_blur=30.0 "
                   "(prepare_ravdess.py:131, webcam_service.py:592-593).")
    add_formula_eq(doc, "3.14", "report/formulas/F19_ema_smoothing.png",
                   "p_t is current softmax vector and alpha_t scales dynamically with prediction confidence to suppress single-frame jitter "
                   "(webcam_service.py:770-775).")
    add_formula_eq(doc, "3.15", "report/formulas/F16_uncertain_rule.png",
                   "p_(1) and p_(2) are top-1 and top-2 probabilities; outputs with p_(1) < 0.38 or margin (p_(1) - p_(2)) < 0.08 trigger "
                   "uncertain flagging (webcam_service.py:832-846).")
    add_formula_eq(doc, "3.16", "report/formulas/F20_meld_audio_pooling_mlp.png",
                   "H_wav in R^(T x 768) is mask-mean-pooled over valid acoustic frames and classified via GELU/LayerNorm/Dropout MLP [15][16] "
                   "(audio_service.py:130-142, audio_model.py:21-39).")
    add_formula_eq(doc, "3.17", "report/formulas/F21_meld_fusion_equations.png",
                   "u_m in R^16 are audio/text/video feature vectors encoded by masked Transformer self-attention [18], combined via entropy-"
                   "weighted gates g_m, consensus logits, and temperature T*=0.5953 (fusion_model.py:172-216, fusion_service.py:112,249-262).")
    add_formula_eq(doc, "3.18", "report/formulas/F22_calibration_metrics.png",
                   "B_m are M=15 confidence bins bounding Observed Accuracy vs Mean Confidence gap (ECE) and squared error (Brier) [25] "
                   "(backend/utils/metrics.py:10-48).")
    add_formula_eq(doc, "3.19", "report/formulas/F4_classification_metrics.png",
                   "TP_c, FP_c, FN_c compute per-class precision, recall, F1, and unweighted Macro-F1 across all C=7 emotion classes "
                   "(train_mixed.py:492-505, train_webcam_phase3.py:212-232).")
    add_formula_eq(doc, "3.20", "report/formulas/F5_confusion_matrix.png",
                   "M_ij counts true class i predicted as j, and M~_ij normalizes each row by total true class support (train_mixed.py:515-521, "
                   "backend/utils/metrics.py:55).")
    add_formula_eq(doc, "3.21", "report/formulas/F15_majority_and_pooled_acc.png",
                   "Acc_pooled evaluates aggregate frame accuracy over all N=916 clip frames, whereas majority vote checks clip modal class "
                   "(eval_clips.py:113-115).")
    add_formula_eq(doc, "3.22", "report/formulas/F17_percentile_latency.png",
                   "L_(k) are ordered per-frame CPU millisecond timings on k=1..200 frames giving median p50, tail p95, and throughput FPS = 1000/p50 "
                   "(eval_clips.py:207-208).")

    add_h2(doc, "3.5 Tool and Technique Justification")
    add_para(doc,
             "Each model family, signal encoder, and software framework was selected based on empirical CPU latency, licensing openness, and "
             "reproducibility across PyTorch [29] and OpenCV runtimes, as structured in Table 3.2.")
    tool_rows = [
        ("YuNet (face_detection_yunet_2023mar.onnx) [13]", "OpenCV 5.1 DNN", "75.8K-param anchor-free detector outputting bounding box + 5 landmarks (eyes, nose, mouth) required for affine alignment at CPU speed."),
        ("ResNet-18 (128x128 RGB) [4]", "TorchVision", "11.18M parameters; residual skip connections support partial fine-tuning (--freeze-until layer3) with 31.5-33.0 ms end-to-end p50 latency."),
        ("EfficientNet-B0 (96x96 RGB) [5]", "TorchVision", "Legacy MELD-pretrained webcam baseline retained inside runtime switcher to demonstrate domain-shift degradation vs RAF/RAVDESS."),
        ("Wav2Vec 2.0 Base (768-D) + StrongerMLP [15][16]", "Transformers", "Self-supervised speech encoder capturing pitch/prosody over hand-crafted MFCCs; StrongerMLP with WeightedRandomSampler raises Macro-F1."),
        ("all-MiniLM-L6-v2 (384-D) [19][20]", "Sentence-Transformers", "Compact 384-D semantic encoder outperforming sparse 12k TF-IDF features on MELD text utterances (42.87% vs 40.15% Acc)."),
        ("ResNet-50 (8-frame temporal pool) [4]", "TorchVision", "Extracts 2048-D visual representations for offline MELD uploaded video clips prior to VideoClassifier and 16-D fusion projection."),
        ("PyTorch 2.14.0+cpu & FastAPI 0.162 [29]", "Core Runtime", "Unified autograd/checkpoint serialization plus async ASGI REST and WebSocket endpoints serving the zero-build Web Studio SPA."),
    ]
    add_styled_table(
        doc, "3.2", "Technical Justification of Models, Signal Encoders, and Software Frameworks",
        ["Component / Technology", "Ecosystem", "Engineering Justification"],
        tool_rows, [1.60, 1.05, 3.10], font_pt=8.0,
    )

    add_mapped_pos(doc, "PO3 (Design/Development of Solutions), PO5 (Modern Tool Usage).")


# ---------------------------------------------------------------------------
# CHAPTER 4: IMPLEMENTATION AND TESTING
# ---------------------------------------------------------------------------
def build_chapter4(doc: Document) -> None:
    add_chapter_heading(doc, "4", "Implementation and Testing", page_break=True)

    add_h2(doc, "4.1 Step-by-Step System Development")
    add_para(doc,
             "Implementation progressed across seven documented milestones audited in results/webcam_fix_log.md and repository commits, converting "
             "fragmented experimental scripts into an integrated production-tested application.")

    add_h3(doc, "4.1.1 Pipeline Parity Fix Between Live Webcam Endpoints and Offline Evaluation")
    add_para(doc,
             "Initial testing uncovered silent discrepancies between browser live frames, base64 /frame-json requests, WebSocket streams, and "
             "offline clip scripts. All routes (/predict/webcam/frame, /frame-json, /ws, /ws/webcam, /multimodal) plus eval_clips.py were "
             "refactored onto a single shared engine (WebcamService.process_frame). Simulating WebRTC browser transport at 320x180 JPEG quality 85 "
             "confirmed exact 916/916 frame parity (100.0% label agreement) between HTTP upload inference and local disk evaluation.")

    add_h3(doc, "4.1.2 Unmirrored Inference and Geometric Quality Gating")
    add_para(doc,
             "Browser selfie canvases frequently mirror video frames for user comfort, which altered asymmetrical expression gradients during "
             "classification. Inference defaults were fixed to mirror=False in WebcamService while keeping CSS visual mirroring independent. "
             "YuNet 5-point landmarks enabled in-plane eye-line affine alignment via aligned_square_crop, rejecting faces with roll |theta| > 45 "
             "degrees to fallback upright bounds, side < 64 px, or Laplacian blur variance < 30.0.")

    add_h3(doc, "4.1.3 Removal of Hard-Coded Action Unit Rules")
    add_para(doc,
             "Early webcam heuristics applied rule-based facial geometry overrides that suppressed legitimate model posteriors. Phase-2 ablation "
             "(results/phase2_ablations.json) verified that removing these hard-coded Action Unit rules raised MELD validation macro-F1 from "
             "0.2872 to 0.3868 (+0.0996), confirming pure learned feature decoding.")

    add_h3(doc, "4.1.4 Dynamic Multi-Backbone Integration")
    add_para(doc,
             "WebcamService was extended with a dynamic backbone switcher (POST /predict/webcam/backbone) supporting three registered models: "
             "resnet18_ravdess (default), resnet18_raf, and efficientnet_current. Each checkpoint embeds its preprocessing dictionary (input size, "
             "eval_resize=146/CenterCrop=128 vs 96x96, RGB ImageNet normalization), preventing stale image resolution mismatches upon hot-swapping.")

    add_h3(doc, "4.1.5 RAVDESS Dynamic Crop Extraction and Mixed Fine-Tuning")
    add_para(doc,
             "Using prepare_ravdess.py, 1,200 RAVDESS files were filtered down to 518 speech clips (skipping 600 non-AV/song clips, 80 calm clips, "
             "and 2 clips lacking valid faces), yielding 3,970 eye-aligned quality-gated crops split cleanly by actor identity (3,184 train from "
             "Actors 01–08; 786 val from Actors 09–10). Mixed training (train_mixed.py) initialized from best_resnet18.pt, froze stages up to "
             "layer3, blended 26 RAF-DB + 6 RAVDESS balanced crops per 32-sample batch, and peaked at Epoch 7 (best_resnet18_ravdess.pt).")

    add_h3(doc, "4.1.6 MELD Unimodal Improvement, Fusion Optimization, and Calibration")
    add_para(doc,
             "In the MELD subsystem, acoustic collapse toward majority neutral was mitigated using StrongerAudioClassifier (768->512->256->7 with "
             "LayerNorm and GELU) and inverse-frequency WeightedRandomSampler. Contextual MiniLM embeddings replaced TF-IDF with CalibratedClassifierCV. "
             "Finally, AdvancedCrossModalAttentionFusionNet (16-D stream inputs, d_model=96, 2 Transformer encoder layers, 4 heads) was trained for "
             "45 epochs and temperature-calibrated (T* = 0.5953) via L-BFGS optimization.")

    add_h2(doc, "4.2 Web Studio Frontend Walkthrough")
    add_para(doc,
             "The frontend (frontend/index.html and frontend/js/app.js) implements a responsive Single-Page Application requiring zero Node/npm build "
             "steps. All eight primary operational interfaces were captured against the live running local server at 1280x800 resolution.")

    add_para(doc,
             "Figure 4.1 shows the Unified Multimodal Dashboard landing page, providing file upload slots for synchronized video, audio, and "
             "text inputs alongside quick-load MELD sample presets (such as Friends dialogue dia0_utt0).")
    add_figure(doc, "report/figures/ui_home_landing.png", "4.1",
               "Web Studio landing view (Unified Multimodal Dashboard) rendered at 1280x800 resolution.", width_in=5.35)

    add_para(doc,
             "Figure 4.2 displays the Real-Time Webcam Detection Studio exposing the runtime backbone selector (defaulting to ResNet-18 RAF+RAVDESS), "
             "25-frame neutral calibration trigger, uncertainty indicator, and live WebSocket/REST streaming controls. Note: because the verification "
             "environment had no attached physical camera, this screenshot depicts the live studio in verified STANDBY mode (Camera: Inactive), "
             "while frame-level accuracy metrics were evaluated on recorded evaluation MP4 clips.")
    add_figure(doc, "report/figures/ui_live_webcam.png", "4.2",
               "Real-Time Webcam Detection Studio in verified STANDBY mode showing active backbone selector and diagnostic pills.", width_in=5.35)

    add_para(doc,
             "Figure 4.3 illustrates the Video Upload Clip Studio backed by ResNet-50 8-frame temporal pooling on dia0_utt0.mp4, returning the top "
             "predicted class alongside seven-emotion probability bars and frame quality metrics.")
    add_figure(doc, "report/figures/ui_upload_clip_video.png", "4.3",
               "Video Upload Clip Studio (ResNet-50 8-frame temporal pooling) prediction view on dia0_utt0.mp4.", width_in=5.35)

    add_para(doc,
             "Figure 4.4 presents the Benchmarks & Ablation research view, surfacing the stored MELD 7-configuration ablation table and temperature "
             "calibration reliability plot directly inside the browser.")
    add_figure(doc, "report/figures/ui_results_benchmarks.png", "4.4",
               "Benchmarks & Ablation UI tab displaying MELD configuration comparison and reliability curve.", width_in=5.35)

    add_para(doc,
             "Figure 4.5 demonstrates a completed Unified Multimodal prediction on MELD preset dia0_utt0, visualizing modality contribution weights "
             "(Audio, Text, Video) from AdvancedCrossModalAttentionFusionNet alongside conflict and consensus diagnostics.")
    add_figure(doc, "report/figures/ui_multimodal_unified.png", "4.5",
               "Unified Multimodal Emotion Analysis output for MELD preset dia0_utt0 showing attention gates and diagnostics.", width_in=5.35)

    add_para(doc,
             "Figure 4.6 shows the Acoustic Modality Studio processing dia0_utt0.wav (16 kHz) through Wav2Vec 2.0 Base and StrongerAudioClassifier.")
    add_figure(doc, "report/figures/ui_audio_studio.png", "4.6",
               "Acoustic Modality Studio (Wav2Vec2 + balanced StrongerMLP) prediction view on dia0_utt0.wav.", width_in=5.35)

    add_para(doc,
             "Figure 4.7 depicts the Linguistic Modality Studio evaluating arbitrary text prompts through the contextual all-MiniLM-L6-v2 pipeline.")
    add_figure(doc, "report/figures/ui_text_studio.png", "4.7",
               "Linguistic Modality Studio (all-MiniLM-L6-v2 + CalibratedClassifierCV) prediction interface.", width_in=5.35)

    add_para(doc,
             "Figure 4.8 displays the Session Prediction History and Audit Log view stored strictly client-side in session memory for review or export.")
    add_figure(doc, "report/figures/ui_history.png", "4.8",
               "Session Prediction History and Audit Log view inside the Web Studio UI.", width_in=5.35)

    add_h2(doc, "4.3 Experimental Setup and Training Hyperparameters")
    add_para(doc,
             "All training, benchmarking, and server tests were conducted on the CPU workstation detailed in Table 4.1 (verified via platform, "
             "Win32 CIM queries, and active .venv package inspection), and exact model hyperparameters across all checkpoints are consolidated in "
             "Table 4.2.")
    add_csv_table(
        doc, "4.1", "Verified Hardware and Software Execution Environment Specifications",
        "report/tables/environment.csv",
        col_widths=[0.85, 1.35, 2.35, 1.20], font_pt=8.2,
    )
    add_csv_table(
        doc, "4.2", "Training Hyperparameters Across Webcam Backbones and MELD Multimodal Fusion",
        "report/tables/hyperparameters.csv",
        col_widths=[0.95, 1.05, 0.90, 0.95, 0.90, 1.00], font_pt=7.4,
    )

    add_h2(doc, "4.4 FastAPI REST and WebSocket Endpoints")
    add_para(doc,
             "Table 4.3 inventories all 17 REST and WebSocket endpoints registered across backend/main.py and backend/api/*.py, together with input "
             "schemas and structured JSON response contracts.")
    add_csv_table(
        doc, "4.3", "Complete List of FastAPI REST and WebSocket Endpoints",
        "report/tables/api_endpoints.csv",
        col_widths=[0.50, 1.15, 0.85, 0.95, 0.95, 1.35], font_pt=7.4,
    )

    add_h2(doc, "4.5 Automated Test Verification (Input vs. Expected vs. Actual)")
    add_para(doc,
             "End-to-end verification was executed via .\\.venv\\Scripts\\python.exe tests\\run_tests.py --all, covering 5 unit test suites (22 checks "
             "within test_webcam.py plus audio, text, video, and fusion modules) and 11 live ASGI FastAPI integration requests. All 16 top-level test "
             "groups passed without skipped assertions, as itemized in Table 4.4.")
    add_csv_table(
        doc, "4.4", "Automated Unit and Integration Test Cases (tests/run_tests.py --all, 16/16 PASS)",
        "report/tables/test_cases.csv",
        col_indices=[0, 1, 2, 3, 4, 5],
        rename_headers=["Suite", "Test Name", "Test Input", "Expected Output", "Actual Output", "Status"],
        col_widths=[0.70, 1.05, 1.15, 1.25, 1.15, 0.45], font_pt=7.3,
    )

    add_h2(doc, "4.6 Safety, Privacy, and Cybersecurity Measures")
    add_para(doc,
             "Because affective analysis processes sensitive biometric video and voice streams, the platform enforces five explicit security and "
             "privacy controls in code:")
    add_bullet(doc, "Strictly Local Loopback Binding: ",
               "FastAPI runs locally on 127.0.0.1:8000 with zero outbound telemetry, external cloud inference calls, or third-party analytics.")
    add_bullet(doc, "Volatile In-Memory Live Webcam Processing: ",
               "Live /predict/webcam/frame, /frame-json, and WebSocket routes decode numpy frame arrays strictly in volatile RAM and discard them "
               "immediately after returning JSON metadata; diagnostic frame capture occurs only on explicit POST /predict/webcam/debug-capture opt-in.")
    add_bullet(doc, "Temporary Upload Cleanup & Input Validation: ",
               "Uploaded audio/video files use isolated temporary handles validated for format/non-empty payloads (blank text or empty forms return "
               "HTTP 400 Bad Request, as verified by Integration Tests 04 and 11) and are deleted after inference.")
    add_bullet(doc, "Repository Hygiene (.gitignore): ",
               "Raw datasets (DATASET/, MELD.Raw/, ravdess_crops/), private evaluation webcam clips (New folder/), live debug dumps, and large binary "
               "checkpoints are excluded from Git tracking to prevent inadvertent biometric data leaks.")
    add_bullet(doc, "Ephemeral Client Calibration State: ",
               "Neutral baseline logits and rolling EMA buffers are isolated per client_id in volatile server memory and can be cleared instantly via "
               "POST /reset-calibration or POST /reset-session.")

    add_mapped_pos(doc, "PO3 (Design/Development of Solutions), PO4 (Conduct Investigations of Complex Problems), "
                        "PO5 (Modern Tool Usage), PO6 (The Engineer and The World).")


# ---------------------------------------------------------------------------
# CHAPTER 5: RESULTS AND DISCUSSION
# ---------------------------------------------------------------------------
def build_chapter5(doc: Document) -> None:
    add_chapter_heading(doc, "5", "Results and Discussion", page_break=True)

    add_h2(doc, "5.1 Dataset Composition and Evaluation Protocol")
    add_para(doc,
             "Experimental evaluation covers four distinct benchmarks unified under the shared seven-emotion taxonomy (anger, disgust, fear, joy, "
             "neutral, sadness, surprise), as summarized in Table 5.1 and Figure 5.1. Because MELD [1] exhibits severe majority skew (47.1% "
             "neutral utterances vs. <2.8% disgust and fear), full-test unimodal scoring (N = 2,610) reports Macro-F1 alongside Accuracy, while "
             "multimodal fusion evaluation uses a strictly class-balanced 105-utterance held-out subset (15 per class, random chance = 14.3%). "
             "For vision training, two exact within-train RGB duplicates were removed from RAF-DB [2] (10,428 train / 1,841 val / 3,068 test), "
             "and 3,970 aligned speech crops were extracted from 518 RAVDESS [3] clips split strictly by speaker identity (Actors 01–08 train; "
             "Actors 09–10 validation).")
    add_figure(doc, "report/figures/fig_class_dist_datasets.png", "5.1",
               "Class distributions across splits for RAF-DB, extracted RAVDESS speech crops, and MELD utterances.", width_in=5.55)
    add_csv_table(
        doc, "5.1", "Summary of Datasets Across Training, Validation, and Held-Out Evaluation",
        "report/tables/datasets_summary.csv",
        col_widths=[0.95, 1.05, 0.95, 1.15, 0.70, 0.95], font_pt=7.7,
    )

    add_h2(doc, "5.2 Webcam Vision Progression, Confusion Matrices, and CPU Latency")
    add_para(doc,
             "Table 5.2 and Figure 5.2 summarize the quantitative progression across all three selectable webcam backbones. Replacing the legacy "
             "MELD-pretrained EfficientNet-B0 baseline (efficientnet_current: 20.4% RAF-DB test acc; 15.61% pooled webcam acc, 1/9 clips) with "
             "ResNet-18 trained on RAF-DB (resnet18_raf) lifted RAF-DB test accuracy to 79.53% (0.6721 macro-F1) and personal webcam clip "
             "accuracy to 44.00% (4/9 clips). Mixed fine-tuning on RAF-DB blended with aligned RAVDESS speech crops (resnet18_ravdess, best "
             "Epoch 7) achieved the strongest overall balance: improving RAF-DB test macro-F1 further to 0.7077 (79.27% acc), reaching 56.87% "
             "accuracy (0.5354 macro-F1) on unseen RAVDESS Actors 09–10, and raising personal webcam performance to 57.31% pooled (525/916 "
             "frames, 5/9 clips)—cutting surprise-vs-anger errors (101 down to 62 frames) and neutral-vs-surprise errors (82 down to 51).")
    add_csv_table(
        doc, "5.2", "Summary Comparison of Webcam Backbones Across RAF-DB, RAVDESS, Personal Clips, and Unified CPU Latency",
        "report/tables/backbone_comparison.csv",
        col_indices=[0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        rename_headers=["Backbone", "Arch.", "Px", "RAF Acc", "RAF F1", "RAV Acc", "RAV F1", "Clips", "Pooled", "p50 ms", "p95 ms"],
        col_widths=[0.88, 0.62, 0.34, 0.48, 0.48, 0.48, 0.48, 0.45, 0.50, 0.52, 0.52], font_pt=7.0,
    )
    add_figure(doc, "report/figures/fig_webcam_backbone_comparison.png", "5.2",
               "Per-clip frame accuracy, pooled accuracy, and majority-vote clip score across all three webcam backbones (9 clips, 916 frames).", width_in=5.55)

    add_para(doc,
             "Figures 5.3, 5.4, and 5.5 inspect the confusion matrices of the promoted default model (resnet18_ravdess) across all three vision "
             "domains. On RAF-DB test (Figure 5.3, N = 3,068), strong diagonal recall confirms mixed fine-tuning preserved general facial "
             "features without catastrophic forgetting. On held-out RAVDESS speakers (Figure 5.4, N = 786), dynamic speaking faces separate cleanly "
             "on joy, disgust, anger, and surprise, while Figure 5.5 and Table 5.3 confirm 100.0% frame accuracy on Disgust (2).mp4 (172/172), "
             "Joy.mp4 (115/115), and Neutal.mp4 (99/99), plus 79.78% on disgust.mp4 and 43.75% (majority pass) on Fear (2).mp4. Figure 5.6 shows "
             "stable mixed training convergence peaking at Epoch 7 (RAF val macro-F1 = 0.8749, RAVDESS val macro-F1 = 0.5354).")
    add_figure(doc, "report/figures/fig_cm_rafdb_resnet18_ravdess.png", "5.3",
               "7x7 confusion matrix for default ResNet-18 (RAF+RAVDESS) on RAF-DB test set (N=3,068, 79.27% acc, 0.7077 macro-F1).", width_in=4.55)
    add_figure(doc, "report/figures/fig_cm_ravdess_val_resnet18_ravdess.png", "5.4",
               "7x7 confusion matrix for default ResNet-18 (RAF+RAVDESS) on unseen RAVDESS validation actors (N=786, 56.87% acc).", width_in=4.55)
    add_figure(doc, "report/figures/fig_cm_webcam_resnet18_ravdess.png", "5.5",
               "7x7 row-normalized confusion matrix for default ResNet-18 (RAF+RAVDESS) on 9 personal webcam clips (916 frames, 57.31% pooled).", width_in=4.55)
    add_figure(doc, "report/figures/fig_train_curves_ravdess.png", "5.6",
               "Mixed RAF-DB + RAVDESS ResNet-18 training loss and validation macro-F1 convergence curves (best Epoch 7).", width_in=5.20)
    add_csv_table(
        doc, "5.3", "Per-Clip Ground Truth, Majority Vote, and Frame Accuracy Across All 9 Webcam Clips (916/916 Detected Faces)",
        "report/tables/per_clip_results.csv",
        col_indices=[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11],
        rename_headers=["Clip", "Truth", "Frm", "Eff Maj", "Eff Acc", "RAF Maj", "RAF Acc", "RAV Maj", "RAV Acc", "RAV TopErr", "Err%"],
        col_widths=[0.74, 0.46, 0.34, 0.50, 0.46, 0.50, 0.46, 0.55, 0.46, 0.64, 0.64], font_pt=7.0,
    )

    add_para(doc,
             "Figure 5.7 and Table 5.4 measure CPU inference latency in ONE controlled 200-frame benchmark on Disgust (2).mp4 through WebcamService "
             "(earlier isolated CLI timings are non-comparable due to background variance). Because shared YuNet detection and affine crop alignment "
             "dominate fixed per-frame CPU work, all three backbones run at virtually identical real-time speed (~30 FPS): 32.98 ms p50 (34.59 ms p95) "
             "for default resnet18_ravdess vs. 33.63 ms p50 for 96-px efficientnet_current—well within the <= 150 ms interactive budget.")
    add_figure(doc, "report/figures/fig_latency_comparison.png", "5.7",
               "Unified 200-frame CPU p50/p95 latency across the three webcam backbones (earlier single-run timings are non-comparable).", width_in=5.10)
    add_csv_table(
        doc, "5.4", "Unified 200-Frame CPU Latency Benchmark on Disgust (2).mp4 (source=172 frames, measured=200 iterations, device=cpu)",
        "report/tables/latency_benchmark.csv",
        col_indices=[0, 1, 2, 7, 8, 9, 10, 11, 12, 13, 14],
        rename_headers=["Backbone", "Arch.", "Px", "Mean", "Std", "p50", "p95", "p99", "Min", "Max", "FPS(p50)"],
        col_widths=[0.94, 0.72, 0.34, 0.45, 0.42, 0.46, 0.46, 0.46, 0.46, 0.46, 0.58], font_pt=7.2,
    )

    add_h2(doc, "5.3 MELD Multimodal Ablations, AU-Rule Removal, and Calibration")
    add_para(doc,
             "Across MELD optimization, three defendable engineering results stand out: First, stripping hard-coded facial Action Unit heuristics "
             "that conflicted with learned representations lifted MELD validation macro-F1 from 0.2872 to 0.3868 (+0.0996, Figure 5.8). Second, "
             "fixing unimodal neutral collapse with StrongerAudioClassifier + WeightedRandomSampler raised Wav2Vec 2.0 acoustic macro-F1 from 0.1771 "
             "to 0.2209 (N = 2,610), while all-MiniLM-L6-v2 text classification improved over TF-IDF from 40.15% (0.2989 macro-F1) to 42.87% "
             "(0.3326 macro-F1, Table 5.5). Third, on the balanced 105-sample multimodal test set (Figures 5.9–5.12 and Table 5.5), although "
             "Text-only and Audio+Text achieve 37.14% accuracy (noisy MELD video cuts dilute point accuracy slightly to 36.19% when added), "
             "AdvancedCrossModalAttentionFusionNet with learned temperature scaling (T* = 0.5953) boosts multimodal system macro-F1 from 0.2833 "
             "to 0.3574 and cuts Expected Calibration Error (ECE) by nearly two-thirds from 0.2130 down to 0.0734 (Brier 0.8811 -> 0.7785).")
    add_figure(doc, "report/figures/fig_ablation_phase2_au_rules.png", "5.8",
               "Phase-2 ablation showing MELD validation macro-F1 improvement from 0.2872 to 0.3868 upon removing hard-coded AU rules.", width_in=4.70)
    add_figure(doc, "report/figures/fig_ablation_meld_7config.png", "5.9",
               "Seven-configuration MELD multimodal ablation benchmark showing accuracy, macro-F1, and ECE across modality subsets (N=105).", width_in=5.45)
    add_figure(doc, "report/figures/fig_ablation_fusion_optimization.png", "5.10",
               "Pre-optimization multimodal baseline vs. optimized calibrated fusion system across accuracy, macro-F1, ECE, Brier score, and NLL.", width_in=5.30)
    add_figure(doc, "report/figures/fig_calibration_reliability.png", "5.11",
               "Reliability diagram confirming calibrated probability alignment (ECE 0.2130 -> 0.0734 at T*=0.5953).", width_in=4.85)
    add_figure(doc, "report/figures/fig_cm_meld_fusion_optimized.png", "5.12",
               "7x7 confusion matrix of final calibrated AdvancedCrossModalAttentionFusionNet on balanced MELD fusion test set (N=105).", width_in=4.55)

    add_csv_table(
        doc, "5.5", "Curated MELD Unimodal, Baseline Fusion, and 7-Configuration Multimodal Ablation Results",
        "report/tables/meld_model_results.csv",
        col_indices=[0, 2, 3, 4, 5, 6, 7, 8],
        rename_headers=["Model / Config", "Architecture", "Eval Set", "N", "Acc", "MacroF1", "W-F1", "ECE"],
        filter_fn=lambda r: not r[0].startswith("Audio Exp"),
        col_widths=[1.15, 1.15, 0.85, 0.42, 0.54, 0.56, 0.58, 0.50], font_pt=7.2,
    )

    add_h2(doc, "5.4 Verification Against Chapter 1 Objectives")
    add_para(doc,
             "Table 5.6 summarizes an objective-by-objective audit comparing targets established in Chapter 1 against observed empirical metrics.")
    obj_rows = [
        ("Obj 1: Parity & Quality Gating", "100% endpoint/eval parity; unmirrored; align + blur >= 30", "916/916 frames 100% API vs eval parity; AU rules removed (0.2872->0.3868)", "Met"),
        ("Obj 2: Webcam Generalization & Latency", "Beat 15.6%/44.0% webcam, keep RAF F1 >= 0.6721, p95 <= 150 ms", "57.31% pooled (5/9 clips), RAF F1 0.7077, CPU p95 34.59 ms; anger/sadness still 0%", "Partly Met"),
        ("Obj 3: Balanced MELD Unimodal", "Mitigate neutral collapse; compare MiniLM vs TF-IDF (N=2,610)", "Audio macro-F1 0.1771->0.2209; MiniLM 0.3326 vs TF-IDF 0.2989", "Met"),
        ("Obj 4: Calibrated Attention Fusion", "Missing-modality masking + temperature scaling ECE < 0.10", "Masking verified; ECE reduced 0.2130->0.0734; A+T marginally beat A+V+T Acc", "Partly Met"),
        ("Obj 5: Full-Stack Privacy & Tests", "Volatile live RAM + 100% automated tests passing", "5/5 unit + 11/11 FastAPI integration pass; 8 UI routes verified in Chrome", "Met"),
    ]
    add_styled_table(
        doc, "5.6", "Honest Audit of Achieved Metrics Against Chapter 1 Quantitative Objectives",
        ["Objective", "Target Criterion", "Observed Empirical Outcome", "Status"],
        obj_rows, [1.20, 1.50, 2.30, 0.75], font_pt=7.8,
    )

    add_h2(doc, "5.5 Economic Analysis and Bill of Materials")
    add_para(doc,
             "As itemized in Table 5.7, the system eliminates recurring cloud GPU/API expenses by executing solely with open-source frameworks and "
             "pretrained weights on existing laptop hardware (Intel i7-12650H + 15.68 GiB RAM, INR 0 cloud cost). Retail hardware acquisition "
             "costs are left as [TO BE FILLED].")
    add_csv_table(
        doc, "5.7", "Project Bill of Materials (Software, Pretrained Weights, Datasets, and Hardware Placeholders)",
        "report/tables/bill_of_materials.csv",
        col_widths=[0.78, 1.05, 1.42, 0.34, 0.86, 1.30], font_pt=7.6,
    )

    add_h2(doc, "5.6 Positioning Against Verified Benchmarks and Failure Mode Defense")
    add_para(doc,
             "Table 5.8 contrasts our lightweight CPU-first system against verified peer-reviewed RAF-DB and MELD works. State-of-the-art RAF-DB "
             "models (POSTER++ 92.21% [9], DDAMFN 91.35% [10], EAC 90.35% [7]) employ heavy transformer/landmark branches at GPU budgets, whereas "
             "our 128-px ResNet-18 prioritizes ~30 FPS CPU execution alongside dynamic speech transfer. Similarly, published MELD models (TelME "
             "~67% [24], UniMSE 65.51% [23], MM-DFN 62.49% [21]) ingest multi-turn conversation histories on the imbalanced 2,610-utterance split, "
             "whereas our fusion test evaluates single isolated utterances under strict 105-sample class balance.")
    sota_rows = [
        ("POSTER++ (Mao et al., 2024) [9]", "RAF-DB Test (7-class)", "92.21% Acc", "Dual-branch landmark-image ViT; static images"),
        ("DDAMFN (Zhang et al., 2023) [10]", "RAF-DB Test (7-class)", "91.35% Acc", "MixConv + dual attention; static FER focus"),
        ("EAC (Zhang et al., 2022) [7]", "RAF-DB Test (7-class)", "89.99%–90.35% Acc", "ResNet-50 noisy-label regularization"),
        ("Ours: ResNet-18 (RAF+RAVDESS)", "RAF-DB / RAVDESS / 9 Clips", "79.27% RAF / 56.87% RAV / 57.31% Cam", "CPU ~30 FPS, aligned crop, domain-adapted"),
        ("TelME (Yun et al., 2024) [24]", "MELD Test (Full conv.)", "~66.36%–67.37% Acc/WF1", "Conversation history + text-teacher distillation"),
        ("UniMSE (Hu et al., 2022) [23]", "MELD Test (Full conv.)", "65.51% Acc / 65.91% W-F1", "Multi-turn T5 multimodal pretraining"),
        ("MM-DFN (Hu et al., 2022) [21]", "MELD Test (Full conv.)", "62.49% Acc / 59.46% W-F1", "Dynamic dialogue graph; uncalibrated logits"),
        ("Ours: Text (MiniLM) / Opt. Fusion", "MELD Test (2,610) / Balanced (105)", "42.87% Text (2,610) / 36.19% Fusion (105)", "Single utterance, CPU, T*=0.5953 ECE=0.0734"),
    ]
    add_styled_table(
        doc, "5.8", "Positioning Against Verified Published FER and MELD Multimodal Benchmarks",
        ["Method", "Benchmark / Split", "Reported Metrics", "Scope Difference vs. This System"],
        sota_rows, [1.45, 1.35, 1.35, 1.60], font_pt=7.7,
    )

    add_para(doc,
             "Finally, four concise failure-mode findings anchor honest panel defense:")
    add_bullet(doc, "Anger Read as Surprise (61.39%) & Sadness Read as Neutral (100%): ",
               "In single-frame Anger.mp4 crops, raised/furrowed brows mimic static surprise patterns without vocal intensity, while subtle "
               "low-arousal sadness collapses to neutral without temporal onset.")
    add_bullet(doc, "Fear vs. Disgust Ambiguity: ",
               "Disgust stabilized reliably (100% on Disgust (2).mp4, 79.78% on disgust.mp4), while Fear split across clips (43.75% majority "
               "pass on Fear (2).mp4 vs. 93.98% disgust confusion on Fear.mp4) due to shared upper-lip tension cues.")
    add_bullet(doc, "Single-Speaker Selection Scope & 8-Actor RAVDESS Gap: ",
               "Because the 9 clips represent one subject and guided default backbone promotion, 57.31% is disclosed strictly as a fixed-clip "
               "benchmark alongside the 0.2205 RAVDESS train-val macro-F1 gap across 8 training actors.")
    add_bullet(doc, "Why MELD Trimodal Calibration Justifies Fusion Over Text-Only: ",
               "Though TV scene clutter makes raw MELD video noisy (diluting Text Acc 37.14% -> 36.19%), trimodal attention cuts Expected "
               "Calibration Error from 0.1903 down to 0.0734 while gracefully masking missing streams.")

    add_mapped_pos(doc, "PO4 (Conduct Investigations of Complex Problems).")


# ---------------------------------------------------------------------------
# CHAPTER 6: CONCLUSIONS AND FUTURE WORK
# ---------------------------------------------------------------------------
def build_chapter6(doc: Document) -> None:
    add_chapter_heading(doc, "6", "Conclusions and Future Work", page_break=True)

    add_h2(doc, "6.1 Summary and Key Conclusions")
    add_para(doc,
             "This project delivered an integrated, privacy-preserving local Emotion Recognition system bridging real-time webcam facial analysis "
             "and MELD conversational multimodal classification under one FastAPI service and interactive Web Studio. Systematic root-cause "
             "debugging demonstrated that fixing preprocessing parity (916/916 HTTP-vs-offline match), enforcing unmirrored affine face alignment "
             "and Laplacian blur checks, and stripping brittle hand-crafted AU rules (+0.0996 validation macro-F1) were prerequisites before "
             "model upgrades could yield reliable gains.")
    add_para(doc,
             "Across the webcam pipeline, mixed fine-tuning of ResNet-18 on RAF-DB blended with 3,970 aligned RAVDESS speech crops raised pooled "
             "frame accuracy across 9 personal webcam clips from 15.61% (1/9 clips, efficientnet_current) to 44.00% (4/9 clips, resnet18_raf) "
             "and 57.31% (5/9 clips, resnet18_ravdess), while simultaneously lifting RAF-DB test macro-F1 to 0.7077 and holding CPU inference at "
             "~30.3 FPS (32.98 ms p50, 34.59 ms p95). Across the MELD multimodal subsystem, balanced Wav2Vec 2.0 sampling lifted acoustic macro-F1 "
             "from 0.1771 to 0.2209, all-MiniLM-L6-v2 improved text accuracy to 42.87%, and AdvancedCrossModalAttentionFusionNet with post-hoc "
             "temperature scaling (T* = 0.5953) improved fusion macro-F1 from 0.2833 to 0.3574 and lowered ECE from 0.2130 to 0.0734.")

    add_h2(doc, "6.2 Future Research and Engineering Directions")
    add_para(doc,
             "Several promising extensions remain unimplemented in the current codebase and represent concrete avenues for future work:")
    add_bullet(doc, "Held-Out Personal Webcam Adaptation and Multi-Subject Evaluation: ",
               "Collecting diverse multi-speaker webcam benchmarks under natural illumination and exploring ethical, opt-in speaker calibration "
               "(beyond the current 25-frame additive resting-neutral offset) to address residual anger/surprise and sadness/neutral failures.")
    add_bullet(doc, "HSEmotion Integration and Test-Time Augmentation: ",
               "Evaluating pretrained HSEmotion/EmotiEffNet checkpoints [11][12] directly inside the runtime WebcamService switcher alongside "
               "horizontal-flip test-time averaging if CPU budget permits.")
    add_bullet(doc, "Learned Temporal Sequence Modeling: ",
               "Replacing single-frame EMA probability filtering with lightweight temporal convolutions or attention over 1–2 second facial landmark "
               "and feature sequences to distinguish resting neutral faces from expressive speech and sadness.")
    add_bullet(doc, "Speaker-Aware Conversational MELD Memory: ",
               "Extending MELD multimodal fusion with cross-utterance dialogue context windows [21][24] and speaker-turn embeddings to close the "
               "gap with conversation-level benchmarks while retaining calibrated uncertainty outputs.")

    add_mapped_pos(doc, "PO6 (The Engineer and The World), PO7 (Environment and Sustainability), "
                        "PO8 (Ethics), PO11 (Life-Long Learning).")


# ---------------------------------------------------------------------------
# REFERENCES (IEEE FORMAT — Strict 1-to-1 Order of First Appearance [1]..[29])
# ---------------------------------------------------------------------------
def build_references(doc: Document) -> None:
    add_chapter_heading(doc, "REFERENCES", "References", page_break=True)
    # Note: Overwrite heading text cleanly to simply "REFERENCES"
    doc.paragraphs[-1].text = "REFERENCES"
    for r in doc.paragraphs[-1].runs:
        r.font.name = "Times New Roman"
        r.font.size = Pt(16)
        r.bold = True

    add_para(doc,
             "All references below correspond one-to-one with in-text IEEE citations [1]–[29] in order of first appearance. Exactly 15 of 29 "
             "cited works (51.7%) were published between 2021 and 2024.",
             italic=True, size=10.5, space_after=8.0)

    refs = [
        '[1] S. Poria, D. Hazarika, N. Majumder, G. Naik, E. Cambria, and R. Mihalcea, "MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations," in Proc. Assoc. Comput. Linguistics (ACL), 2019.',
        '[2] S. Li, W. Deng, and J. Du, "Reliable Crowdsourcing and Deep Locality-Preserving Learning for Expression Recognition in the Wild," in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2017.',
        '[3] S. R. Livingstone and F. A. Russo, "The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English," PLOS ONE, vol. 13, no. 5, e0196391, 2018.',
        '[4] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2016.',
        '[5] M. Tan and Q. V. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," in Proc. Int. Conf. Mach. Learn. (ICML), 2019.',
        '[6] Z. Wen, W. Lin, T. Wang, and G. Xu, "Distract Your Attention: Multi-Head Cross Attention Network for Facial Expression Recognition," Biomimetics, 2023.',
        '[7] Y. Zhang, C. Wang, X. Ling, and W. Deng, "Learn from All: Erasing Attention Consistency for Noisy Label Facial Expression Recognition," in Proc. Eur. Conf. Comput. Vis. (ECCV), 2022.',
        '[8] C. Zheng, M. Mendieta, and C. Chen, "POSTER: A Pyramid Cross-Fusion Transformer Network for Facial Expression Recognition," in Proc. IEEE/CVF Int. Conf. Comput. Vis. Workshops (ICCVW), 2023.',
        '[9] J. Mao, R. Xu, X. Yin, Y. Chang, B. Nie, A. Huang, and Y. Wang, "POSTER++: A simpler and stronger facial expression recognition network," Pattern Recognition, 2024.',
        '[10] S. Zhang, Y. Zhang, Y. Zhang, Y. Wang, and Z. Song, "A Dual-Direction Attention Mixed Feature Network for Facial Expression Recognition," Electronics, vol. 12, no. 17, 3595, 2023.',
        '[11] A. V. Savchenko, "HSEmotion: High-speed emotion recognition library," Software Impacts, 2022.',
        '[12] A. V. Savchenko, "EmotiEffNets for facial processing in video-based valence-arousal prediction, expression classification and action unit detection," in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. Workshops (CVPRW), 2023.',
        '[13] W. Wu, H. Peng, and S. Yu, "YuNet: A Tiny Millisecond-level Face Detector," Machine Intelligence Research, 2023.',
        '[14] T. Chen, T. Pu, H. Wu, Y. Xie, L. Liu, and L. Lin, "Cross-Domain Facial Expression Recognition: A Unified Evaluation Benchmark and Adversarial Graph Learning," IEEE Trans. Pattern Anal. Mach. Intell. (TPAMI), 2022.',
        '[15] A. Baevski, Y. Zhou, A. Mohamed, and M. Auli, "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations," in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2020.',
        '[16] L. Pepino, P. Riera, and L. Ferrer, "Emotion Recognition from Speech Using wav2vec 2.0 Embeddings," in Proc. Interspeech, 2021.',
        '[17] J. Wagner et al., "Dawn of the Transformer Era in Speech Emotion Recognition: Closing the Valence Gap," IEEE Trans. Pattern Anal. Mach. Intell. (TPAMI), 2023.',
        '[18] A. Vaswani et al., "Attention Is All You Need," in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2017.',
        '[19] W. Wang et al., "MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers," in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2020.',
        '[20] N. Reimers and I. Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," in Proc. Conf. Empirical Methods Natural Lang. Process. (EMNLP), 2019.',
        '[21] D. Hu, X. Hou, L. Wei, L. Jiang, and Y. Mo, "MM-DFN: Multimodal Dynamic Fusion Network for Emotion Recognition in Conversations," in Proc. IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP), 2022.',
        '[22] Z. Li, F. Tang, M. Zhao, and Y. Zhu, "EmoCaps: Emotion Capsule based Model for Conversational Emotion Recognition," in Findings Assoc. Comput. Linguistics (ACL), 2022.',
        '[23] G. Hu, T.-E. Lin, Y. Zhao, G. Lu, Y. Wu, and Y. Li, "UniMSE: Towards Unified Multimodal Sentiment Analysis and Emotion Recognition," in Proc. Conf. Empirical Methods Natural Lang. Process. (EMNLP), 2022.',
        '[24] T. Yun, H. Lim, J. Lee, and M. Song, "TelME: Teacher-Leading Multimodal Fusion Network for Emotion Recognition in Conversation," in Proc. Conf. North American Chapter Assoc. Comput. Linguistics (NAACL), 2024.',
        '[25] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On Calibration of Modern Neural Networks," in Proc. Int. Conf. Mach. Learn. (ICML), 2017.',
        '[26] R. Müller, S. Kornblith, and G. E. Hinton, "When Does Label Smoothing Help?," in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2019.',
        '[27] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal Loss for Dense Object Detection," in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2017.',
        '[28] I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization," in Proc. Int. Conf. Learn. Represent. (ICLR), 2019.',
        '[29] A. Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library," in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2019.',
    ]
    for ref in refs:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.left_indent = Inches(0.32)
        p.paragraph_format.first_line_indent = Inches(-0.32)
        p.paragraph_format.line_spacing = 1.25
        p.paragraph_format.space_after = Pt(4)
        r = p.add_run(ref)
        r.font.name = "Times New Roman"
        r.font.size = Pt(11.0)


# ---------------------------------------------------------------------------
# APPENDICES & ANNEXURES
# ---------------------------------------------------------------------------
def build_appendices(doc: Document) -> None:
    add_chapter_heading(doc, "APPENDIX A", "Dataset Audit and Preprocessing Parity Summary", page_break=True)
    doc.paragraphs[-1].text = "APPENDIX A: DATASET AUDIT AND REPOSITORY VERIFICATION"
    for r in doc.paragraphs[-1].runs:
        r.font.name = "Times New Roman"
        r.font.size = Pt(15)
        r.bold = True

    add_para(doc,
             "All dataset statistics, duplicate checks, and parity audits verified across report generation are consolidated below for reproducibility:")
    add_bullet(doc, "RAF-DB Deduplication Audit: ",
               "Exact MD5/RGB inspection of official RAF-DB labels identified 2 exact within-train duplicates removed prior to stratified splitting "
               "(10,428 train, 1,841 val, 3,068 test; zero train-test leakage).")
    add_bullet(doc, "RAVDESS Filtering Log (results/prepare.log): ",
               "From 1,200 source MP4 files, prepare_ravdess.py skipped 600 non-AV/song clips, dropped 80 calm clips, and skipped 2 clips with no valid "
               "aligned crops, saving 3,970 high-sharpness crops across 518 speech clips (Actors 01–08 train = 3,184; Actors 09–10 val = 786).")
    add_bullet(doc, "HTTP-vs-Disk Frame Parity (results/webcam_fix_log.md): ",
               "Sending all 916 frames from the 9 evaluation clips resized to 320x180 JPEG85 to FastAPI yielded 916/916 identical emotion predictions "
               "and matching probability vectors relative to direct disk evaluation in eval_clips.py.")
    add_bullet(doc, "Report Asset Manifest Summary: ",
               "A curated, viva-focused selection of 28 high-signal PNG figures (200 dpi), 11 curated CSV tables, and all 22 mathtext formula "
               "PNGs from the complete 78-asset archive in report/assets_manifest.md is embedded across Chapters 1, 3, 4, and 5.")

    add_chapter_heading(doc, "APPENDIX B", "Selected Core Implementation Snippets", page_break=True)
    doc.paragraphs[-1].text = "APPENDIX B: SELECTED CORE IMPLEMENTATION SNIPPETS"
    for r in doc.paragraphs[-1].runs:
        r.font.name = "Times New Roman"
        r.font.size = Pt(15)
        r.bold = True

    add_para(doc, "Five concise verbatim excerpts (each under 40 lines) illustrate the key algorithmic mechanisms verified in code:")

    add_code_snippet(doc, "Snippet B.1: Eye-Line Affine Alignment & Square Crop (live_resnet18_demo.py:49-81)", [
        "def aligned_square_crop(frame, face, pad, min_side=64):",
        "    x, y, w, h = face[:4]",
        "    rex, rey, lex, ley = face[4:8]",
        "    angle = math.degrees(math.atan2(ley - rey, lex - rex))",
        "    if abs(angle) > 45:  # implausible tilt: skip alignment instead of guessing",
        "        angle = 0.0",
        "    cx, cy = (rex + lex) / 2.0, (rey + ley) / 2.0",
        "    m = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)",
        "    height, width = frame.shape[:2]",
        "    rotated = cv2.warpAffine(",
        "        frame, m, (width, height), flags=cv2.INTER_LINEAR,",
        "        borderMode=cv2.BORDER_REPLICATE,",
        "    )",
        "    centre = m @ np.array([x + w / 2.0, y + h / 2.0, 1.0])",
        "    side = int(max(w, h) * pad)",
        "    if side < min_side:",
        "        return None",
        "    x0 = int(round(centre[0] - side / 2.0))",
        "    y0 = int(round(centre[1] - side / 2.0))",
        "    pad_l, pad_t = max(0, -x0), max(0, -y0)",
        "    pad_r, pad_b = max(0, x0 + side - width), max(0, y0 + side - height)",
        "    if pad_l or pad_t or pad_r or pad_b:",
        "        rotated = cv2.copyMakeBorder(",
        "            rotated, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE",
        "        )",
        "        x0 += pad_l",
        "        y0 += pad_t",
        "    return rotated[y0:y0 + side, x0:x0 + side]",
    ])

    add_code_snippet(doc, "Snippet B.2: RAVDESS Temporal Windowing & Laplacian Quality Gate (prepare_ravdess.py:101-133)", [
        "def compute_sample_indices(frame_count: int, fps: float, target_fps: float) -> list[int]:",
        "    if frame_count <= 0:",
        "        return []",
        "    safe_fps = fps if fps and not math.isnan(fps) and fps > 1.0 else 30.0",
        "    start_idx = max(0, int(math.floor(0.30 * frame_count)))",
        "    end_idx = min(frame_count, int(math.ceil(0.70 * frame_count)))",
        "    if end_idx <= start_idx:",
        "        end_idx = min(frame_count, start_idx + 1)",
        "    stride = max(1, int(round(safe_fps / max(0.5, target_fps))))",
        "    return list(range(start_idx, end_idx, stride))",
        "",
        "def _passes_quality_gate(",
        "    frame, face, pad: float, min_side: int, blur_threshold: float, min_conf: float",
        "):",
        "    fw, fh = float(face[2]), float(face[3])",
        "    conf = float(face[-1])",
        "    if conf < min_conf or fw < min_side or fh < min_side:",
        "        return None",
        "    crop = aligned_square_crop(frame, face, pad=pad, min_side=min_side)",
        "    if crop is None or crop.size == 0:",
        "        return None",
        "    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop",
        "    if float(cv2.Laplacian(gray, cv2.CV_64F).var()) < blur_threshold:",
        "        return None",
        "    return crop",
    ])

    add_code_snippet(doc, "Snippet B.3: Mixed Balanced Mini-Batch Constructor (train_mixed.py:606-634)", [
        "def _build_mixed_loaders(",
        "    raf_train: pd.DataFrame, extra_train: pd.DataFrame, train_tf: Callable,",
        "    batch_size: int, extra_share: float, epoch_seed: int, pin_memory: bool,",
        ") -> tuple[DataLoader, Optional[DataLoader], int, int, int]:",
        "    if not 0.0 <= extra_share < 1.0:",
        "        raise ValueError(f\"--extra-share must be in [0, 1), got {extra_share}\")",
        "    n_extra = int(round(batch_size * extra_share))",
        "    if extra_share > 0.0 and batch_size >= 2:",
        "        n_extra = max(1, min(batch_size - 1, n_extra))",
        "    n_raf = batch_size - n_extra",
        "    steps = max(1, math.ceil(len(raf_train) / max(1, n_raf)))",
        "    raf_loader = make_balanced_train_loader(",
        "        raf_train, train_tf, n_raf, steps * n_raf, epoch_seed, pin_memory",
        "    )",
        "    extra_loader = (",
        "        make_balanced_train_loader(",
        "            extra_train, train_tf, n_extra, steps * n_extra, epoch_seed + 10000, pin_memory",
        "        )",
        "        if n_extra > 0 else None",
        "    )",
        "    return raf_loader, extra_loader, steps, n_raf, n_extra",
    ])

    add_code_snippet(doc, "Snippet B.4: Attention Masking, Entropy Gating & Consensus Logit Blend (fusion_model.py:179-217)", [
        "        key_padding_mask = ~mask",
        "        all_masked = key_padding_mask.all(dim=1)",
        "        if all_masked.any():",
        "            key_padding_mask[all_masked, 0] = False",
        "        attended = self.transformer_encoder(tokens, src_key_padding_mask=key_padding_mask)",
        "        attended = self.norm(attended)",
        "        cert_qual_a = audio_feat[:, 14:16].unsqueeze(1)",
        "        cert_qual_t = text_feat[:, 14:16].unsqueeze(1)",
        "        cert_qual_v = video_feat[:, 14:16].unsqueeze(1)",
        "        cq = torch.cat([cert_qual_a, cert_qual_t, cert_qual_v], dim=1)",
        "        raw_gates = self.gate_mlp(torch.cat([attended, cq], dim=-1))",
        "        effective_gates = raw_gates * mask.unsqueeze(-1).float()",
        "        normalized_gates = effective_gates / effective_gates.sum(dim=1, keepdim=True).clamp(min=1e-7)",
        "        neural_logits = self.classifier((attended * normalized_gates).sum(dim=1))",
        "        cls_all = torch.cat([audio_feat[:, 7:14].unsqueeze(1),",
        "                             text_feat[:, 7:14].unsqueeze(1),",
        "                             video_feat[:, 7:14].unsqueeze(1)], dim=1)",
        "        consensus_logits = (cls_all * normalized_gates).sum(dim=1)",
        "        gamma = torch.clamp(self.consensus_weight, 0.1, 0.9)",
        "        final_logits = (1.0 - gamma) * neural_logits + gamma * consensus_logits",
        "        return final_logits, normalized_gates.squeeze(-1)",
    ])

    add_code_snippet(doc, "Snippet B.5: Confidence Tier & Uncertain Flagging Rule (webcam_service.py:832-853)", [
        "        sorted_probs = np.sort(calibrated_probs)[::-1]",
        "        p_top1 = float(sorted_probs[0])",
        "        p_top2 = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0",
        "        conf_margin = float(p_top1 - p_top2)",
        "        if calibrated_conf >= 0.65 and conf_margin >= 0.15:",
        "            conf_tier = \"Robust\"",
        "        elif calibrated_conf >= 0.45 and conf_margin >= 0.08:",
        "            conf_tier = \"Moderate\"",
        "        else:",
        "            conf_tier = \"Subtle / Ambiguous\"",
        "        is_uncertain = bool(p_top1 < 0.38 or conf_margin < 0.08)",
        "        uncertainty_reason = None",
        "        if is_uncertain:",
        "            uncertainty_reason = (",
        "                f\"Low top-1 confidence ({p_top1:.1%})\" if p_top1 < 0.38",
        "                else f\"Ambiguous margin between top classes ({conf_margin:.1%})\"",
        "            )",
    ])


def build_annexure(doc: Document) -> None:
    add_chapter_heading(doc, "ANNEXURE", "Accreditation Mappings, Outcomes, Sustainability, and Team Roles", page_break=True)
    doc.paragraphs[-1].text = "ANNEXURES"
    for r in doc.paragraphs[-1].runs:
        r.font.name = "Times New Roman"
        r.font.size = Pt(16)
        r.bold = True

    add_h2(doc, "Annexure A: Program Outcomes (POs) and Program Specific Outcomes (PSOs) Mapping")
    add_para(doc,
             "Note: Specific institute PSO wording is marked [TO BE FILLED: Department PSO 1–3 Definitions]. Chapter 1 teamwork explicitly "
             "maps to PO9 (Individual and Teamwork), and plagiarism/integrity to PO8 (Ethics).")
    po_rows = [
        ("PO1", "Engineering Knowledge", "Ch 1", "High (3)", "Signal processing, CNN/Transformer architectures, probability theory"),
        ("PO2", "Problem Analysis", "Ch 1, Ch 2", "High (3)", "Diagnosing domain shift, train-serve mismatch, class imbalance, ECE"),
        ("PO3", "Design/Development of Solutions", "Ch 3, Ch 4", "High (3)", "Unified WebcamService, Prepare/Mixed Trainer, Cross-Modal Attention"),
        ("PO4", "Conduct Investigations", "Ch 4, Ch 5", "High (3)", "Confusion matrices, ablations, unified 200-frame latency, test suite"),
        ("PO5", "Modern Tool Usage", "Ch 3, Ch 4", "High (3)", "PyTorch, TorchVision, OpenCV YuNet, Transformers, FastAPI, WebSockets"),
        ("PO6", "Engineer and The World", "Ch 1, Ch 4, Ch 6", "High (3)", "Local CPU execution, non-clinical apparent-expression disclaimers"),
        ("PO7", "Environment & Sustainability", "Ch 6, Ann C", "Medium (2)", "Frozen feature reuse, partial layer freeze, CPU-only inference"),
        ("PO8", "Ethics", "Decl., Ch 6", "High (3)", "Zero plagiarism, transparent failure disclosure, biometric RAM-only path"),
        ("PO9", "Individual and Teamwork", "Ch 1, Ann D", "High (3)", "Partitioned multi-developer responsibility across Vision, Fusion, and UI"),
        ("PO10", "Project Management & Finance", "Ch 1, Ch 2", "Medium (2)", "Git-grounded WBS/Gantt, risk register, zero-cloud-cost BOM"),
        ("PO11", "Life-Long Learning", "Ch 2, Ch 6", "High (3)", "Literature comparison (2021–2024), future temporal/HSEmotion paths"),
        ("PSO1–3", "[TO BE FILLED: Dept PSO Text]", "Ch 3–Ch 5", "[TO BE FILLED]", "[TO BE FILLED: Align with institutional PSO criteria]"),
    ]
    add_styled_table(
        doc, "A.1", "Mapping of Project Chapters and Deliverables to NBA Program Outcomes",
        ["Outcome", "Outcome Description", "Mapped Chapters", "Level", "Evidence in Project"],
        po_rows, [0.58, 1.35, 0.85, 0.72, 2.25], font_pt=7.8,
    )

    add_h2(doc, "Annexure B: Outcome Summary and Real-World Application Potential")
    add_para(doc,
             "Project Evaluation Status / Publications / Patents: [TO BE FILLED: Jury Grade / Publication / Patent Filing Details if any].")
    add_para(doc,
             "Practical Deployment Potential: The delivered local FastAPI engine and zero-build Web Studio apply directly to opt-in accessible "
             "communication aids, offline media/usability testing dashboards, and research prototypes requiring calibrated multimodal uncertainty "
             "without transmitting private webcam video or microphone audio to remote cloud servers.")

    add_h2(doc, "Annexure C: Project Sustainability Statement")
    sust_rows = [
        ("Environmental", "Energy efficiency", "Reuses frozen Wav2Vec2/MiniLM/ResNet-50 embeddings; freezes stages through layer3; runs CPU-only without GPU clusters."),
        ("Social & Ethical", "Biometric privacy & consent", "Zero disk writes in live stream routes; clear subtle/ambiguous uncertainty flags; no diagnostic clinical overclaims."),
        ("Economic", "Zero proprietary lock-in", "100% open-source Python stack with ~30 FPS real-time execution on commodity laptop CPUs."),
    ]
    add_styled_table(
        doc, "A.2", "Three-Pillar Project Sustainability Assessment",
        ["Pillar", "Focus Criterion", "Verified System Implementation"],
        sust_rows, [0.95, 1.45, 3.35], font_pt=8.2,
    )

    add_h2(doc, "Annexure D: Student Roles and Responsibilities")
    role_rows = [
        ("[TO BE FILLED: Student Name 1]", "[TO BE FILLED: Enrollment No. 1]", "Multimodal & Backend Lead", "AdvancedCrossModalAttentionFusionNet, temperature scaling, FastAPI endpoints, unit/integration runner"),
        ("[TO BE FILLED: Student Name 2]", "[TO BE FILLED: Enrollment No. 2]", "Webcam Vision & Domain Lead", "YuNet integration, aligned_square_crop, prepare_ravdess.py, train_mixed.py, parity & latency benchmarks"),
        ("[TO BE FILLED: Student Name 3]", "[TO BE FILLED: Enrollment No. 3]", "Speech, NLP & Frontend Lead", "Wav2Vec2 balanced sampling experiments, MiniLM text model, Web Studio SPA, report assets & build script"),
    ]
    add_styled_table(
        doc, "A.3", "Student Roles and Responsibilities Breakdown (Aligned with PO9)",
        ["Student Name", "Enrollment No.", "Assigned Role", "Verified Deliverables Owned"],
        role_rows, [1.20, 1.15, 1.25, 2.15], font_pt=8.0,
    )


def main() -> None:
    doc = Document()
    setup_styles(doc)
    build_prelims(doc)
    build_chapter1(doc)
    build_chapter2(doc)
    build_chapter3(doc)
    build_chapter4(doc)
    build_chapter5(doc)
    build_chapter6(doc)
    build_references(doc)
    build_appendices(doc)
    build_annexure(doc)

    out_docx = REPORT_DIR / "Project_Report.docx"
    doc.save(str(out_docx))
    print(f"Wrote {out_docx} ({out_docx.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
