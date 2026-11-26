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

    n_rows = len(table.rows)
    v_pad = 36 if font_pt <= 8.0 else 55
    ls = 1.05 if font_pt <= 8.0 else 1.1
    sp_pt = 0.5 if font_pt <= 8.0 else 1.0
    for r_idx, row in enumerate(table.rows):
        trPr = row._tr.get_or_add_trPr()
        if r_idx == 0:
            trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
        trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
        keep_row = (r_idx < n_rows - 1) if n_rows <= 10 else (r_idx == 0 or r_idx == n_rows - 2)
        for c_idx, cell in enumerate(row.cells):
            cell.width = Inches(col_widths[c_idx])
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            tcPr = cell._tc.get_or_add_tcPr()
            tcMar = parse_xml(
                f'<w:tcMar {nsdecls("w")}>'
                f'<w:top w:w="{v_pad}" w:type="dxa"/><w:bottom w:w="{v_pad}" w:type="dxa"/>'
                '<w:left w:w="75" w:type="dxa"/><w:right w:w="75" w:type="dxa"/>'
                "</w:tcMar>"
            )
            for old_m in tcPr.findall(qn("w:tcMar")):
                tcPr.remove(old_m)
            tcPr.append(tcMar)
            if r_idx == 0:
                tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{header_fill}"/>'))
            for p_idx, p in enumerate(cell.paragraphs):
                p.paragraph_format.line_spacing = ls
                p.paragraph_format.space_before = Pt(sp_pt)
                p.paragraph_format.space_after = Pt(sp_pt)
                if keep_row or p_idx < len(cell.paragraphs) - 1:
                    p.paragraph_format.keep_with_next = True
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
    tbl.rows[0]._tr.get_or_add_trPr().append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
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
             "Automated human emotion recognition spans two complementary problem regimes: spontaneous real-time facial expression analysis from "
             "live consumer cameras and multi-channel conversational affect modeling across speech prosody, spoken text, and visual context. In "
             "practice, standalone systems degrade under train-serve preprocessing mismatches, domain shift between curated corpora and webcams, "
             "severe conversational class imbalance, conflicting unimodal predictions, and overconfident uncalibrated probabilities.")
    add_para(doc,
             "This project presents a unified, strictly local CPU emotion recognition platform built across a shared seven-emotion schema (anger, "
             "disgust, fear, joy, neutral, sadness, and surprise), hosted via FastAPI and a responsive Web Studio UI: (1) a MELD conversational "
             "multimodal subsystem (Sections 3.3.4–3.3.8, 3.4) integrating Wav2Vec 2.0 Base acoustic embeddings with balanced sampling, "
             "contextual all-MiniLM-L6-v2 text representations, ResNet-50 temporal pooling, and an AdvancedCrossModalAttentionFusionNet with a "
             "16-D per-modality contract, missing-modality Transformer masking, Shannon-entropy reliability gating, consensus logit blending, "
             "and post-hoc temperature scaling; and (2) a real-time webcam engine (Sections 3.3.1–3.3.3) unifying YuNet eye-line alignment, "
             "Laplacian blur quality gating, and selectable CNN backbones including mixed RAF-DB + aligned RAVDESS fine-tuning.")
    add_para(doc,
             "Systematic empirical evaluation (Chapter 5) demonstrates balanced improvements alongside documented limitations. On MELD, balanced "
             "acoustic sampling resolves minority-class collapse, contextual embeddings improve linguistic discrimination over TF-IDF baselines, "
             "and cross-modal attention fusion improves accuracy over baseline fusion and outperforms every unimodal stream on fully conflicting "
             "samples while cutting calibration error below target bounds. Across the webcam subsystem, unifying live HTTP/WebSocket transport "
             "and offline evaluation, removing heuristic Action Unit overrides, and mixed RAVDESS adaptation substantially raise clip and frame "
             "generalization at real-time laptop CPU frame rates. Automated unit and integration verification confirms reliable end-to-end "
             "execution while keeping live biometric streams strictly in volatile RAM without clinical diagnostic claims.")
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
        ("BS", "Brier Score"),
        ("CD-FER", "Cross-Domain Facial Expression Recognition"),
        ("CE", "Cross-Entropy Loss"),
        ("CLAHE", "Contrast Limited Adaptive Histogram Equalization"),
        ("CNN", "Convolutional Neural Network"),
        ("CPU", "Central Processing Unit"),
        ("ECE", "Expected Calibration Error"),
        ("EMA", "Exponential Moving Average"),
        ("ERC", "Emotion Recognition in Conversations"),
        ("FER", "Facial Expression Recognition"),
        ("FFN", "Feed-Forward Network"),
        ("FPS", "Frames Per Second"),
        ("GELU / ReLU / SiLU", "Gaussian / Rectified / Sigmoid Linear Unit Activations"),
        ("HTTP / WS", "Hypertext Transfer Protocol / WebSocket"),
        ("JSON", "JavaScript Object Notation"),
        ("LN", "Layer Normalization"),
        ("MCE", "Maximum Calibration Error"),
        ("MELD", "Multimodal EmotionLines Dataset"),
        ("MHA", "Multi-Head Attention"),
        ("MLP", "Multi-Layer Perceptron"),
        ("NBA", "National Board of Accreditation"),
        ("NLL", "Negative Log-Likelihood"),
        ("NLP", "Natural Language Processing"),
        ("ONNX", "Open Neural Network Exchange"),
        ("PO / PSO", "Program Outcome / Program Specific Outcome"),
        ("RAF-DB", "Real-world Affective Faces Database"),
        ("RAM", "Random Access Memory"),
        ("RAVDESS", "Ryerson Audio-Visual Database of Emotional Speech and Song"),
        ("REST", "Representational State Transfer"),
        ("RMS", "Root Mean Square"),
        ("SER", "Speech Emotion Recognition"),
        ("SNR", "Signal-to-Noise Ratio"),
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
             "Empirical auditing of the initial codebase uncovered four systemic bottlenecks affecting both the conversational multimodal and live "
             "webcam pipelines. First, severe class imbalance in MELD—where 47.1% of training utterances are neutral while disgust and fear each "
             "represent under 2.8%—caused baseline acoustic classifiers to collapse toward majority-class predictions (quantified in Section 5.2). "
             "Second, unimodal television streams frequently disagree or carry noisy facial/scene cuts, while uncalibrated baseline fusion output "
             "overconfident probability estimates on conflicting samples (Section 5.2) without mechanisms to handle missing channels.")
    add_para(doc,
             "Third, train-serve pipeline mismatch occurred when REST/WebSocket webcam routes, offline evaluation scripts, and training loaders "
             "used inconsistent face cropping, horizontal mirroring, or hand-crafted Action Unit (AU) heuristic overrides, degrading validation F1 "
             "(Sections 5.1–5.2). Fourth, severe domain shift caused a MELD-trained EfficientNet-B0 baseline to generalize poorly onto consumer "
             "webcam video (Section 5.1). Resolving these bottlenecks motivated a cohesive, locally deployable architecture pairing balanced "
             "unimodal encoders, missing-modality attention fusion, and temperature calibration with unified webcam preprocessing and mixed-domain "
             "fine-tuning.")

    add_h2(doc, "1.3 Problem Statement")
    add_para(doc,
             "The objective of this project is to design, implement, and empirically validate a unified local-inference emotion recognition "
             "system covering both (a) a MELD multimodal conversational analysis and calibration subsystem fusing acoustic (Section 3.3.5), "
             "linguistic (Section 3.3.6), and visual (Section 3.3.4) streams through cross-modal attention fusion (Sections 3.3.7–3.3.8), and "
             "(b) a real-time webcam facial expression subsystem with unified alignment, quality gating, and selectable vision backbones "
             "(Sections 3.3.1–3.3.3) across a shared seven-emotion taxonomy, served through a FastAPI backend and browser studio interface.")

    add_h2(doc, "1.4 Project Objectives")
    add_para(doc, "To resolve the identified engineering gaps across both subsystems evenly, five core objectives were established:")
    add_bullet(doc, "Objective 1 — Balanced MELD Unimodal Modeling: ",
               "Mitigate majority-class collapse in conversational speech emotion recognition via balanced acoustic sampling (Section 3.3.5) and "
               "benchmark dense contextual sentence embeddings against sparse lexical baselines across held-out MELD utterances (Section 5.2).")
    add_bullet(doc, "Objective 2 — Calibrated Missing-Modality Attention Fusion: ",
               "Construct an attention-gated neural fusion architecture supporting arbitrary unimodal, bimodal, and trimodal subsets, conflict "
               "resolution, and post-hoc temperature scaling targeting Expected Calibration Error below 0.10 (Sections 3.4, 5.2–5.3).")
    add_bullet(doc, "Objective 3 — Webcam Train-Serve Parity & Quality Gating: ",
               "Unify all live webcam endpoints (/frame, /frame-json, /ws, /ws/webcam, /multimodal) and offline evaluation onto a single "
               "WebcamService.process_frame engine enforcing unmirrored eye-line alignment, blur gating, and removal of brittle AU heuristics (Section 5.1).")
    add_bullet(doc, "Objective 4 — Cross-Domain Webcam Generalization under CPU Budget: ",
               "Adapt static in-the-wild facial expression backbones to speaking consumer webcam video via mixed RAF-DB + RAVDESS fine-tuning "
               "(Section 3.3.2) without catastrophic forgetting while holding CPU p95 latency <= 150 ms (Section 5.1).")
    add_bullet(doc, "Objective 5 — Privacy-Preserving Full-Stack Verification: ",
               "Deploy the unified engine through FastAPI REST/WebSocket endpoints and a zero-build Web Studio SPA enforcing volatile RAM "
               "processing for live streams and full automated pass verification across unit and integration suites (Sections 4.4–4.6).")

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
             "compute energy. This project minimizes energy consumption by reusing frozen pretrained encoders (Sections 3.3.4–3.3.6), partially "
             "freezing early convolutional stages during mixed adaptation (Section 3.3.2), running compact vision backbones directly on laptop "
             "CPU hardware without dedicated GPU clusters, and gating blurred or redundant frame inference.")

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
        col_widths=[0.36, 0.90, 0.88, 0.63, 0.53, 1.20, 1.25],
        font_pt=7.8,
        rename_headers=["ID", "Risk", "Category", "Likelihood", "Impact", "Observed Evidence", "Mitigation Implemented"],
    )

    add_h2(doc, "1.7 Scope and Limitations")
    add_para(doc,
             "The system covers seven discrete emotion categories (anger, disgust, fear, joy, neutral, sadness, and surprise) across MELD "
             "conversational audio, text, and video utterances as well as live CPU webcam streams. Honest operational boundaries are "
             "acknowledged upfront (detailed empirically in Sections 5.1–5.4): MELD multimodal ablations reflect a balanced test partition "
             "evaluated turn-by-turn without cross-utterance dialogue context memory, visual TV streams contain frequent scene cuts that depress "
             "unimodal video accuracy, and single-frame webcam evaluation clips represent one acted speaker serving for both selection and "
             "evaluation with unresolved anger and sadness confusion.")
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
             "text-teacher cross-modal knowledge distillation to boost weak acoustic and visual streams to ~66.36%–67.37%. However, these "
             "conversational models depend on multi-turn dialogue graphs or heavy sequence backbones, assume complete trimodal inputs at test "
             "time, and omit explicit reliability gating or uncertainty reporting when noisy TV camera cuts conflict with spoken cues.")

    add_h2(doc, "2.3 Uncertainty Calibration and Imbalanced Learning")
    add_para(doc,
             "Modern deep classifiers frequently exhibit severe overconfidence under distribution mismatch or class imbalance. Guo et al. [25] "
             "demonstrated that post-hoc single-parameter temperature scaling sharply reduces Expected Calibration Error (ECE) without altering "
             "decision boundaries, while Müller et al. [26] showed that label smoothing improves generalization and implicit calibration. To "
             "address extreme class skew (such as 47% neutral utterances in MELD), Lin et al. [27] proposed Focal Loss to down-weight well-"
             "classified majority samples, complementing inverse-frequency class-balanced sampling and decoupled weight decay optimization (AdamW) [28]. "
             "For multimodal fusion, entropy-based uncertainty estimation and calibrated log-odds consensus blending complement temperature "
             "scaling by down-weighting uninformative or corrupted streams prior to final probability normalization, though fitted trimodal "
             "temperatures require careful verification across masked single-modality subsets.")

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
    add_bullet(doc, "Gap 3 — Calibration, Conflict Resolution, and Modality-Absence Resilience in Multimodal Fusion: ",
               "Most conversational ERC benchmarks [21]–[24] assume all three modalities are present simultaneously and optimize accuracy or F1 "
               "without evaluating Expected Calibration Error (ECE), Brier score [25], or degraded subset performance; lightweight single-utterance "
               "fusion requires entropy/quality reliability gating, key-padding masking for missing channels, and calibrated conflict arbitration.")
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
             "shaped the design; concrete model realizations are detailed in Section 3.3.")
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

    add_h2(doc, "3.2 Overall System Architecture and Pipeline Overview")
    add_para(doc,
             "Figure 3.1 depicts the top-level platform architecture hosted inside backend/main.py. HTTP and WebSocket traffic enters the FastAPI "
             "server on local loopback (127.0.0.1:8000), which mounts five service modules sharing a unified seven-class taxonomy (anger, disgust, "
             "fear, joy, neutral, sadness, surprise): AudioEmotionService (Section 3.3.5), TextEmotionService (Section 3.3.6), "
             "VideoEmotionService (Section 3.3.4), MultimodalFusionService (Sections 3.3.7–3.3.8 and 3.4), and WebcamService (Sections 3.3.1–3.3.3).")
    add_figure(doc, "report/figures/fig_diag_system_architecture.png", "3.1",
               "Overall system architecture integrating MELD multimodal pipelines, WebcamService, FastAPI, and Web Studio UI.", width_in=5.65)

    add_para(doc,
             "Real-time webcam analysis follows the single-source inference pipeline in Figure 3.2 (formalized in Algorithm 3.1). Incoming BGR "
             "frames pass through YuNet face and 5-landmark detection (Section 3.3.1), eye-line affine alignment, Laplacian blur quality gating, "
             "and the active CNN expression backbone (Sections 3.3.2–3.3.3), followed by optional per-client neutral baseline subtraction, "
             "confidence-adaptive EMA temporal smoothing, 3-frame hysteresis filtering, and uncertainty flagging prior to overlay.")
    add_figure(doc, "report/figures/fig_diag_live_inference.png", "3.2",
               "Live webcam inference pipeline from capture through YuNet alignment, quality gate, backbone, EMA, and uncertainty overlay.", width_in=5.65)

    add_para(doc,
             "To overcome static-to-speaking domain shift without losing RAF-DB discrimination, RAVDESS speech clips are curated via "
             "prepare_ravdess.py (Figure 3.3): from 1,200 source MP4 files, filtering for audio-visual speech (modality=01, channel=01) and "
             "dropping calm (code 02) leaves 520 eligible clips, from which middle 30%–70% frames sampled at 5.0 fps pass YuNet eye-line "
             "alignment and Laplacian blur gating (threshold 30.0) to yield 3,970 JPG crops across 518 valid clips (3,184 train from Actors 01–08; "
             "786 validation from Actors 09–10; 2 clips yielded no valid crops). Mixed fine-tuning in train_mixed.py (see Section 3.3.2 for "
             "layer freezing, batch composition, and optimizer hyperparameters) then trains jointly on balanced RAF-DB and RAVDESS speech crops.")
    add_figure(doc, "report/figures/fig_diag_ravdess_prep.png", "3.3",
               "RAVDESS speech video preparation flow yielding 3,970 eye-aligned crops across 518 valid clips.", width_in=5.60)

    add_para(doc,
             "Finally, Figure 3.4 documents the REST and WebSocket deployment boundary exposed by FastAPI across backend/main.py and "
             "backend/api/*.py, organizing all 17 routes detailed in Table 4.3 into MELD unimodal endpoints, the multimodal fusion router, "
             "webcam streaming routes, and session/calibration controls.")
    add_figure(doc, "report/figures/fig_diag_api_deployment.png", "3.4",
               "FastAPI deployment and REST/WebSocket endpoint architecture across 17 routes on 127.0.0.1:8000.", width_in=5.65)

    # -----------------------------------------------------------------------
    # 3.3 MODEL DESCRIPTIONS (SINGLE CANONICAL SPECIFICATION PER MODEL)
    # -----------------------------------------------------------------------
    add_h2(doc, "3.3 Model Descriptions")
    add_para(doc,
             "Each neural and statistical model employed across the webcam and MELD multimodal subsystems is specified once below using a uniform "
             "seven-part technical template; subsequent chapters cross-reference these subsections without repeating model internals.")

    def add_model_spec(sub_title: str, fields: Sequence[tuple[str, str]]) -> None:
        add_h3(doc, sub_title)
        for label, content in fields:
            add_bullet(doc, f"{label}: ", content)

    add_model_spec("3.3.1 YuNet Face and Landmark Detector", [
        ("Role in system", "Detects the primary facial bounding box and five anatomical landmarks for real-time webcam tracking and offline RAVDESS crop extraction."),
        ("Architecture and parameter count", "Anchor-free lightweight CNN in OpenCV DNN ONNX format (models/face_detection_yunet_2023mar.onnx, 232,589 bytes); the ONNX graph contains 106 nodes, 112 initializers, and 53,121 stored initializer parameters (the upstream paper reports 75.8K parameters [13])."),
        ("Input and preprocessing", "Raw unmirrored BGR image frame with dynamic input_size set via setInputSize((w, h)); runtime thresholds score_threshold=0.55, nms_threshold=0.3, top_k=5000 (WebcamService._init_face_detector)."),
        ("Output", "Per-face 15-element vector: bounding box (x, y, w, h), five 2D landmarks (right eye, left eye, nose tip, right mouth, left mouth), and detection confidence score."),
        ("Training and hyperparameters", "Pretrained OpenCV Zoo ONNX artifact (2023mar release); no project-side retraining performed."),
        ("Why chosen", "Provides sub-few-millisecond CPU detection alongside the five facial landmarks required for eye-line affine rotation alignment, unlike box-only Haar cascades."),
        ("Where deployed", "backend/services/webcam_service.py (live and offline clip evaluation), prepare_ravdess.py (RAVDESS crop pipeline), and train_webcam_model.py (legacy MELD crop extraction)."),
    ])

    add_model_spec("3.3.2 ResNet-18 Facial Expression Backbones (RAF-DB and Mixed RAF-DB + RAVDESS)", [
        ("Role in system", "Primary real-time webcam facial expression classifier, deployed as two checkpoints: resnet18_raf (static RAF-DB baseline) and resnet18_ravdess (default mixed-domain checkpoint)."),
        ("Architecture and parameter count", "TorchVision ResNet-18 [4] (conv1 7x7, bn1, layer1–layer4 at widths 64/128/256/512 with residual skip connections) followed by a Linear(512, 7) head; 11,180,103 total parameters. Under --freeze-until layer3, conv1–layer3 (and their BatchNorm running stats) are frozen, leaving 8,397,319 trainable parameters across layer4 and fc."),
        ("Input and preprocessing", "Eye-aligned square BGR crop (pad=1.10) converted to RGB PIL, evaluated via Resize(146, bilinear, antialias=True) -> CenterCrop(128) -> ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]). Training augments with RandomResizedCrop(128), flip, 12-deg rotation, brightness/contrast/gamma/color-temp jitter, downscale-upscale, Gaussian blur, JPEG compression (Q=35–85), Gaussian noise, and synthetic eyeglasses occlusion (train_mixed.py:273–370)."),
        ("Output", "7-D raw emotion logits in R^7; resnet18_raf outputs RAF-DB folder order and is remapped by class name to canonical order, while resnet18_ravdess natively outputs canonical order (anger, disgust, fear, joy, neutral, sadness, surprise)."),
        ("Training and hyperparameters", "resnet18_raf (train_raf_resnet18.py): ImageNet-pretrained init, AdamW (wd=1e-4), backbone_lr=1e-4, head_lr=1e-3, CrossEntropyLoss (smoothing=0.0), bs=32, seed=42, 10 epochs (early stop 9, best epoch 6). resnet18_ravdess (train_mixed.py): initialized from best_resnet18.pt, 26 RAF-DB + 6 RAVDESS balanced crops per batch (extra_share=0.2), AdamW (wd=1e-4), backbone_lr=2e-5 (layer4), head_lr=2e-4 (fc), CrossEntropyLoss (label_smoothing=0.1), bs=32, seed=42, 8 epochs (best epoch 7 by mean of RAF and RAVDESS val Macro-F1)."),
        ("Why chosen", "Residual skip connections support stable partial fine-tuning on CPU without catastrophic forgetting, sustaining real-time frame rates while bridging static-photo and speaking-face domains (evaluated in Section 5.1, Table 5.2 and Table 5.3)."),
        ("Where deployed", "external_models/ravdess/best_resnet18_ravdess.pt (default) and external_models/resnet18_raf/best_resnet18.pt, hot-swappable in backend/services/webcam_service.py and live_resnet18_demo.py."),
    ])

    add_model_spec("3.3.3 EfficientNet-B0 Legacy Webcam Backbone", [
        ("Role in system", "Retained in the runtime backbone switcher as the legacy MELD-trained webcam baseline to quantify cross-domain shift against ResNet-18."),
        ("Architecture and parameter count", "TorchVision EfficientNet-B0 [5] compound-scaled MBConv backbone (4,007,548 parameters, 1,280-D pooled output) coupled to a custom classifier head Dropout(0.35) -> Linear(1280, 256) -> SiLU -> Dropout(0.25) -> Linear(256, 7) (329,735 parameters), totaling 4,337,283 parameters."),
        ("Input and preprocessing", "Aligned square BGR crop (pad=1.25) resized directly to 96x96 RGB, scaled to [0, 1], and normalized with ImageNet mean/std."),
        ("Output", "7-D canonical emotion logits in R^7, temperature-scaled at T=0.85 when post-processed in WebcamService."),
        ("Training and hyperparameters", "Initial MELD face-crop warm-up/fine-tune (train_webcam_model.py), followed by 10 epochs of Phase-3 webcam-augmented fine-tuning (train_webcam_phase3.py, saved at checkpoint epoch 34) using WeightedRandomSampler, FocalLoss (gamma=2.0, label_smoothing=0.05), AdamW (wd=1e-4, backbone_lr=1.5e-5, head_lr=7.5e-5), CosineAnnealingLR (eta_min=1e-6), bs=64, seed=101."),
        ("Why chosen", "Provides a lightweight 96x96 reference baseline illustrating that MELD television face crops generalize poorly to unseen consumer webcams."),
        ("Where deployed", "models/video_classifier/best_webcam_model.pth, selectable as 'efficientnet_current' in backend/services/webcam_service.py."),
    ])

    add_model_spec("3.3.4 ResNet-50 and VideoClassifier (MELD Visual Stream)", [
        ("Role in system", "Extracts visual facial/scene representations from uploaded MELD conversational video clips for unimodal video scoring and multimodal fusion."),
        ("Architecture and parameter count", "Frozen ImageNet-pretrained TorchVision ResNet-50 [4] with classification layer replaced by nn.Identity() (23,508,032 parameters, 2048-D feature output), paired with VideoClassifier MLP (backend/models/video_model.py): Linear(2048, 256) -> ReLU -> Dropout(0.3) -> Linear(256, 7) (526,343 parameters); combined 24,034,375 parameters."),
        ("Input and preprocessing", "8 uniformly spaced frames sampled via torch.linspace(0, frame_count - 1, 8). Each frame receives an aspect-preserving square crop (side=min(h,w), y_start=(h-side)*0.15, centered horizontally), LAB-space CLAHE (clipLimit=2.0, tileGridSize=(8,8)), resize to 224x224 RGB, and ImageNet normalization. Luminance (<35 or >220) and contrast (<15) degrade quality_discount to 0.80."),
        ("Output", "Temporal mean-pooled 2048-D embedding passed through VideoClassifier to yield 7-D logits, softmax probabilities, and visual quality_metrics (later converted to a 16-D vector in Section 3.4)."),
        ("Training and hyperparameters", "ResNet-50 weights are frozen (ResNet50_Weights.DEFAULT). Note: train.py:train_video is a stub without a training loop, so the optimizer, epochs, learning rate, and loss schedule used to produce models/video_classifier/best_video_model.pth are MISSING from repository logs (stored and test metrics are reported in Section 5.2, Table 5.4)."),
        ("Why chosen", "Temporal pooling over 8 CLAHE-normalized head crops enables fast CPU inference on arbitrary-length MP4 uploads without training a 3D video transformer."),
        ("Where deployed", "backend/services/video_service.py (VideoInferenceService) serving POST /predict/video and feeding POST /predict/multimodal."),
    ])

    add_model_spec("3.3.5 Wav2Vec 2.0 Base and StrongerAudioClassifier (MELD Acoustic Stream)", [
        ("Role in system", "Models vocal prosody and acoustic emotion cues from 16 kHz conversational speech waveforms in MELD and live microphone streams."),
        ("Architecture and parameter count", "Frozen self-supervised facebook/wav2vec2-base [15] (94,371,712 parameters: 7-layer CNN feature extractor 4,200,448 + 12-layer Transformer encoder 89,775,488, hidden_dim=768, 12 heads). Production head StrongerAudioClassifier (backend/models/audio_model.py:21–39): Linear(768, 512) -> ReLU -> BatchNorm1d(512) -> Dropout(0.3) -> Linear(512, 256) -> ReLU -> Dropout(0.3) -> Linear(256, 7) (527,879 parameters). Baseline AudioClassifier: Linear(768, 256) -> ReLU -> Dropout(0.3) -> Linear(256, 7) (198,663 parameters)."),
        ("Input and preprocessing", "Audio decoded via soundfile/librosa, averaged to mono, resampled to 16 kHz, DC-offset removed, peak-normalized, trimmed of silence (librosa.effects.trim top_db=30 if >=1600 samples), and capped at 30 s. Wav2Vec2Processor prepares waveforms; last_hidden_state is mask-mean-pooled into a 768-D vector. Duration <0.6 s, RMS <0.008, or clipping >5% sets quality_discount=0.85."),
        ("Output", "7-D acoustic logits, softmax probabilities, and acoustic SNR/RMS/clipping quality_metrics."),
        ("Training and hyperparameters", "In train.py:train_audio, StrongerAudioClassifier is trained on pre-extracted 768-D MELD embeddings (9,988 train / 1,108 dev) with AdamW (lr=1e-3, wd=1e-4), ReduceLROnPlateau (mode='max', factor=0.5, patience=5), bs=32, and WeightedRandomSampler (inverse class frequency). Across results/audio_classifier_improvement/, 5 strategies were benchmarked for 14–28 epochs to train production checkpoint best_audio_model_balanced.pth (per-epoch log for this exact file is MISSING; test outcomes on N=2,610 appear in Section 5.2, Table 5.4)."),
        ("Why chosen", "Self-supervised speech transformer representations capture prosodic pitch and energy dynamics beyond hand-crafted spectral statistics [16], while balanced sampling prevents 47.1% neutral collapse."),
        ("Where deployed", "backend/services/audio_service.py (AudioInferenceService) serving POST /predict/audio, POST /predict/multimodal, and POST /predict/webcam/multimodal."),
    ])

    add_model_spec("3.3.6 all-MiniLM-L6-v2 and TF-IDF Baselines (MELD Linguistic Stream)", [
        ("Role in system", "Encodes spoken or transcribed conversational utterances into semantic vectors for 7-class linguistic emotion classification."),
        ("Architecture and parameter count", "Frozen sentence-transformers/all-MiniLM-L6-v2 [19] (22,713,216 parameters: 6 Transformer layers, hidden_size=384, 12 attention heads). Saved production classifier (models/text_classifier/best_text_model_contextual.pt): sklearn LogisticRegression (C=1.2, class_weight='balanced', solver='lbfgs', max_iter=1000) with coef_ (7, 384) and intercept_ (7,) totaling 2,695 trainable parameters (22,715,911 combined). Legacy baseline (best_text_model.joblib): TfidfVectorizer(ngram_range=(1,3), max_features=12000, sublinear_tf=True) + LogisticRegression(C=1.5, class_weight='balanced', lbfgs) (84,007 classifier weights)."),
        ("Input and preprocessing", "TextPreprocessingUtils.clean_and_normalize (backend/models/text_model.py) replaces smart quotes/apostrophes and expands 19 common English contractions. MiniLM tokenizes with padding=True, truncation=True, max_length=128 and computes attention-mask mean pooling -> 384-D vector. Utterances <=1 word and <5 chars receive quality_discount=0.85."),
        ("Output", "7-D class probability vector via predict_proba(), predicted label, and token/character quality_metrics."),
        ("Training and hyperparameters", "Note: train.py:train_text defines CalibratedClassifierCV(LogisticRegression(C=1.0, class_weight='balanced'), method='sigmoid', cv=3) on 80 samples/emotion (560 total), whereas the saved production bundle best_text_model_contextual.pt stores a direct LogisticRegression(C=1.2, class_weight='balanced') without a CalibratedClassifierCV wrapper (the exact script/split producing C=1.2 and TF-IDF C=1.5 is MISSING from the repo)."),
        ("Why chosen", "Dense 384-D contextual embeddings handle paraphrasing, short conversational turns, and semantic negation at low CPU latency while outperforming sparse 12k n-gram TF-IDF features (comparative MELD test results appear in Section 5.2, Table 5.4)."),
        ("Where deployed", "backend/services/text_service.py (TextInferenceService) paired with TranscriptionService (canonical MELD index + Google Speech ASR) behind /predict/text, /predict/text/file, /predict/multimodal, and /predict/webcam/multimodal."),
    ])

    add_model_spec("3.3.7 AdvancedCrossModalAttentionFusionNet (and LearnedAttentionFusionNet Baseline)", [
        ("Role in system", "Fuses available acoustic, linguistic, and visual probability/logit representations into one calibrated emotion prediction with per-modality reliability weights."),
        ("Architecture and parameter count", "AdvancedCrossModalAttentionFusionNet (backend/models/fusion_model.py:98–217): 167,113 parameters total — three modality projectors Linear(16,96)->LN->GELU->Dropout(0.2) (3x1,824=5,472), learnable modality_type_emb (1,3,96) (288), 2-layer TransformerEncoder (d_model=96, 4 heads, dim_feedforward=192, GELU, dropout=0.2; 149,568), output LayerNorm(96) (192), reliability gate_mlp Linear(98,48)->GELU->Linear(48,1)->Sigmoid (4,801), classification head Linear(96,64)->LN->GELU->Dropout(0.2)->Linear(64,7) (6,791), and learnable scalar consensus_weight (1, learned value 0.5102). Baseline LearnedAttentionFusionNet: 26,920 parameters (14-D input, d=64, 1 MHA layer, 4 heads)."),
        ("Input and preprocessing", "Three 16-D modality feature vectors (Section 3.4) plus a boolean presence mask M in {0,1}^3 indicating active streams."),
        ("Output", "Fused 7-D logits z_fused in R^7 and normalized modality attention/reliability gates g in [0,1]^3 summing to 1.0 across active streams."),
        ("Training and hyperparameters", "Trained in train.py:train_fusion on pre-extracted balanced MELD splits (data/fusion_train_dataset.pt N=210 [30/class]; fusion_val_dataset.pt N=105 [15/class]) for 45 epochs, bs=32, AdamW (lr=1.5e-3, wd=1e-4), CosineAnnealingLR (T_max=45, eta_min=1e-5), CrossEntropyLoss (label_smoothing=0.05), grad clip max_norm=1.0, 15% single-modality dropout, seed=42, selecting minimum validation loss checkpoint (per-epoch log is MISSING)."),
        ("Why chosen", "Models cross-channel dependencies across arbitrary modality subsets (unimodal, bimodal, trimodal) via key_padding_mask while blending learned neural features with centered log-odds consensus."),
        ("Where deployed", "models/fusion_classifier/optimized_fusion_model.pth loaded by backend/services/fusion_service.py (MultimodalFusionService) and evaluated in evaluate.py."),
    ])

    add_model_spec("3.3.8 Post-Hoc Temperature Scaling and Runtime Concordance Calibration", [
        ("Role in system", "Aligns predicted confidence with empirical accuracy without altering argmax ranking, mitigating overconfidence under class imbalance and noisy streams."),
        ("Architecture and parameter count", "Single scalar temperature parameter T > 0 (1 parameter) scaling fusion logits z / T prior to softmax."),
        ("Input and preprocessing", "Uncalibrated 7-D logits z_fused produced by AdvancedCrossModalAttentionFusionNet (or webcam logits in WebcamService)."),
        ("Output", "Calibrated 7-class probability distribution P_cal = softmax(z / T)."),
        ("Training and hyperparameters", "Offline optimization in train.py:344–354 applies bounded scalar minimization (scipy.optimize.minimize_scalar, bounds=(0.10, 2.50), Brent method — not L-BFGS) to minimize validation NLL on fusion_val_dataset.pt (N=105), yielding checkpoint scalar T* for optimized_fusion_model.pth (reported with validation and test curves in Section 5.2; baseline best_fusion_model.pth stores T=1.3490). In live service inference, MultimodalFusionService (fusion_service.py:62–63,196–263) overrides checkpoint T* with concordance-aware T_adaptive in [0.24, 0.28] (0.28 full consensus, 0.24–0.28 partial consensus, 0.26 conflicting) plus concordance logit scaling (1.15x–1.30x)."),
        ("Why chosen", "Single-parameter temperature scaling is computationally free at inference, preserves decision boundaries, and lowers trimodal Expected Calibration Error on both validation and test splits below the 0.10 target (quantified in Section 5.2) [25]."),
        ("Where deployed", "Stored in models/fusion_classifier/optimized_fusion_model.pth (used by evaluate.py) and adapted in backend/services/fusion_service.py; webcam backbones use T=1.0 (ResNet-18) or T=0.85 (EfficientNet-B0)."),
    ])

    # -----------------------------------------------------------------------
    # 3.4 MULTIMODAL CROSS-MODAL ATTENTION FUSION DESIGN & WORKED EXAMPLE
    # -----------------------------------------------------------------------
    add_h2(doc, "3.4 Multimodal Cross-Modal Attention Fusion Design and Worked Example")
    add_para(doc,
             "Figure 3.5 illustrates the complete two-stage fusion architecture implemented in MultimodalFusionService and "
             "AdvancedCrossModalAttentionFusionNet (Section 3.3.7). Rather than concatenating raw 768-D, 384-D, and 2048-D embeddings—which "
             "overfits rapidly on small conversational fusion splits—the architecture operates on a compact, calibrated 16-D per-modality contract.")
    add_figure(doc, "report/figures/fig_diag_meld_fusion.png", "3.5",
               "MELD AdvancedCrossModalAttentionFusionNet and MultimodalFusionService entropy gating and temperature calibration.", width_in=5.65)

    add_bullet(doc, "16-D Per-Modality Feature Contract (fusion_service.py:100–115): ",
               "For each active stream m in {audio, text, video}, _build_16d_feature constructs u_m in R^16 by concatenating: "
               "(i) normalized 7-class probabilities p_m = u_m[0:7]; (ii) centered log-probabilities l_m = ln(clip(p_m, 1e-6, 1.0)) - mean(ln p_m) "
               "at u_m[7:14]; (iii) normalized Shannon certainty c_m = max(0.02, 1 - H_tilde(p_m)) at u_m[14]; and (iv) signal quality discount "
               "q_m in {0.80, 0.85, 1.0} at u_m[15]. Absent streams receive the neutral fallback [1/7,...,1/7, 0,...,0, 0.0, 0.5].")
    add_bullet(doc, "Modality Projections & Transformer Encoder: ",
               "Dedicated linear projectors (see Section 3.3.7 for exact layer dimensions and dropout) map each u_m into shared 96-D space and "
               "add learnable modality_type_emb in R^(1x3x96), after which a 2-layer TransformerEncoder exchanges cross-modal context.")
    add_bullet(doc, "Missing-Modality Masking: ",
               "Boolean presence mask M in {0,1}^3 produces key_padding_mask = ~M so absent channels receive zero attention weight; an explicit "
               "safety guard (key_padding_mask[all_masked, 0] = False) prevents NaN attention outputs if all masks are false.")
    add_bullet(doc, "Entropy Reliability Gate & Masked Normalization: ",
               "Each attended token h_tilde_m in R^96 is concatenated with [c_m, q_m] and scored by gate_mlp to yield raw_g_m. Multiplying by "
               "M_m and dividing by sum_k(raw_g_k * M_k) ensures zero leakage from absent streams and convex reliability weights sum_m g_m = 1.")
    add_bullet(doc, "Dual-Branch Consensus Logit Blend & Calibration: ",
               "A neural head maps the gated representation sum_m g_m * h_tilde_m to z_neural, while a residual consensus branch computes z_cons = "
               "sum_m g_m * l_m directly over centered unimodal log-odds. Learned scalar gamma = clamp(consensus_weight, 0.1, 0.9) = 0.5102 blends "
               "z_fused = (1 - gamma) * z_neural + gamma * z_cons prior to temperature scaling (see Section 3.3.8).")

    add_h3(doc, "3.4.1 Worked Numeric Example on MELD Utterance dia0_utt0")
    add_para(doc,
             "To trace every stage numerically, consider real MELD sample dia0_utt0 (frontend/samples/dia0_utt0.wav and dia0_utt0.mp4, "
             "canonical MELD dev utterance spoken by Phoebe at 00:20:57,256–00:21:00,049: \"Oh my God, he's lost it. He's totally lost it.\"). "
             "All three streams pass signal-quality checks (q_a = q_t = q_v = 1.0): Audio predicts DISGUST (0.2010) with near-uniform "
             "certainty c_a = 0.0252 (SNR 31.9 dB); Text predicts SADNESS (0.5114, c_t = 0.2964); and Video predicts SADNESS (0.9190, "
             "c_v = 0.8269, mean luminance 59.39). Passing u_a, u_t, u_v through AdvancedCrossModalAttentionFusionNet yields raw gates "
             "[0.4558, 0.5947, 0.4204], normalizing to g = [Audio=0.3098, Text=0.4043, Video=0.2858]. Notice that although Video has higher "
             "raw certainty (0.8269 vs. 0.2964), the Transformer and gate MLP assign highest trust to Text (40.43%). Blending z_neural and "
             "z_cons at gamma = 0.5102 yields z_fused[sadness] = +1.0015 (Table 3.2), producing calibrated SADNESS confidence 49.18% at "
             "validation-fitted T* (Section 5.2; and 94.55% under runtime Partial Consensus scaling T_adaptive = 0.2514, 1.25x boost, matching "
             "Figure 4.5). Masking Video out (M = [1, 1, 0]) dynamically renormalizes gates to [A=0.4292, T=0.5708, V=0.0000] and retains "
             "SADNESS (z_fused = +0.4524, P(T*) = 28.46%), confirming graceful missing-modality behavior.")

    dia0_rows = [
        ("Anger", "0.1258 (-0.0689)", "0.0215 (-1.1183)", "0.0000 (-7.8915)", "-0.2960", "-2.7291", "-1.5373", "0.0267", "0.0069", "0.0000"),
        ("Disgust", "0.2010 (+0.3998)", "0.0110 (-1.7884)", "0.0034 (+0.2400)", "+0.4430", "-0.5307", "-0.0538", "0.1175", "0.0835", "0.0053"),
        ("Fear", "0.0592 (-0.8226)", "0.1256 (+0.6468)", "0.0116 (+1.4672)", "+0.1779", "+0.4260", "+0.3045", "0.1682", "0.1525", "0.0218"),
        ("Joy", "0.1761 (+0.2675)", "0.2027 (+1.1254)", "0.0003 (-2.1877)", "-0.1229", "-0.0874", "-0.1048", "0.1117", "0.0767", "0.0043"),
        ("Neutral", "0.1533 (+0.1288)", "0.0154 (-1.4519)", "0.0014 (-0.6473)", "-0.4391", "-0.7322", "-0.5886", "0.0688", "0.0340", "0.0006"),
        ("Sadness", "0.1592 (+0.1666)", "0.5114 (+2.0506)", "0.9190 (+5.8395)", "-0.6113", "+2.5499", "+1.0015", "0.3376", "0.4918", "0.9455"),
        ("Surprise", "0.1255 (-0.0712)", "0.1124 (+0.5358)", "0.0643 (+3.1798)", "-0.5116", "+1.1034", "+0.3123", "0.1695", "0.1545", "0.0225"),
    ]
    add_styled_table(
        doc, "3.2", "Worked Numeric Trace of AdvancedCrossModalAttentionFusionNet on MELD Sample dia0_utt0 (g=[0.3098, 0.4043, 0.2858], gamma=0.5102)",
        ["Emotion", "Audio p (l_a)", "Text p (l_t)", "Video p (l_v)", "z_neural", "z_cons", "z_fused", "P(T=1)", "P(T=T*)", "P_live"],
        dia0_rows, [0.54, 0.84, 0.84, 0.84, 0.46, 0.46, 0.46, 0.42, 0.48, 0.41], font_pt=6.4,
    )

    # -----------------------------------------------------------------------
    # 3.5 CORE ALGORITHMS (PSEUDO-CODE)
    # -----------------------------------------------------------------------
    add_h2(doc, "3.5 Core Algorithms (Pseudo-Code)")
    add_para(doc,
             "Algorithms 3.1 and 3.2 specify the runtime execution contracts for the real-time webcam engine (WebcamService.process_frame) "
             "and the cross-modal attention fusion forward/calibration pipeline (fusion_model.py and fusion_service.py). Offline RAVDESS "
             "crop extraction (prepare_ravdess.py) and mixed-domain training (train_mixed.py) are summarized concisely in Section 3.2 and "
             "Section 3.3.2 without duplicate pseudo-code listings.")

    add_pseudocode_block(doc, "Algorithm 3.1: Unified Webcam Frame Processing & Uncertainty Gating (WebcamService.process_frame)", [
        "Input: BGR frame I, client_id c, mirror flag m (default False), smooth flag s, calibrate flag cal",
        "1.  If m is True (legacy preview only): flip I horizontally.",
        "2.  Run YuNet (Section 3.3.1) on I -> select largest/nearest face F = (x,y,w,h, re, le, nose, rm, lm, conf).",
        "3.  If F is empty: reset EMA buffer and return status = NO_FACE.",
        "4.  Compute eye angle theta = atan2(ley - rey, lex - rex); if |theta| <= 45 deg,",
        "    rotate I around eye midpoint (cx, cy) and extract BORDER_REPLICATE square crop C of side floor(max(w,h)*pad).",
        "5.  Evaluate Laplacian variance v = Var(Laplacian(Gray(C))); hold prior prediction if v < 30.0 or conf < 0.50.",
        "6.  Preprocess C for active backbone (Sections 3.3.2–3.3.3) -> raw logits z in R^7; apply neutral baseline offset if cal active.",
        "7.  Compute softmax p_raw = softmax(z); apply confidence-adaptive EMA p_ema = alpha * p_raw + (1 - alpha) * p_prev if s.",
        "8.  Temperature-scale centered log(p_ema) (T=1.0 for ResNet-18, T=0.85 for EfficientNet-B0), update 3-frame hysteresis.",
        "9.  Check p_top1 < 0.38 or (p_top1 - p_top2) < 0.08 -> set is_uncertain flag and confidence tier (Robust/Moderate/Subtle).",
        "10. Return JSON payload (apparent_expression, confidence_tier, is_uncertain, probabilities, bbox, quality_status).",
    ])

    add_pseudocode_block(doc, "Algorithm 3.2: Cross-Modal Attention Fusion, Masking & Calibration (fusion_model.py / fusion_service.py)", [
        "Input: Optional unimodal results (audio_result, text_result, video_result), learned model weights, T_base = 0.28 (or checkpoint T*)",
        "1.  Initialize active mask M in {0,1}^3; if sum(M) == 0 return error ('No valid modalities').",
        "2.  For each stream m in {audio, text, video}:",
        "    a. If present: extract p_m in R^7, l_m = ln(p_m) - mean(ln p_m), c_m = max(0.02, 1 - H_tilde(p_m)), q_m from quality_metrics.",
        "    b. If absent: set u_m = [1/7,...,1/7, 0,...,0, 0.0, 0.5]; emit missing-modality notice.",
        "3.  Project u_m -> h_m^(0) = Proj_m(u_m) + modality_type_emb[m] in R^96; set key_padding_mask = ~M (guard if all True).",
        "4.  Run 2-layer TransformerEncoder + LayerNorm -> attended tokens h_tilde_m in R^96.",
        "5.  Compute raw_g_m = gate_mlp(concat(h_tilde_m, c_m, q_m)); normalize g_m = (raw_g_m * M_m) / max(sum_k raw_g_k * M_k, 1e-7).",
        "6.  Compute neural branch z_neural = Classifier(sum_m g_m * h_tilde_m) and consensus branch z_cons = sum_m g_m * l_m.",
        "7.  Clamp gamma = clamp(consensus_weight, 0.1, 0.9); compute z_fused = (1 - gamma) * z_neural + gamma * z_cons.",
        "8.  Calibrate: offline eval divides z_fused by fitted scalar T* (Section 5.2); live service applies concordance boost + T_adaptive in [0.24, 0.28].",
        "9.  Apply softmax(z_scaled), resolve Full/Partial/Conflicting agreement narrative, and return MultimodalPredictionResponse.",
    ])

    # -----------------------------------------------------------------------
    # 3.6 MATHEMATICAL MODELLING (ALIGNED & COMPLETE (3.1)-(3.25))
    # -----------------------------------------------------------------------
    add_h2(doc, "3.6 Mathematical Modelling")
    add_para(doc,
             "Every mathematical formulation implemented across unimodal encoding, multimodal attention fusion, temperature calibration, "
             "webcam geometry/smoothing, and evaluation in this project is itemized below in standardized equation numbering (3.1)–(3.25), "
             "verified one-to-one against its rendered equation and source location.")

    add_formula_eq(doc, "3.1", "report/formulas/F1_softmax.png",
                   "z in R^K (K=7) denotes raw class logits, c indexing the 7 discrete emotions, yielding normalized probabilities sum_c p_c = 1 "
                   "(webcam_service.py:765, audio_service.py:142, video_service.py:141).")
    add_formula_eq(doc, "3.2", "report/formulas/F2_cross_entropy_label_smoothing.png",
                   "y is the ground-truth label index, K=7, and eps is the label-smoothing factor (0.10 in mixed ResNet-18 training, 0.05 in MELD "
                   "fusion) regularizing target distributions against noisy labels [26] (train_mixed.py:768, train.py:283).")
    add_formula_eq(doc, "3.3", "report/formulas/F3_focal_loss.png",
                   "alpha_t weights inverse class frequency, p_t is predicted true-class probability, and gamma=2.0 down-weights easy majority "
                   "samples under MELD/webcam imbalance [27] (audio_model.py:41–59, train_webcam_phase3.py:163–192).")
    add_formula_eq(doc, "3.4", "report/formulas/F6_class_balanced_weight.png",
                   "N is dataset size, n_c is sample count of class c, yielding WeightedRandomSampler probabilities inversely proportional to "
                   "class frequency (train_mixed.py:424–431, train.py:100–106).")
    add_formula_eq(doc, "3.5", "report/formulas/F7_input_normalization.png",
                   "x_c in [0,1] are RGB channels normalized by ImageNet channel mean mu=[0.485,0.456,0.406] and std sigma=[0.229,0.224,0.225] "
                   "(train_mixed.py:383, video_service.py:40, webcam_service.py:137).")
    add_formula_eq(doc, "3.6", "report/formulas/F9_conv_relu.png",
                   "W and b denote 2D convolutional filter kernels and biases followed by ReLU nonlinearity across ResNet-18/50 and MLP stages "
                   "(torchvision ResNet-18/50, audio_model.py:13).")
    add_formula_eq(doc, "3.7", "report/formulas/F10_batch_norm.png",
                   "mu_B and sigma_B^2 are mini-batch feature moments standardized with learnable affine parameters gamma, beta (frozen through "
                   "layer3 in mixed ResNet-18 fine-tuning; active in StrongerAudioClassifier; train_mixed.py:465–474, audio_model.py:30).")
    add_formula_eq(doc, "3.8", "report/formulas/F8_resnet_residual_block.png",
                   "F(x, {W_i}) denotes the residual mapping summed with the identity skip connection in ResNet-18 and ResNet-50 [4] "
                   "(train_mixed.py:29, video_service.py:33).")
    add_formula_eq(doc, "3.9", "report/formulas/F18_efficientnet_compound_scaling.png",
                   "phi scales EfficientNet-B0 depth d, width w, and input resolution r under resource constraint alpha*beta^2*gamma^2 approx 2 "
                   "[5] for legacy efficientnet_current (96x96 px, webcam_service.py:140–144).")
    add_formula_eq(doc, "3.10", "report/formulas/F11_adamw_update.png",
                   "m_hat_t, v_hat_t are bias-corrected gradient moments, eta_g is parameter-group learning rate, and lambda=1e-4 is decoupled "
                   "weight decay [28] (train_mixed.py:477–489, train.py:119,284).")
    add_formula_eq(doc, "3.11", "report/formulas/F12_mixed_batch_composition.png",
                   "B=32 total mini-batch size and s_extra=0.2 enforce n_extra=6 RAVDESS crops and n_RAF=26 RAF-DB crops per training step "
                   "(train_mixed.py:618–621).")
    add_formula_eq(doc, "3.12", "report/formulas/F13_aligned_square_crop.png",
                   "(x_R, y_R) and (x_L, y_L) are YuNet eye coordinates defining rotation angle theta (if |theta|<=45 deg), eye midpoint c_eye, "
                   "and padded square crop side S = floor(max(w, h) * pad) (live_resnet18_demo.py:49–81, webcam_service.py:585–586).")
    add_formula_eq(doc, "3.13", "report/formulas/F14_blur_quality_gate.png",
                   "nabla^2 I_gray is the 2D Laplacian operator on grayscale crop C whose spatial variance must meet threshold 30.0 "
                   "(prepare_ravdess.py:131, webcam_service.py:592–593).")
    add_formula_eq(doc, "3.14", "report/formulas/F19_ema_smoothing.png",
                   "p_t is the current softmax vector and alpha_t scales dynamically with top-1 probability p_max,t to dampen single-frame jitter "
                   "(webcam_service.py:770–775).")
    add_formula_eq(doc, "3.15", "report/formulas/F16_uncertain_rule.png",
                   "p_(1) and p_(2) are top-1 and top-2 calibrated probabilities; outputs with p_(1) < 0.38 or margin (p_(1) - p_(2)) < 0.08 "
                   "trigger the is_uncertain warning flag (webcam_service.py:832–846).")
    add_formula_eq(doc, "3.16", "report/formulas/F20_meld_audio_pooling_mlp.png",
                   "h_t in R^768 are Wav2Vec 2.0 hidden frames mask-mean-pooled into e_a in R^768 and classified via StrongerAudioClassifier "
                   "(ReLU, BatchNorm1d(512), Dropout(0.3), 768->512->256->7) [15][16] (audio_service.py:130–142, audio_model.py:21–39).")
    add_formula_eq(doc, "3.17", "report/formulas/F23_meld_16d_contract.png",
                   "p_m in R^7 are normalized modality probabilities, l_tilde_m are centered log-probabilities, c_m in [0.02, 1] is normalized "
                   "Shannon certainty, and q_m in {0.80, 0.85, 1.0} is signal quality discount forming u_m in R^16 (fusion_service.py:100–115).")
    add_formula_eq(doc, "3.18", "report/formulas/F24_meld_transformer_masking.png",
                   "Proj_m maps 16-D stream features to R^96 plus learned modality_type_emb e_type_m; a 2-layer TransformerEncoder with "
                   "key_padding_mask = ~M produces LayerNorm-attended representations H_tilde (fusion_model.py:172–185).")
    add_formula_eq(doc, "3.19", "report/formulas/F21_meld_fusion_equations.png",
                   "H_tilde_m is normalized Shannon entropy, g_m are masked sigmoid reliability gates over [h_tilde_m, c_m, q_m], and z_fused "
                   "blends neural logits with consensus centered log-odds at learned gamma=0.5102 (fusion_model.py:189–216).")
    add_formula_eq(doc, "3.20", "report/formulas/F25_temperature_scaling_nll.png",
                   "T > 0 scales logits before softmax; offline bounded scalar optimization minimizes NLL on the N_val=105 validation split "
                   "to yield optimal scalar T* (evaluated in Section 5.2) [25] (train.py:345–354).")
    add_formula_eq(doc, "3.21", "report/formulas/F22_calibration_metrics.png",
                   "B_b are confidence bins (B=10 in metrics.py) measuring Expected Calibration Error (ECE) and multi-class Brier Score (BS) "
                   "[25] (backend/utils/metrics.py:10–48).")
    add_formula_eq(doc, "3.22", "report/formulas/F4_classification_metrics.png",
                   "TP_c, FP_c, FN_c compute per-class precision, recall, F1, and unweighted Macro-F1 across all K=7 emotion categories "
                   "(train_mixed.py:492–505, train_webcam_phase3.py:212–232).")
    add_formula_eq(doc, "3.23", "report/formulas/F5_confusion_matrix.png",
                   "C_ij counts true class i predicted as j, and C_tilde_ij normalizes each row by total true-class support "
                   "(train_mixed.py:515–521, backend/utils/metrics.py:55).")
    add_formula_eq(doc, "3.24", "report/formulas/F15_majority_and_pooled_acc.png",
                   "y_hat_clip selects clip-level modal class over T_k frames while Acc_pooled averages frame accuracy over all N=916 clip frames "
                   "(eval_clips.py:113–115).")
    add_formula_eq(doc, "3.25", "report/formulas/F17_percentile_latency.png",
                   "L_p orders per-frame CPU millisecond timings Delta t_i over N=200 frames to extract median p50 and tail p95 latency "
                   "(eval_clips.py:207–208).")

    add_h2(doc, "3.7 Tool and Technique Justification")
    add_para(doc,
             "Table 3.3 summarizes the open-source runtime stack and links each model family to its dedicated specification in Section 3.3, "
             "avoiding redundant parameter or metric repetition.")
    tool_rows = [
        ("PyTorch 2.14.0+cpu & TorchVision 0.29.1 [29]", "Core DL Runtime", "Unified autograd, CPU inference mode, and TorchVision backbones (Sections 3.3.2–3.3.4, 3.3.7–3.3.8)."),
        ("OpenCV 5.1.0 DNN", "Vision Runtime", "Runs YuNet ONNX (Section 3.3.1), eye-line affine warping, Laplacian blur gating, and LAB CLAHE."),
        ("Transformers 5.17.0 (HuggingFace)", "Speech & NLP Encoders", "Hosts offline cached Wav2Vec 2.0 Base (Section 3.3.5) and all-MiniLM-L6-v2 (Section 3.3.6)."),
        ("scikit-learn 1.9.1 & Joblib", "Statistical Heads & Metrics", "Hosts text LogisticRegression/TF-IDF classifiers (Section 3.3.6) and F1/confusion metrics."),
        ("Librosa 1.0.0 & SoundFile", "Audio DSP", "16 kHz resampling, silence trimming (top_db=30), RMS/SNR/clipping diagnostics (Section 3.3.5)."),
        ("FastAPI 0.162.0 & Uvicorn", "ASGI Web Server", "Serves 17 REST/WebSocket routes (Figure 3.4, Table 4.3) and zero-build Web Studio SPA."),
        ("SciPy Bounded Optimization", "Calibration Solver", "Minimizes validation NLL via minimize_scalar for post-hoc temperature T* (Section 3.3.8)."),
    ]
    add_styled_table(
        doc, "3.3", "Runtime Ecosystem Justification and Model Cross-Reference Index",
        ["Software / Library Component", "Role", "Technical Justification & Model Section Cross-Reference"],
        tool_rows, [1.65, 1.05, 3.05], font_pt=8.0,
    )

    add_mapped_pos(doc, "PO3 (Design/Development of Solutions), PO5 (Modern Tool Usage).")


# ---------------------------------------------------------------------------
# CHAPTER 4: IMPLEMENTATION AND TESTING
# ---------------------------------------------------------------------------
def build_chapter4(doc: Document) -> None:
    add_chapter_heading(doc, "4", "Implementation and Testing", page_break=True)

    add_h2(doc, "4.1 Step-by-Step System Development")
    add_para(doc,
             "Implementation progressed across both subsystems—real-time webcam vision and MELD conversational multimodal analysis—converting "
             "standalone training scripts into an integrated, test-verified FastAPI application. Rather than restating model layer counts or "
             "preprocessing specs already established in Section 3.3, the subsections below document the engineering integrations across each module.")

    add_h3(doc, "4.1.1 Webcam Pipeline Parity, Quality Gating, and Dynamic Backbone Switching")
    add_para(doc,
             "Early auditing uncovered silent divergences between browser live requests, WebSocket streams, and offline evaluation scripts, as well "
             "as horizontal mirroring and brittle Action Unit (AU) geometry heuristics. All live routes (/predict/webcam/frame, /frame-json, "
             "/ws, /ws/webcam, /multimodal) and eval_clips.py were unified onto WebcamService.process_frame (Algorithm 3.1), achieving exact "
             "HTTP-vs-offline label parity under simulated 320x180 JPEG85 browser transport (verified in Section 5.1). Inference defaults were "
             "fixed to mirror=False (decoupled from CSS preview mirroring), brittle AU override rules were removed after ablation confirmed "
             "stronger validation F1 (Section 5.1), and POST /predict/webcam/backbone was added to hot-swap between resnet18_ravdess (default), "
             "resnet18_raf, and efficientnet_current (Sections 3.3.1–3.3.3) while clearing stale per-client EMA buffers.")

    add_h3(doc, "4.1.2 RAVDESS Crop Curation and Mixed-Domain Fine-Tuning")
    add_para(doc,
             "To bridge static RAF-DB training photographs to dynamic speaking webcam faces without catastrophic forgetting, prepare_ravdess.py "
             "executed an actor-disjoint curation pass separating Actors 01–08 (training) from unseen Actors 09–10 (validation; Figure 3.3). "
             "Mixed-domain fine-tuning in train_mixed.py (Section 3.3.2) then adapted the baseline RAF-DB weights on joint mini-batches and "
             "selected the checkpoint maximizing mean RAF-DB and RAVDESS validation Macro-F1.")

    add_h3(doc, "4.1.3 MELD Acoustic Pipeline and Class-Imbalance Mitigation")
    add_para(doc,
             "MELD training and validation utterances exhibited severe class skew (~47% neutral vs. <2.8% disgust and fear), causing the "
             "baseline AudioClassifier (Section 3.3.5) to suffer majority-class collapse on held-out test categories (quantified in Section 5.2, "
             "Table 5.4). Offline extraction via pipeline.py (--extract audio and --extract wav2vec) invoked ffmpeg to cache MELD embeddings across "
             "train/dev/test splits (9,988 / 1,108 / 2,610). Five imbalance strategies were benchmarked in results/audio_classifier_improvement/ "
             "(A_Baseline, B_Stronger_MLP, C_Weighted_CE, D_Weighted_Sampler, F_Focal_Loss), motivating the inverse-frequency sampled production "
             "checkpoint best_audio_model_balanced.pth integrated with AudioInferenceService quality diagnostics (Section 3.3.5).")

    add_h3(doc, "4.1.4 MELD Linguistic Pipeline and Canonical/ASR Transcription")
    add_para(doc,
             "Linguistic processing migrated from the legacy TF-IDF baseline to dense contextual embeddings inside TextInferenceService "
             "(backend/services/text_service.py; see Section 3.3.6), while retaining automatic fallback loading to best_text_model.joblib if the "
             "contextual bundle is absent. Repository auditing documented an implementation discrepancy between train.py:train_text and the "
             "deployed best_text_model_contextual.pt weights (detailed in Section 3.3.6). For audio-only or video-only uploads lacking an "
             "explicit transcript, TranscriptionService (backend/services/transcription_service.py) indexes 11,132 unique dia{D}_utt{U} keys "
             "across the MELD dev (1,109), test (2,610), and train (9,989) CSVs (unique count reflects split-overlapping dialogue IDs) and falls "
             "back to Google Speech Recognition ASR for non-MELD user recordings.")

    add_h3(doc, "4.1.5 MELD Visual Stream and Temporal Frame Pooling")
    add_para(doc,
             "VideoInferenceService (backend/services/video_service.py; see Section 3.3.4) handles uploaded MP4/AVI/MOV clips via OpenCV frame "
             "sampling, speaker head framing, and illumination-normalized ResNet-50 pooling. At runtime, diagnostic checks monitor exposure and "
             "contrast statistics, flagging degraded visual quality and emitting up to four base64 preview thumbnails for the Web Studio UI. "
             "Note that train.py:train_video remains an unimplemented stub, leaving training curves for best_video_model.pth unlogged.")

    add_h3(doc, "4.1.6 Cross-Modal Attention Fusion Training, Calibration, and Conflict Diagnostics")
    add_para(doc,
             "Fusion training in train.py:train_fusion operates on pre-extracted balanced 16-D MELD partitions (data/fusion_train_dataset.pt, "
             "fusion_val_dataset.pt, fusion_test_dataset.pt), dropping a random modality during training so attention masking and reliability "
             "gates (Sections 3.3.7 and 3.4) learn to re-weight surviving channels, followed by bounded scalar calibration (Section 3.3.8). At "
             "serving time, MultimodalFusionService (backend/services/fusion_service.py) accepts arbitrary 1-, 2-, or 3-modality payloads, falls "
             "back to entropy-weighted Bayesian integration if neural weights are absent, applies runtime concordance scaling, and synthesizes "
             "natural-language agreement rationales (Full Consensus, Partial Consensus, or Conflicting / Divergent).")

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
             "Figure 4.4 presents the Benchmarks & Ablation research view, surfacing the stored MELD ablation table and temperature "
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
               "Linguistic Modality Studio (all-MiniLM-L6-v2 + LogisticRegression) prediction interface.", width_in=5.35)

    add_para(doc,
             "Figure 4.8 displays the Session Prediction History and Audit Log view stored strictly client-side in session memory for review or export.")
    add_figure(doc, "report/figures/ui_history.png", "4.8",
               "Session Prediction History and Audit Log view inside the Web Studio UI.", width_in=5.35)

    add_h2(doc, "4.3 Experimental Setup and Training Hyperparameters")
    add_para(doc,
             "All training, benchmarking, and server tests were conducted on the CPU workstation detailed in Table 4.1 (verified via platform, "
             "Win32 CIM queries, and active .venv package inspection). Training hyperparameters across all webcam and MELD unimodal/fusion "
             "models are consolidated in Table 4.2, with unlogged checkpoint fields marked MISSING.")
    add_csv_table(
        doc, "4.1", "Verified Hardware and Software Execution Environment Specifications",
        "report/tables/environment.csv",
        col_widths=[0.85, 1.35, 2.35, 1.20], font_pt=8.2,
    )
    hp_rows = [
        ("resnet18_ravdess", "128 (eval 146)", "AdamW (wd=1e-4)", "2e-5 (layer4) / 2e-4 (fc)", "CE (smooth=0.1)", "8 (best ep 7)", "32 (26 RAF+6 RAV)", "train_mixed.py / train_ravdess.log"),
        ("resnet18_raf", "128 (eval 146)", "AdamW (wd=1e-4)", "1e-4 (bb) / 1e-3 (fc)", "CE (smooth=0.0)", "10 (stop 9, best 6)", "32", "train_raf_resnet18.py / config.json"),
        ("efficientnet_current", "96 (96)", "AdamW (wd=1e-4)", "1.5e-5 (bb) / 7.5e-5 (fc)", "Focal (g=2.0, sm=0.05)", "10 ph3 (ckpt ep 34)", "64", "train_webcam_phase3.py"),
        ("MELD Audio (StrongerMLP)", "768-D vec", "AdamW (wd=1e-4)", "1e-3 + Plateau(0.5,p=5)", "CE + BalancedSampler", "14–28 exp (ckpt MISSING)", "32", "train.py / audio_classifier_improvement/"),
        ("MELD Text (MiniLM / TF-IDF)", "384-D / 12k", "L-BFGS LogReg", "C=1.2 / C=1.5 (bal.)", "Multinomial LogLoss", "Closed-fit (script MISSING)", "N/A", "best_text_model_contextual.pt / .joblib"),
        ("MELD Video (ResNet-50 MLP)", "2048-D (224px)", "MISSING (stub)", "MISSING (stub)", "MISSING (stub)", "MISSING (stub)", "MISSING", "train.py:172 (stub); best_video_model.pth"),
        ("MELD Fusion Baseline", "14-D vec", "MISSING", "MISSING", "MISSING", "MISSING (T=1.3490)", "MISSING", "best_fusion_model.pth"),
        ("MELD Fusion Optimized", "16-D vec", "AdamW (wd=1e-4)", "1.5e-3 + Cosine(1e-5)", "CE (sm=0.05) + 15% Drop", "45 (ep curve MISSING)", "32", "train.py:250–376 / optimized_fusion_model.pth"),
    ]
    add_styled_table(
        doc, "4.2", "Training Hyperparameters Across Webcam Backbones and MELD Unimodal / Fusion Models",
        ["Model", "Input Size", "Optimizer", "LR / Reg", "Loss / Sampling", "Epochs / Best", "Batch", "Source"],
        hp_rows, [0.95, 0.62, 0.72, 0.82, 0.76, 0.68, 0.50, 0.70], font_pt=6.7,
    )

    add_h2(doc, "4.4 FastAPI REST and WebSocket Endpoints")
    add_para(doc,
             "Table 4.3 inventories all 17 REST and WebSocket endpoints registered across backend/main.py and backend/api/*.py (visualized in "
             "Figure 3.4), together with input schemas and structured JSON response contracts.")
    add_csv_table(
        doc, "4.3", "Complete List of FastAPI REST and WebSocket Endpoints",
        "report/tables/api_endpoints.csv",
        col_widths=[0.46, 1.25, 0.92, 0.92, 1.00, 1.20], font_pt=7.4,
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

    add_h2(doc, "5.1 Datasets and Real-Time Webcam Evaluation")
    add_para(doc,
             "Empirical evaluation spans four benchmarks across the shared seven-emotion taxonomy (Table 5.1): RAF-DB [2] (10,428 train / "
             "1,841 val / 3,068 test after removing 2 exact within-train duplicates), actor-disjoint RAVDESS [3] speech crops (3,184 Actors 01–08 "
             "train / 786 Actors 09–10 val), MELD [1] (N=2,610 held-out unimodal test; 105-sample balanced fusion validation and test splits, "
             "15/class), and 9 personal webcam evaluation clips (916 frames, evaluation-only).",
             space_after=4.0)
    add_csv_table(
        doc, "5.1", "Datasets Summary Across Training, Validation, and Evaluation Splits",
        "report/tables/datasets_summary.csv",
        col_widths=[0.95, 1.05, 0.95, 1.15, 0.70, 0.95], font_pt=7.2,
    )

    add_para(doc,
             "Across the live vision pipeline (Sections 3.3.1–3.3.3), unifying REST, WebSocket, and offline evaluation on WebcamService.process_frame "
             "verified 916/916 HTTP-vs-disk frame label parity across all 9 evaluation clips, while removing brittle Action Unit geometry overrides "
             "raised MELD validation macro-F1 from 0.2872 to 0.3868 (+0.0996, results/phase2_ablations.json). As summarized in Table 5.2, replacing "
             "legacy EfficientNet-B0 (15.61% pooled webcam, 1/9 clips; 20.4% RAF-DB) with RAF-trained ResNet-18 lifted webcam accuracy to 44.00% "
             "(4/9 clips; 79.53% RAF-DB), and mixed RAF-DB + RAVDESS fine-tuning (resnet18_ravdess) reached 57.31% pooled accuracy (525/916 frames, "
             "5/9 clips), raised RAF-DB macro-F1 from 0.6721 to 0.7077 (79.27% accuracy), and achieved 56.87% accuracy (0.5354 macro-F1) on unseen "
             "RAVDESS actors. Across the 9 evaluation clips, the confusion matrix (Figure 5.1) and clip-level logs confirm 100.0% accuracy on "
             "Disgust (2).mp4, Joy.mp4, and Neutal.mp4 (100% Joy and Neutral recall; 93.1% Disgust recall overall), 79.78% on disgust.mp4, and "
             "43.75% (majority pass) on Fear (2).mp4, whereas Anger (61.4% surprise, 38.6% neutral) and Sadness (100% neutral) remain unsolved "
             "single-frame failure cases. In the unified 200-frame CPU latency benchmark (Table 5.3), all three backbones held 31.50–33.63 ms median "
             "(p50) and 33.26–35.82 ms tail (p95) latency (~29.7–31.8 FPS), well within the 150 ms budget.",
             space_after=4.0)
    add_csv_table(
        doc, "5.2", "Webcam Backbones Across RAF-DB, RAVDESS, 9 Clips, and Unified CPU Latency",
        "report/tables/backbone_comparison.csv",
        col_indices=[0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        rename_headers=["Backbone", "Arch.", "Px", "RAF Acc", "RAF F1", "RAV Acc", "RAV F1", "Clips", "Pooled", "p50 ms", "p95 ms"],
        col_widths=[0.88, 0.62, 0.34, 0.48, 0.48, 0.48, 0.48, 0.45, 0.50, 0.52, 0.52], font_pt=6.8,
    )
    add_figure(doc, "report/figures/fig_cm_webcam_resnet18_ravdess.png", "5.1",
               "Final 7x7 row-normalized webcam confusion matrix: default ResNet-18 (RAF+RAVDESS) on 9 clips (57.31% pooled, N=916).", width_in=3.75)
    add_csv_table(
        doc, "5.3", "Unified 200-Frame CPU Latency on Disgust (2).mp4 (Earlier Single Runs Non-Comparable)",
        "report/tables/latency_benchmark.csv",
        col_indices=[0, 1, 2, 7, 8, 9, 10, 11, 12, 13, 14],
        rename_headers=["Backbone", "Arch.", "Px", "Mean", "Std", "p50", "p95", "p99", "Min", "Max", "FPS(p50)"],
        col_widths=[0.94, 0.72, 0.34, 0.45, 0.42, 0.46, 0.46, 0.46, 0.46, 0.46, 0.58], font_pt=6.8,
    )

    add_h2(doc, "5.2 MELD Unimodal and Multimodal Benchmark, Ablation, and Calibration")
    add_para(doc,
             "Table 5.4 consolidates all MELD unimodal and multimodal fusion benchmarks across held-out evaluation splits using exact checkpoint "
             "and service evaluation (metrics from backend/utils/metrics.py, results/audio_classifier_improvement/comparison.csv, "
             "results/video_metrics/classification_report.txt, results/baseline_audit/baseline_performance.json, and "
             "results/metrics/multimodal_optimization_comparison.json). To resolve an inconsistency present in earlier project tables, two "
             "distinct fusion baselines are separated cleanly on data/fusion_test_dataset.pt (N=105): (1) the Pre-Optimization System Baseline "
             "(baseline_performance.json: Acc 29.52%, Macro-F1 0.2833, ECE 0.2130, Brier 0.8811, NLL 1.9835), which is the canonical baseline for "
             "optimization gains; and (2) the 14-D LearnedAttentionFusionNet checkpoint (best_fusion_model.pth, T=1.3490), whose 14-D test slice "
             "achieves 30.48% accuracy, 0.2488 Macro-F1, ECE 0.0520 (0.0606 unscaled), and Brier 0.8317 (earlier tables had paired 30.48%/0.2488 "
             "test accuracy/F1 with the checkpoint's stored validation ECE of 0.0713 at 34.29% accuracy).",
             space_after=4.0)

    meld_full_rows = [
        ("Audio Baseline (Sec. 3.3.5)", "AudioClassifier (768->256->7)", "Test Emb.", "2,610", "0.4923", "0.1771", "0.3973", "0.0702", "0.6819"),
        ("Audio Exp B: Stronger MLP", "StrongerAudioClassifier", "Test Emb.", "2,610", "0.4900", "0.1986", "0.4138", "—", "—"),
        ("Audio Exp C: Weighted CE", "StrongerAudioClassifier", "Test Emb.", "2,610", "0.2686", "0.2110", "0.3025", "—", "—"),
        ("Audio Exp D: Balanced Sampler", "StrongerAudioClassifier", "Test Emb.", "2,610", "0.2854", "0.2041", "0.3201", "—", "—"),
        ("Audio Exp F: Focal Loss", "StrongerAudioClassifier", "Test Emb.", "2,610", "0.4866", "0.1972", "0.4109", "—", "—"),
        ("Audio Balanced (Production)", "StrongerMLP + Sampler", "Test Emb.", "2,610", "0.3287", "0.2209", "0.3580", "0.0578", "0.7844"),
        ("Text Baseline (Sec. 3.3.6)", "12k TF-IDF + LogReg (C=1.5)", "MELD Test", "2,610", "0.4015", "0.2989", "0.4196", "0.0595", "0.7387"),
        ("Text Contextual (Production)", "MiniLM + LogReg (C=1.2)", "MELD Test", "2,610", "0.4287", "0.3326", "0.4725", "0.1073", "0.7215"),
        ("Video Unimodal (Ckpt Report)", "ResNet-50 + VideoClassifier", "Ckpt Subset", "35", "0.3714", "0.3684", "0.3684", "MISSING", "MISSING"),
        ("Video Raw Stream (Fusion Test)", "X_v[:, :7] unimodal logits", "Fusion Test", "105", "0.0857", "0.0900", "0.0900", "0.4972", "1.1752"),
        ("Pre-Opt Fusion Baseline", "Baseline system (audit JSON)", "Fusion Test", "105", "0.2952", "0.2833", "0.2833", "0.2130", "0.8811"),
        ("LearnedAttentionFusionNet", "14-D Baseline (T=1.3490)", "Fusion Test", "105", "0.3048", "0.2488", "0.2488", "0.0520", "0.8317"),
        ("Optimized Fusion (Uncalibrated)", "AdvCrossModal (T=1.0)", "Fusion Test", "105", "0.3619", "0.3574", "0.3574", "0.0972", "0.7803"),
        ("Optimized Fusion (Scaled)", "AdvCrossModal (T*=0.5953)", "Fusion Test", "105", "0.3619", "0.3574", "0.3574", "0.0734", "0.7785"),
    ]
    add_styled_table(
        doc, "5.4", "Complete MELD Unimodal and Multimodal Fusion Benchmark Results Across Held-Out Splits",
        ["Model / Config", "Architecture / Setup", "Eval Split", "N", "Acc", "MacroF1", "W-F1", "ECE", "Brier"],
        meld_full_rows, [1.18, 1.15, 0.60, 0.36, 0.45, 0.50, 0.47, 0.52, 0.52], font_pt=6.8,
    )

    add_para(doc,
             "On the acoustic channel (Table 5.4), why does baseline accuracy (0.4923) exceed balanced production accuracy (0.3287) while Macro-F1 "
             "improves from 0.1771 to 0.2209? Inspection of results/metrics/classification_report.csv reveals severe majority-class collapse: MELD "
             "test contains 1,256/2,610 (48.1%) neutral utterances versus 68 disgust (2.6%), 50 fear (1.9%), and 208 sadness (8.0%). The un-balanced "
             "baseline reached 49.23% accuracy primarily by recalling 91.08% of neutral samples (0.6698 neutral F1) while scoring 0.0000 F1 across "
             "disgust, fear, and sadness. Applying inverse-frequency WeightedRandomSampler in best_audio_model_balanced.pth sacrificed neutral recall "
             "(91.08% -> 42.04%) to restore minority predictions (anger 0.2554 -> 0.3434, sadness 0.0000 -> 0.2079, surprise 0.1570 -> 0.2479, "
             "disgust 0.0498, fear 0.0088) and lowered acoustic ECE (0.0702 -> 0.0578). On text, dense contextual all-MiniLM-L6-v2 embeddings "
             "improved accuracy (40.15% -> 42.87%) and Macro-F1 (0.2989 -> 0.3326) over 12k TF-IDF; stored checkpoint dicts on un-normalized text "
             "show 0.4134/0.3074 (TF-IDF) and 0.4276/0.3334 (MiniLM).",
             space_after=4.0)

    add_para(doc,
             "Table 5.5 and Figure 5.2 present the full seven-configuration modality ablation (Audio, Video, Text, all three bimodal pairs, and "
             "Trimodal A+V+T) evaluated on the balanced 105-sample fusion test set (15/class) using optimized_fusion_model.pth (T* = 0.5953, "
             "results/full_ablation_benchmark_results.json). Single-stream Video Only collapses to 6.67% accuracy (0.4125 ECE), whereas Audio Only "
             "reaches 20.95% (0.1725 ECE) and Text Only reaches 37.14% (0.3650 Macro-F1). Combining Audio + Text matches top accuracy (37.14%) "
             "with the highest overall Macro-F1 (0.3670) and lowers ECE from 0.1903 to 0.1146. Full Trimodal fusion (36.19% accuracy, 0.3574 Macro-F1) "
             "achieves +6.67 pp accuracy and +0.0741 Macro-F1 over the Pre-Optimization Baseline (29.52% / 0.2833) while delivering the lowest "
             "Expected Calibration Error (0.0734) and NLL (1.7244) of all tested configurations.",
             space_after=4.0)

    ablation7_rows = [
        ("Audio Only", "[T, F, F]", "0.2095", "0.2022", "0.2022", "0.3576", "0.1725", "0.8655", "1.9861", "0.35"),
        ("Video Only", "[F, F, T]", "0.0667", "0.0681", "0.0681", "0.4792", "0.4125", "1.0758", "2.5764", "0.04"),
        ("Text Only", "[F, T, F]", "0.3714", "0.3650", "0.3650", "0.5580", "0.1903", "0.8200", "1.9435", "0.03"),
        ("Audio + Video", "[T, F, T]", "0.1714", "0.1618", "0.1618", "0.3320", "0.1606", "0.8745", "1.9841", "0.03"),
        ("Audio + Text", "[T, T, F]", "0.3714", "0.3670", "0.3670", "0.4492", "0.1146", "0.7706", "1.7354", "0.14"),
        ("Video + Text", "[F, T, T]", "0.3524", "0.3394", "0.3394", "0.4620", "0.1128", "0.8140", "1.8302", "0.03"),
        ("Audio + Video + Text (Trimodal)", "[T, T, T]", "0.3619", "0.3574", "0.3574", "0.4007", "0.0734", "0.7785", "1.7244", "0.19"),
    ]
    add_styled_table(
        doc, "5.5", "Full 7-Configuration Multimodal Ablation Benchmark on Balanced MELD Test Set (N=105, T*=0.5953)",
        ["Configuration", "Mask [A,T,V]", "Acc", "MacroF1", "W-F1", "Conf", "ECE-10", "Brier", "NLL", "Lat (ms)"],
        ablation7_rows, [1.28, 0.56, 0.48, 0.52, 0.48, 0.46, 0.50, 0.50, 0.50, 0.47], font_pt=6.8,
    )
    add_figure(doc, "report/figures/fig_ablation_meld_7config.png", "5.2",
               "Real 7-configuration MELD multimodal ablation benchmark (N=105, T*=0.5953): Accuracy, Macro-F1, and ECE.", width_in=5.55)

    add_para(doc,
             "Figure 5.3 plots the empirical reliability curves generated directly from optimized_fusion_model.pth across validation (N=105) "
             "and held-out test (N=105) splits. Bounded validation NLL minimization yields T* = 0.5953 (0.595251), reducing validation ECE from "
             "0.1377 to 0.0668 (NLL 1.5829 -> 1.5226, Brier 0.7305 -> 0.7052) and reducing held-out test ECE from 0.0972 (uncalibrated T=1.0) "
             "to 0.0734 (Brier 0.7803 -> 0.7785), a 65.5% relative ECE reduction below the 0.2130 pre-optimization baseline (0.2130 -> 0.0734). "
             "Crucially, why is Trimodal fusion accuracy (36.19%) slightly below Text Only (37.14%) while calibration improves? In MELD TV scenes, "
             "rapid cuts and background speakers inject severe visual noise (Video-Only Acc 6.67%, ECE 0.4125), flipping one marginal utterance "
             "compared to Text Only; however, combining multi-stream representations with Shannon-entropy gating tempers overconfident unimodal text "
             "logits (lowering mean confidence from 55.80% in Text Only at T*=0.5953 to 40.07% in Trimodal), cutting ECE by 61.4% (0.1903 -> "
             "0.0734) and improving NLL (1.9435 -> 1.7244). Applying trimodal T* < 1.0 to masked single streams sharpens their single-stream "
             "logits (e.g., Text Only ECE 0.0801 unscaled vs. 0.1903 scaled).",
             space_after=4.0)
    add_figure(doc, "report/figures/fig_calibration_reliability.png", "5.3",
               "Reliability diagrams for optimized_fusion_model.pth at T*=0.5953 on Validation (ECE 0.1377 -> 0.0668) and Held-Out Test (ECE 0.0972 -> 0.0734; vs. Pre-Opt Baseline 0.2130).",
               width_in=5.55)

    add_h2(doc, "5.3 Fusion Confusion Matrix, Gate Allocation, Missing Modality, and Conflict Analysis")
    add_para(doc,
             "Figure 5.4 and Table 5.6 detail per-class performance on data/fusion_test_dataset.pt (N=105, 15/class). Trimodal fusion excels on "
             "high-cue expressions—surprise (0.6667 F1, 9/15 recall, 0.7500 precision), neutral (0.4444 F1, 8/15 recall), and joy (0.4242 F1, "
             "7/15 recall)—while low-arousal sadness exhibits the steepest confusion (0.1538 F1: 5 sadness utterances predicted as disgust and 4 as "
             "neutral), and fear (0.2963 F1) and anger (0.2308 F1) surpass Audio Only (+0.2296 and +0.1508 F1) and Video Only.",
             space_after=4.0)
    add_figure(doc, "report/figures/fig_cm_meld_fusion_optimized.png", "5.4",
               "MELD fusion test confusion matrix: AdvancedCrossModalAttentionFusionNet (Acc 36.19%, Macro-F1 0.3574, T*=0.5953, N=105).",
               width_in=3.95)

    per_class_rows = [
        ("Anger (15)", "0.2727 / 0.2000 / 0.2308", "0.3333 / 0.2000 / 0.2500", "0.1000 / 0.0667 / 0.0800", "0.0000 / 0.0000 / 0.0000"),
        ("Disgust (15)", "0.2500 / 0.3333 / 0.2857", "0.2500 / 0.3333 / 0.2857", "0.1739 / 0.2667 / 0.2105", "0.1765 / 0.2000 / 0.1875"),
        ("Fear (15)", "0.3333 / 0.2667 / 0.2963", "0.2500 / 0.2667 / 0.2581", "0.0667 / 0.0667 / 0.0667", "0.0000 / 0.0000 / 0.0000"),
        ("Joy (15)", "0.3889 / 0.4667 / 0.4242", "0.3750 / 0.4000 / 0.3871", "0.2727 / 0.2000 / 0.2308", "0.0417 / 0.0667 / 0.0513"),
        ("Neutral (15)", "0.3810 / 0.5333 / 0.4444", "0.4286 / 0.6000 / 0.5000", "0.2222 / 0.1333 / 0.1667", "0.2000 / 0.0667 / 0.1000"),
        ("Sadness (15)", "0.1818 / 0.1333 / 0.1538", "0.2000 / 0.1333 / 0.1600", "0.2500 / 0.3333 / 0.2857", "0.0000 / 0.0000 / 0.0000"),
        ("Surprise (15)", "0.7500 / 0.6000 / 0.6667", "0.7692 / 0.6667 / 0.7143", "0.3529 / 0.4000 / 0.3750", "0.1429 / 0.1333 / 0.1379"),
        ("Macro Avg (105)", "0.3654 / 0.3619 / 0.3574", "0.3723 / 0.3714 / 0.3650", "0.2055 / 0.2095 / 0.2022", "0.0801 / 0.0667 / 0.0681"),
    ]
    add_styled_table(
        doc, "5.6", "Per-Class Precision / Recall / F1 on Balanced MELD Fusion Test Set (data/fusion_test_dataset.pt, N=105)",
        ["Emotion (Supp.)", "Trimodal (A+V+T) P/R/F1", "Text-Only P/R/F1", "Audio-Only P/R/F1", "Video-Only P/R/F1"],
        per_class_rows, [0.95, 1.25, 1.18, 1.18, 1.19], font_pt=6.8,
    )

    add_para(doc,
             "Inspection of learned gates (g_a, g_t, g_v) and consensus scalar gamma = 0.5102 (Table 5.7) demonstrates that "
             "AdvancedCrossModalAttentionFusionNet automatically learned the true unimodal reliability hierarchy—assigning highest mean weight to "
             "Text (0.4059), intermediate weight to Audio (0.3570), and lowest weight to noisy TV Video (0.2371)—without hard-coded modality priors. "
             "Under missing-modality masking, key_padding_mask and masked gate normalization smoothly re-allocate all probability mass across "
             "surviving streams with zero absent-stream leakage (e.g., Audio + Text [0.4701, 0.5299, 0.0000]; Video + Text [0.0000, 0.6295, 0.3705]). "
             "Crucially, partitioning test samples by unimodal argmax agreement reveals that on the 60 hardest Conflicting / Divergent cases (57.1% of "
             "the test set where all three unimodal predictions disagree), Trimodal fusion achieves 38.33% accuracy—outperforming Text Only (33.33%, "
             "+5.00 pp), Audio Only (16.67%), and Video Only (5.00%). Table 5.8 traces three real conflict instances from the repository.",
             space_after=4.0)

    gate_rows = [
        ("Trimodal [T, T, T]", "N=105 (All)", "0.3570 +- 0.0252 [0.289, 0.416]", "0.4059 +- 0.0308 [0.322, 0.490]", "0.2371 +- 0.0382 [0.179, 0.328]", "Fusion 36.19% (ECE 0.0734)"),
        ("Audio + Text [T, T, F]", "N=105 (Masked V)", "0.4701", "0.5299", "0.0000 (masked)", "Fusion 37.14% (Macro-F1 0.3670)"),
        ("Audio + Video [T, F, T]", "N=105 (Masked T)", "0.5959", "0.0000 (masked)", "0.4041", "Fusion 17.14%"),
        ("Video + Text [F, T, T]", "N=105 (Masked A)", "0.0000 (masked)", "0.6295", "0.3705", "Fusion 35.24%"),
        ("Full Consensus (A=T=V)", "n=5 (4.8%)", "0.3635", "0.4004", "0.2361", "Fusion 20.0% = A/T/V 20.0%"),
        ("Partial Consensus (2 of 3)", "n=40 (38.1%)", "0.3538", "0.4018", "0.2444", "Fusion 35.0% | T 40.0% | A 35.0% | V 12.5%"),
        ("Conflicting (All 3 Differ)", "n=60 (57.1%)", "0.3585", "0.4091", "0.2324", "Fusion 38.33% > T 33.33% (+5.0 pp)"),
    ]
    add_styled_table(
        doc, "5.7", "Learned Modality Gate Allocation, Missing-Modality Masking, and Conflict-Category Performance (N=105)",
        ["Subset / Mask", "Support", "Audio Gate", "Text Gate", "Video Gate", "Accuracy / Outcome"],
        gate_rows, [1.05, 0.72, 0.92, 0.92, 0.92, 1.22], font_pt=6.7,
    )

    conflict_rows = [
        ("Test #2 (fusion_test_dataset.pt)", "anger", "anger (0.216)", "surprise (0.274)", "joy (0.945)", "[0.4047, 0.3951, 0.2002]", "anger (0.2691) — Overturns 94.5% wrong Video"),
        ("Test #19 (fusion_test_dataset.pt)", "disgust", "neutral (0.216)", "fear (0.262)", "joy (0.645)", "[0.3762, 0.3938, 0.2300]", "disgust (0.3365) — Recover truth when all 3 wrong"),
        ("dia0_utt0 (frontend/samples/)", "sadness", "disgust (0.2010)", "sadness (0.5114)", "sadness (0.9190)", "[0.3098, 0.4043, 0.2858]", "sadness (0.4918 T*; 0.9455 live Partial Cons.)"),
    ]
    add_styled_table(
        doc, "5.8", "Real Verified Conflict Examples Resolved by AdvancedCrossModalAttentionFusionNet (T*=0.5953)",
        ["Sample Source", "Truth", "Audio Pred.", "Text Pred.", "Video Pred.", "Gates [A, T, V]", "Fused Prediction & Behavior"],
        conflict_rows, [0.95, 0.48, 0.72, 0.74, 0.74, 0.92, 1.20], font_pt=6.6,
    )
    add_para(doc,
             "Similarly, on Test Sample #36 (truth fear), conflicting inputs Audio sadness (0.260), Text anger (0.279), and overconfident Video joy "
             "(0.904) receive gates [0.3632, 0.4331, 0.2037] and resolve correctly to fear (0.2663), illustrating how neural cross-attention plus "
             "entropy-aware gating prevents extreme unimodal false positives from dominating decisions.",
             space_after=4.0)

    add_h2(doc, "5.4 Objectives Audit, Economic BOM, Benchmark Positioning, and Failure Defense")
    add_para(doc,
             "Tables 5.9, 5.10, and 5.11 audit Chapter 1 objectives, consolidate the zero-cloud open-source Bill of Materials, and position the "
             "system fairly against GPU-only literature benchmarks (POSTER++ 92.21% [9], DDAMFN 91.35% [10], TelME ~67% [24], and UniMSE 65.51% [23]).",
             space_after=4.0)
    obj_rows = [
        ("Obj 1: Balanced MELD Unimodal", "Mitigate audio collapse; MiniLM vs TF-IDF", "Audio F1 0.1771->0.2209; MiniLM 42.87% vs 40.15%", "Met"),
        ("Obj 2: Calibrated Attention Fusion", "Masking, conflicts, ECE < 0.10", "Masking verified; 38.3% conflict acc; ECE 0.2130->0.0734; T-only > Acc", "Partly Met"),
        ("Obj 3: Webcam Parity & Quality Gate", "Unify routes/eval; align+blur; remove AU", "916/916 parity; AU rules removed (0.2872->0.3868)", "Met"),
        ("Obj 4: Cross-Domain Webcam & CPU", "Mixed adaptation; p95 <= 150 ms", "57.31% (5/9), RAF F1 0.7077, p95 34.59 ms; anger/sad 0%", "Partly Met"),
        ("Obj 5: Privacy & Auto QA", "Volatile RAM + full automated tests", "5/5 unit + 11/11 integration pass; zero disk write live", "Met"),
    ]
    add_styled_table(
        doc, "5.9", "Audit of Achieved Metrics Against Chapter 1 Objectives",
        ["Objective", "Target Criterion", "Observed Empirical Outcome", "Status"],
        obj_rows, [1.25, 1.50, 2.25, 0.75], font_pt=7.2,
    )
    add_csv_table(
        doc, "5.10", "Bill of Materials (Open-Source Software, Weights, Datasets, and Hardware Placeholders)",
        "report/tables/bill_of_materials.csv",
        col_widths=[0.78, 1.05, 1.42, 0.34, 0.86, 1.30], font_pt=7.0,
    )
    sota_rows = [
        ("POSTER++ [9] / DDAMFN [10]", "RAF-DB Test (GPU)", "92.21% / 91.35% Acc", "Heavy static landmark-ViT; non-CPU real-time"),
        ("Ours: ResNet-18 (RAF+RAVDESS)", "RAF-DB / RAV / 9 Clips", "79.27% / 56.87% / 57.31%", "~30 FPS laptop CPU, aligned + dynamic speech transfer"),
        ("TelME [24] / UniMSE [23]", "MELD Test (Full conv.)", "~67% / 65.51% Acc", "Multi-turn history + imbalanced neutral prior"),
        ("Ours: Text / Opt. Fusion", "MELD (2,610 / 105 bal.)", "42.87% / 36.19% (ECE 0.0734)", "Single utterance, CPU, missing-stream masking"),
    ]
    add_styled_table(
        doc, "5.11", "Concise Positioning Against Verified Published Benchmarks",
        ["Method", "Benchmark / Split", "Reported Metrics", "Scope Difference vs. This System"],
        sota_rows, [1.35, 1.25, 1.35, 1.80], font_pt=7.2,
    )

    add_bullet(doc, "MELD Visual Dilution, Minority Audio Ceiling & Single-Turn Scope: ",
               "Television video cuts depress unimodal video accuracy (6.67% on fusion test) and keep sadness F1 at 0.1538, while turn-by-turn "
               "inference lacks conversational history compared to multi-turn models [23][24] and audio fear/disgust remain severely limited (<0.05 F1).")
    add_bullet(doc, "Single-Speaker Webcam Bias & Anger/Sadness Failures: ",
               "Nine single-subject clips served for both backbone selection and evaluation alongside a 0.2205 RAVDESS train-val actor gap; under "
               "overhead lighting, anger mimics static surprise (61.39%) and low-arousal sadness collapses to neutral (100%).")

    add_mapped_pos(doc, "PO4 (Conduct Investigations of Complex Problems).")


# ---------------------------------------------------------------------------
# CHAPTER 6: CONCLUSIONS AND FUTURE WORK
# ---------------------------------------------------------------------------
def build_chapter6(doc: Document) -> None:
    add_chapter_heading(doc, "6", "Conclusions and Future Work", page_break=True)

    add_h2(doc, "6.1 Summary and Key Conclusions")
    add_para(doc,
             "This project delivered an integrated, privacy-preserving local Emotion Recognition system bridging conversational MELD multimodal "
             "analysis and real-time webcam facial classification inside one FastAPI architecture and interactive Web Studio UI. Across the MELD "
             "multimodal subsystem (Sections 5.2–5.3), inverse-frequency balanced acoustic sampling mitigated minority-class neutral collapse at the "
             "expected accuracy-versus-Macro-F1 tradeoff, contextual sentence embeddings improved linguistic modeling over sparse TF-IDF, and "
             "AdvancedCrossModalAttentionFusionNet learned to weight channels according to empirical reliability (Text > Audio > Video) while "
             "gracefully re-normalizing missing-modality subsets via Transformer padding masks.")
    add_para(doc,
             "Crucially, ablation and conflict analyses showed that while noisy television video caused trimodal accuracy to trail pure Text Only "
             "slightly, cross-modal gating beat every individual stream on fully conflicting utterances and post-hoc temperature scaling cut calibration "
             "error well below the target threshold across both validation and held-out test splits. In parallel, across the webcam subsystem "
             "(Section 5.1), enforcing strict API-to-offline preprocessing parity, eye-line affine alignment, Laplacian blur checks, removal of "
             "brittle Action Unit heuristics, and mixed RAF-DB + aligned RAVDESS fine-tuning delivered large cross-domain accuracy gains at real-time "
             "laptop CPU speed without losing static RAF-DB discrimination.")

    add_h2(doc, "6.2 Future Research and Engineering Directions")
    add_para(doc,
             "Several promising extensions remain unimplemented in the current codebase and represent concrete avenues for future work across both subsystems:")
    add_bullet(doc, "Speaker-Aware Conversational MELD Memory: ",
               "Extending MELD multimodal fusion with cross-utterance dialogue context windows [21][24] and speaker-turn embeddings to close the "
               "gap with conversation-level benchmarks while preserving missing-modality masking and subset-aware temperature calibration.")
    add_bullet(doc, "Face-Tracked MELD Visual Filtering: ",
               "Applying YuNet speaker face detection and active-speaker tracking inside VideoInferenceService (Section 3.3.4) to reject background "
               "characters and abrupt scene cuts that dilute trimodal visual accuracy.")
    add_bullet(doc, "Held-Out Personal Webcam Adaptation and Multi-Subject Evaluation: ",
               "Collecting diverse multi-speaker webcam benchmarks under natural illumination and exploring opt-in temporal speaker calibration "
               "(beyond the current resting-neutral logit offset) to resolve anger/surprise and sadness/neutral failures.")
    add_bullet(doc, "Learned Temporal Vision Modeling & HSEmotion Backbones: ",
               "Replacing single-frame EMA probability filtering with lightweight 1–2 second temporal convolutions or attention over facial sequences, "
               "alongside evaluating pretrained HSEmotion/EmotiEffNet weights [11][12] within CPU latency budgets.")

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
               "A curated, viva-focused selection of 20 high-signal PNG figures (200 dpi), 10 curated CSV tables, and all 22 mathtext formula "
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
