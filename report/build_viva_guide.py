"""Build report/Viva_Guide.docx from real project code, configs, logs, and report/ assets."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import docx
from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Cm, Inches, Pt, RGBColor

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "report"


# ---------------------------------------------------------------------------
# DOCUMENT & STYLING HELPERS
# ---------------------------------------------------------------------------
def configure_a4_layout(doc: Document) -> None:
    sec = doc.sections[0]
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    sec.left_margin = Inches(0.9)
    sec.right_margin = Inches(0.9)
    sec.top_margin = Inches(0.85)
    sec.bottom_margin = Inches(0.85)
    sec.different_first_page_header_footer = True

    footer = sec.footer
    fp = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_field(fp, "PAGE", font_size=10.5)


def add_field(paragraph, field_text: str, font_size: float = 11.0) -> None:
    run_begin = paragraph.add_run()
    run_begin.font.name = "Times New Roman"
    run_begin.font.size = Pt(font_size)
    fld1 = OxmlElement("w:fldChar")
    fld1.set(qn("w:fldCharType"), "begin")
    run_begin._r.append(fld1)

    run_instr = paragraph.add_run()
    run_instr.font.name = "Times New Roman"
    run_instr.font.size = Pt(font_size)
    it = OxmlElement("w:instrText")
    it.set(qn("xml:space"), "preserve")
    it.text = f" {field_text} "
    run_instr._r.append(it)

    run_sep = paragraph.add_run()
    fld2 = OxmlElement("w:fldChar")
    fld2.set(qn("w:fldCharType"), "separate")
    run_sep._r.append(fld2)

    run_val = paragraph.add_run("1")
    run_val.font.name = "Times New Roman"
    run_val.font.size = Pt(font_size)

    run_end = paragraph.add_run()
    fld3 = OxmlElement("w:fldChar")
    fld3.set(qn("w:fldCharType"), "end")
    run_end._r.append(fld3)


def setup_styles(doc: Document) -> None:
    settings_el = doc.settings.element
    if settings_el.find(qn("w:updateFields")) is None:
        settings_el.append(parse_xml(r'<w:updateFields %s w:val="true"/>' % nsdecls("w")))

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)
    normal.font.color.rgb = RGBColor(15, 23, 42)
    normal.paragraph_format.line_spacing = 1.15
    normal.paragraph_format.space_after = Pt(5)
    normal.paragraph_format.space_before = Pt(0)

    for lvl, sz, sb, sa, pb in [
        ("Heading 1", 16, 10, 6, True),
        ("Heading 2", 13.5, 10, 4, False),
        ("Heading 3", 12, 7, 3, False),
    ]:
        st = doc.styles[lvl]
        st.font.name = "Times New Roman"
        st.font.size = Pt(sz)
        st.font.bold = True
        st.font.color.rgb = RGBColor(20, 48, 88)
        st.paragraph_format.line_spacing = 1.15
        st.paragraph_format.space_before = Pt(sb)
        st.paragraph_format.space_after = Pt(sa)
        st.paragraph_format.keep_with_next = True
        st.paragraph_format.page_break_before = pb


def add_h1(doc: Document, title: str, page_break: bool = True) -> None:
    p = doc.add_paragraph(style="Heading 1")
    p.paragraph_format.page_break_before = page_break
    r = p.add_run(title)
    r.font.name = "Times New Roman"
    r.font.size = Pt(16)
    r.bold = True


def add_h2(doc: Document, title: str) -> None:
    p = doc.add_paragraph(style="Heading 2")
    r = p.add_run(title)
    r.font.name = "Times New Roman"
    r.font.size = Pt(13.5)
    r.bold = True


def add_h3(doc: Document, title: str) -> None:
    p = doc.add_paragraph(style="Heading 3")
    r = p.add_run(title)
    r.font.name = "Times New Roman"
    r.font.size = Pt(12)
    r.bold = True


def add_p(doc: Document, text: str, bold: bool = False, italic: bool = False,
          size: float = 12.0, align=WD_ALIGN_PARAGRAPH.LEFT, sa: float = 5.0) -> None:
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(sa)
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(size)
    r.bold = bold
    r.italic = italic


def add_bullet(doc: Document, lead: str, body: str, size: float = 11.5) -> None:
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(3)
    r1 = p.add_run(lead)
    r1.font.name = "Times New Roman"
    r1.font.size = Pt(size)
    r1.bold = True
    r2 = p.add_run(body)
    r2.font.name = "Times New Roman"
    r2.font.size = Pt(size)


def add_concept_triad(doc: Document, name: str, plain: str, tech: str, code_loc: str) -> None:
    add_h3(doc, name)
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.left_indent = Inches(0.18)
    p.paragraph_format.space_after = Pt(6)
    for idx, (tag, val) in enumerate([
        ("1) Say Aloud: ", plain),
        ("\n2) Technical Depth: ", tech),
        ("\n3) Where in Code: ", code_loc),
    ]):
        r_tag = p.add_run(tag)
        r_tag.font.name = "Times New Roman"
        r_tag.font.size = Pt(11.5)
        r_tag.bold = True
        r_val = p.add_run(val)
        r_val.font.name = "Consolas" if idx == 2 else "Times New Roman"
        r_val.font.size = Pt(10.0 if idx == 2 else 11.5)


def add_script_box(doc: Document, title: str, text: str, bg: str = "F3F6FA") -> None:
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.45)
    cell._tc.get_or_add_tcPr().append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{bg}"/>'))
    tbl._tbl.tblPr.append(parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        '<w:top w:val="single" w:sz="6" w:color="2B4C7E"/>'
        '<w:bottom w:val="single" w:sz="6" w:color="2B4C7E"/>'
        '<w:left w:val="single" w:sz="12" w:color="2B4C7E"/>'
        '<w:right w:val="single" w:sz="4" w:color="CBD5E1"/>'
        "</w:tblBorders>"
    ))
    p = cell.paragraphs[0]
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(3)
    rt = p.add_run(title + "\n")
    rt.font.name = "Times New Roman"
    rt.font.size = Pt(11.5)
    rt.bold = True
    rb = p.add_run(text)
    rb.font.name = "Times New Roman"
    rb.font.size = Pt(11.5)
    sp = doc.add_paragraph()
    sp.paragraph_format.space_after = Pt(4)


def add_img(doc: Document, rel_path: str, caption: str, width_in: float = 5.8) -> None:
    p_img = REPO_ROOT / rel_path
    assert p_img.is_file(), f"Missing image: {p_img}"
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(p_img), width=Inches(min(width_in, 6.4)))

    pc = doc.add_paragraph()
    pc.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pc.paragraph_format.space_after = Pt(6)
    rc = pc.add_run(caption)
    rc.font.name = "Times New Roman"
    rc.font.size = Pt(10.0)
    rc.italic = True


def add_table(doc: Document, title: str, headers: Sequence[str], rows: Sequence[Sequence[str]],
              widths: Sequence[float], font_pt: float = 9.0) -> None:
    pt = doc.add_paragraph()
    pt.paragraph_format.space_before = Pt(5)
    pt.paragraph_format.space_after = Pt(2)
    pt.paragraph_format.keep_with_next = True
    rt = pt.add_run(title)
    rt.font.name = "Times New Roman"
    rt.font.size = Pt(11.0)
    rt.bold = True

    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    table._tbl.tblPr.append(parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        '<w:top w:val="single" w:sz="6" w:color="334155"/>'
        '<w:bottom w:val="single" w:sz="6" w:color="334155"/>'
        '<w:left w:val="single" w:sz="4" w:color="64748B"/>'
        '<w:right w:val="single" w:sz="4" w:color="64748B"/>'
        '<w:insideH w:val="single" w:sz="4" w:color="94A3B8"/>'
        '<w:insideV w:val="single" w:sz="4" w:color="94A3B8"/>'
        "</w:tblBorders>"
    ))
    for j, h in enumerate(headers):
        table.cell(0, j).text = str(h)
    for i, r in enumerate(rows, start=1):
        for j, v in enumerate(r):
            table.cell(i, j).text = str(v)

    for r_idx, row in enumerate(table.rows):
        trPr = row._tr.get_or_add_trPr()
        if r_idx == 0:
            trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
        trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
        for c_idx, cell in enumerate(row.cells):
            cell.width = Inches(widths[c_idx])
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            tcPr = cell._tc.get_or_add_tcPr()
            if r_idx == 0:
                tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="E2E8F0"/>'))
            for p in cell.paragraphs:
                p.paragraph_format.line_spacing = 1.1
                p.paragraph_format.space_before = Pt(1)
                p.paragraph_format.space_after = Pt(1)
                for run in p.runs:
                    run.font.name = "Times New Roman"
                    run.font.size = Pt(font_pt)
                    if r_idx == 0:
                        run.bold = True

    sp = doc.add_paragraph()
    sp.paragraph_format.space_after = Pt(4)


def add_csv(doc: Document, title: str, csv_rel: str, widths: Sequence[float],
            font_pt: float = 8.0, col_indices: Sequence[int] | None = None,
            rename_headers: Sequence[str] | None = None) -> None:
    with open(REPO_ROOT / csv_rel, newline="", encoding="utf-8") as f:
        data = list(csv.reader(f))
    hdrs = rename_headers if rename_headers else ([data[0][i] for i in col_indices] if col_indices else data[0])
    rows = [[r[i] for i in col_indices] if col_indices else r for r in data[1:]]
    add_table(doc, title, hdrs, rows, widths, font_pt=font_pt)


def add_q_item(doc: Document, qnum: int, qtext: str, short_ans: str, deep_ans: str,
               pointer: str, diff: str, like: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    rq = p.add_run(f"Q{qnum}. {qtext} ")
    rq.font.name = "Times New Roman"
    rq.font.size = Pt(11.5)
    rq.bold = True
    rm = p.add_run(f"[{diff} | {like}]\n")
    rm.font.name = "Times New Roman"
    rm.font.size = Pt(10.0)
    rm.italic = True
    rs = p.add_run("Answer: " + short_ans)
    rs.font.name = "Times New Roman"
    rs.font.size = Pt(11.2)
    if deep_ans:
        rd_lbl = p.add_run("\nDeeper note: ")
        rd_lbl.font.name = "Times New Roman"
        rd_lbl.font.size = Pt(10.8)
        rd_lbl.bold = True
        rd = p.add_run(deep_ans)
        rd.font.name = "Times New Roman"
        rd.font.size = Pt(10.8)
    rp_lbl = p.add_run("\nWhere: ")
    rp_lbl.font.name = "Times New Roman"
    rp_lbl.font.size = Pt(10.0)
    rp_lbl.bold = True
    rp = p.add_run(pointer)
    rp.font.name = "Consolas"
    rp.font.size = Pt(9.2)


# ---------------------------------------------------------------------------
# PRELIMS
# ---------------------------------------------------------------------------
def build_prelims(doc: Document) -> None:
    add_p(doc, "END-TO-END VIVA & TECHNICAL STUDY GUIDE", bold=True, size=20.0, align=WD_ALIGN_PARAGRAPH.CENTER, sa=10.0)
    add_p(doc, "Unified Multimodal (MELD) and Real-Time Webcam Emotion Recognition System",
          bold=True, size=14.5, align=WD_ALIGN_PARAGRAPH.CENTER, sa=14.0)
    add_script_box(
        doc,
        "How to Use This Guide (Zero Fluff, 100% Verified Against Project Code & Results)",
        "Every number, file path, function name, and threshold here comes directly from the actual repository code, "
        "checkpoints, results/ logs, and report/tables/*.csv. Whenever an original empirical threshold or scope choice lacks "
        "written justification in logs/code, it is flagged honestly with [VERIFY] so you never bluff in front of examiners.",
        bg="EEF4FA",
    )
    add_p(doc, "Team Members Covered: [TO BE FILLED: Student Name 1, Student Name 2, Student Name 3]",
          size=11.5, align=WD_ALIGN_PARAGRAPH.CENTER, sa=6.0)
    add_p(doc, "Environment: Python 3.14.4 | PyTorch 2.14.0+cpu | OpenCV 5.1.0 | FastAPI 0.162.0 | Intel Core i7-12650H (CPU Only)",
          italic=True, size=10.5, align=WD_ALIGN_PARAGRAPH.CENTER, sa=14.0)

    add_h1(doc, "Table of Contents", page_break=True)
    ptoc = doc.add_paragraph()
    add_field(ptoc, 'TOC \\o "1-3" \\h \\z \\u', font_size=11.0)
    add_p(doc, "Note: If opening raw DOCX in Word and page numbers look blank, press Ctrl+A then F9 to refresh fields.",
          italic=True, size=9.5, sa=4.0)


# ---------------------------------------------------------------------------
# SECTION 1: PROJECT IN 60 SECONDS & 5 MINUTES + TEAM TEMPLATES
# ---------------------------------------------------------------------------
def build_sec1(doc: Document) -> None:
    add_h1(doc, "1. Project in 60 Seconds and 5 Minutes", page_break=True)

    add_script_box(
        doc,
        "1.1 The 60-Second Spoken Pitch (Memorize This First)",
        "\"Our project is a completely local, privacy-preserving Emotion Recognition platform covering two connected parts under seven "
        "shared emotions (anger, disgust, fear, joy, neutral, sadness, and surprise): a real-time webcam facial expression detector and a MELD "
        "multimodal conversational analyzer for audio, text, and uploaded video.\n"
        "On the webcam side, we first fixed critical bugs where live API endpoints diverged from offline tests, stripped brittle hand-coded "
        "geometry rules that had dragged MELD validation macro-F1 down from 0.3868 to 0.2872, and trained a mixed RAF-DB + RAVDESS speech-crop "
        "ResNet-18 model using YuNet eye-line alignment and Laplacian blur checks. That raised accuracy on our 9 personal evaluation clips from "
        "15.6% (1/9 clips) with EfficientNet-B0 and 44.0% (4/9 clips) with RAF-only ResNet-18 up to 57.3% (5/9 clips), while running around "
        "30 frames per second on laptop CPU alone (~33 ms median). On MELD multimodal conversations, cross-modal Transformer attention fusion "
        "with temperature calibration cut calibration error (ECE) from 0.2130 down to 0.0734.\n"
        "We also openly recognize the limits: those 9 single-speaker clips acted as both selection and test data, anger and sadness remain 0% "
        "on those clips, and multimodal trimodal accuracy (36.19%) trails clean text alone (37.14%) because MELD video noise dilutes text.\"",
    )

    add_script_box(
        doc,
        "1.2 The 5-Minute Detailed Viva Walkthrough",
        "1. Problem: Standard emotion models fail outside their datasets. A webcam EfficientNet-B0 checkpoint trained earlier only got 15.6% "
        "frame accuracy (1/9 clips) on our laptop clips and 20.4% on RAF-DB test because of domain shift and live-vs-offline crop mismatches. "
        "Meanwhile, MELD conversations are heavily imbalanced (47.1% neutral utterances), making naive audio models collapse toward neutral "
        "(0.1771 macro-F1) and leaving multimodal probabilities badly overconfident (ECE = 0.2130).\n\n"
        "2. Approach (Webcam Subsystem): First, we consolidated all live FastAPI routes (/frame, /frame-json, /ws, /ws/webcam) and offline "
        "scripts (eval_clips.py) onto one single engine: WebcamService.process_frame() on unmirrored 320x180 JPEG85 frames (only CSS mirrors "
        "the video preview for natural user feel). YuNet detects the face and 5 landmarks; aligned_square_crop() rotates the eyes horizontal; "
        "Laplacian variance < 30 rejects motion blur. Then, using prepare_ravdess.py, we extracted 3,970 aligned speech crops from 518 RAVDESS "
        "video clips (middle 30-70%, 5 fps, Actors 01-08 train / 09-10 val) and fine-tuned ResNet-18 (train_mixed.py) mixing 26 balanced RAF-DB "
        "crops + 6 RAVDESS crops per batch of 32, freezing stages up to layer3 and adding webcam-realistic augmentations like synthetic glasses.\n\n"
        "3. Approach (MELD Multimodal Subsystem): We extract 768-D Wav2Vec 2.0 speech embeddings into StrongerAudioClassifier (527,879 params, "
        "WeightedRandomSampler raising macro-F1 0.1771 -> 0.2209), 384-D all-MiniLM-L6-v2 text embeddings (42.87% accuracy vs 40.15% TF-IDF), "
        "and 8-frame mean-pooled ResNet-50 video embeddings (526,343 params). Each stream passes a 16-D vector [7 probs, 7 centered logits, "
        "entropy certainty, quality score] into AdvancedCrossModalAttentionFusionNet (167,113 params), which masks missing streams dynamically.\n\n"
        "4. Key Numbers & Honest Caveats: On 200 controlled CPU frames, all 3 webcam backbones run at 31.5-33.6 ms p50 (~30 FPS). ResNet-18 "
        "RAF+RAVDESS hits 79.27% acc / 0.7077 macro-F1 on RAF-DB test (up from 0.6721), 56.87% / 0.5354 on unseen RAVDESS Actors 09-10, and "
        "57.31% pooled (5/9 clips) on personal webcam clips. On MELD balanced fusion test (N=105), ECE drops from 0.2130 to 0.0734 at 36.19% acc. "
        "Everything executes locally in volatile RAM with zero cloud uploads. Honestly, anger (0%) and sadness (0%) fail on our 9 single-user "
        "clips, fear/disgust remain fragile, and live webcam screenshots in the UI tab were captured in standby without physical hardware.",
    )

    add_h2(doc, "1.3 One-Page Team Role Speaking Templates")
    role_rows = [
        ("[TO BE FILLED: Member 1]", "Multimodal Fusion, Calibration & FastAPI Backend",
         "\"I built the MELD multimodal fusion engine (fusion_model.py, fusion_service.py), temperature calibration, FastAPI routers (backend/api/*.py), "
         "and tests/run_tests.py (16/16 pass). My core finding is that cross-modal attention with missing-modality masking cut fusion ECE from 0.2130 to 0.0734.\""),
        ("[TO BE FILLED: Member 2]", "Real-Time Webcam Vision, Parity Fix & Mixed Fine-Tuning",
         "\"I unified live and offline inference in WebcamService (916/916 parity on 320x180 JPEG85), fixed the mirror coordinate split, built "
         "prepare_ravdess.py (3,970 crops), and trained mixed ResNet-18 (train_mixed.py), raising webcam accuracy from 15.6% (1/9) to 57.3% (5/9) at 30 FPS.\""),
        ("[TO BE FILLED: Member 3]", "Speech, NLP Models, Frontend Web Studio & Benchmarking",
         "\"I handled MELD acoustic class-imbalance experiments (Wav2Vec2 macro-F1 0.1771 -> 0.2209), contextual MiniLM text classification (42.87%), "
         "the zero-build 7-tab Web Studio UI (frontend/js/app.js), unified 200-frame CPU latency runs, and evaluation tables/charts.\""),
    ]
    add_table(doc, "Table 1.1: Speaking Roles Template for Each Team Member",
              ["Team Member", "Area Owned", "30-Second Opening Script"], role_rows, [1.35, 1.55, 3.55], font_pt=9.0)


# ---------------------------------------------------------------------------
# SECTION 2: SYSTEM OVERVIEW & CODE MAP
# ---------------------------------------------------------------------------
def build_sec2(doc: Document) -> None:
    add_h1(doc, "2. System Overview and Code Map", page_break=True)
    add_concept_triad(
        doc,
        "2.1 How the Whole Platform Fits Together",
        "One FastAPI server connects five local CPU AI services—three offline MELD streams plus fusion, and one real-time YuNet+CNN webcam stream—to a plain HTML/JS Web Studio.",
        "When you run python main.py --serve, Uvicorn starts backend.main:app on 127.0.0.1:8000. On startup, the lifespan context eagerly loads "
        "AudioInferenceService, TextInferenceService, VideoInferenceService, MultimodalFusionService, TranscriptionService, and WebcamService. "
        "All models run strictly on CPU (torch.cuda.is_available() == False) and map outputs into the canonical 7-class index order in config.yaml: "
        "anger(0), disgust(1), fear(2), joy(3), neutral(4), sadness(5), surprise(6).",
        "main.py -> run_server(); backend/main.py -> app / lifespan(); config.yaml -> emotions",
    )
    add_img(doc, "report/figures/fig_diag_system_architecture.png",
            "Figure 2.1: System architecture integrating MELD multimodal services, WebcamService, FastAPI, and UI.", width_in=5.8)

    code_map = [
        ("main.py / config.yaml", "Master CLI (--serve, --demo, --test, --audio/--video/--text) & shared YAML config", "User terminal / all backend services"),
        ("backend/main.py", "FastAPI app setup, model preloading in lifespan, /health endpoint, mounts API & static UI", "uvicorn / FastAPI ASGI server"),
        ("backend/api/webcam.py", "REST & WebSocket routes: /status, /backbone, /frame, /frame-json, /ws, /ws/webcam, /calibrate, /multimodal", "frontend/js/app.js / HTTP clients"),
        ("backend/services/webcam_service.py", "SINGLE live/offline engine: decode, YuNet detect, aligned crop, quality gate, CNN forward, EMA, hysteresis, uncertain check", "webcam.py, eval_clips.py, test_parity_harness.py"),
        ("live_resnet18_demo.py", "Checkpoint loader load_model() (reads prep & classes) and aligned_square_crop() affine geometry helper", "webcam_service.py (_load_resnet18_raf, process_frame)"),
        ("backend/api/{audio,text,video,multimodal}.py", "Upload/text REST endpoints calling MELD unimodal services & MultimodalFusionService", "frontend/js/app.js & main.py presets"),
        ("backend/services/{audio,text,video}_service.py", "Wav2Vec2 (768-D) + StrongerMLP, all-MiniLM-L6-v2/TF-IDF, ResNet-50 8-frame pooled video classifier", "multimodal.py, webcam_service (multimodal stream)"),
        ("backend/services/fusion_service.py", "Builds 16-D vectors per stream, applies attention mask, T_adaptive & concordance boost, generates explanation", "multimodal.py, webcam_service.py"),
        ("backend/models/{audio,video,text,fusion}_model.py", "PyTorch nn.Module definitions: AudioClassifier, StrongerAudioClassifier, FocalLoss, VideoClassifier, LearnedAttentionFusionNet, AdvancedCrossModalAttentionFusionNet", "Training & inference services"),
        ("train_raf_resnet18.py", "Trains ResNet-18 baseline on RAF-DB (DATASET/) with duplicate audit & CosineAnnealingLR", "Standalone training run -> best_resnet18.pt"),
        ("prepare_ravdess.py", "Extracts aligned blurry-gated middle 30-70% speech crops at 5 fps from Video speech dataset/ -> ravdess_crops/", "Standalone prep CLI before train_mixed.py"),
        ("train_mixed.py", "Mixed balanced fine-tuning (RAF-DB + RAVDESS extra_share=0.2, --freeze-until layer3) -> best_resnet18_ravdess.pt", "Standalone training CLI"),
        ("train.py / evaluate.py / train_webcam_phase3.py", "MELD audio/video/fusion trainers + calibration & ablation evaluators + legacy Phase-3 EfficientNet-B0 script", "Offline CLI scripts"),
        ("eval_clips.py / test_parity_harness.py", "Evaluates webcam backbones across New folder/*.mp4 and verifies live-API vs offline 916/916 parity", "CLI verification scripts"),
        ("frontend/{index.html,css/style.css,js/app.js}", "Zero-build 7-tab Web Studio SPA: WS/REST capture, mirrored <video>, unmirrored <canvas> overlay, research & history tabs", "Browser user on http://127.0.0.1:8000/"),
        ("tests/{run_tests.py,test_*.py}", "Automated 5 unit suites + 11 live ASGI integration tests checking models, parity, and HTTP error handling", "python main.py --test / tests/run_tests.py --all"),
    ]
    add_table(doc, "Table 2.1: Complete Repository Code Map (File -> What It Does -> Who Calls It)",
              ["File / Module", "Responsibility", "Called By"], code_map, [1.80, 2.85, 1.80], font_pt=8.2)


# ---------------------------------------------------------------------------
# SECTION 3: LIVE WEBCAM FLOW, STEP BY STEP
# ---------------------------------------------------------------------------
def build_sec3(doc: Document) -> None:
    add_h1(doc, "3. Live Webcam Flow, Step by Step (Browser to Label)", page_break=True)
    add_p(doc,
          "Teachers almost always ask you to trace a single live webcam frame end to end. Memorize Steps 3.1 to 3.16 below—every resolution, "
          "JPEG quality, coordinate flip, matrix op, and threshold matches the live codebase verbatim.")
    add_img(doc, "report/figures/fig_diag_live_inference.png",
            "Figure 3.1: Real-time webcam frame pipeline from browser capture through YuNet, aligned crop, backbone, EMA, hysteresis, and overlay.", width_in=5.8)

    steps = [
        ("Step 3.1 — Browser Mirroring & Unmirrored Capture (frontend)",
         "User sees a natural mirrored selfie video via CSS, while JavaScript draws raw unmirrored pixels onto a hidden 320x180 JPEG canvas.",
         "In style.css (#webcam-video), transform: scaleX(-1) visually flips only the preview so raising your right hand moves right on screen. "
         "In captureAndProcessWebcamFrame(), an offscreen snapCanvas of fixed 320x180 pixels calls sctx.drawImage(video, 0, 0, 320, 180) and encodes "
         "snap.toDataURL('image/jpeg', 0.85). Because canvas capture reads the raw HTMLVideoElement stream (not CSS transforms), the server "
         "always receives an UNMIRRORED image matching training faces.",
         "frontend/css/style.css:1826-1843; frontend/js/app.js -> captureAndProcessWebcamFrame() (lines 1359-1368)",
         "Input: Camera MediaStream | Output: data:image/jpeg;base64,... (320x180, Q=0.85)"),

        ("Step 3.2 — Client ID & WebSocket / HTTP Transport",
         "Each browser session generates a random client_id and sends the JPEG over an open WebSocket, falling back to POST /frame-json if needed.",
         "Browser holds webcamState.clientId = 'webcam_client_' + Math.floor(Math.random() * 1e6). Payload JSON carries "
         "{frame, client_id, apply_smoothing: true, calibrate}. If webcamState.ws.readyState === WebSocket.OPEN, ws.send(JSON.stringify(payload)) "
         "streams over WS /predict/webcam/ws (avoids HTTP headers per frame). Otherwise it falls back to POST /predict/webcam/frame-json (and "
         "multipart uploads hit POST /predict/webcam/frame).",
         "frontend/js/app.js:937, 1379-1399; backend/api/webcam.py -> websocket_webcam_stream() & predict_webcam_frame_json()",
         "Input: JSON payload | Output: Parsed ASGI frame message -> WebcamService.process_frame()"),

        ("Step 3.3 — Per-Client State & FPS Tracking",
         "The server looks up or initializes an isolated state dictionary in RAM for that client_id so concurrent users never mix smoothing buffers.",
         "_get_client_state(client_id) stores ema_probs, last_bbox, last_landmarks, baseline_logits, calibration_buffer, last_good_prediction, "
         "current_label/candidate_count (hysteresis), and rolling 30-entry history. Inter-frame delta updates rolling_fps (default 15.0 before "
         "two valid intervals).",
         "backend/services/webcam_service.py -> _get_client_state(), process_frame() (lines 337-358, 458-476)",
         "Input: client_id string | Output: Mutable per-client dict in server memory"),

        ("Step 3.4 — Base64 Decoding & Color Space (BGR)",
         "The server strips the data-URL prefix, base64-decodes bytes, and decodes JPEG pixels directly into an OpenCV BGR array.",
         "decode_frame() checks str/bytes/ndarray, splits on comma if 'data:image/jpeg;base64,...', calls base64.b64decode, wraps uint8 buffer, "
         "and runs cv2.imdecode(arr, cv2.IMREAD_COLOR). In OpenCV, color channels arrive as Blue-Green-Red (BGR) shape (H, W, 3) = (180, 320, 3). "
         "Note: mirror flag on inference is NOT applied (raw unmirrored inference).",
         "backend/services/webcam_service.py -> decode_frame() (lines 380-402)",
         "Input: Base64 data URL or raw bytes | Output: uint8 BGR ndarray (180, 320, 3)"),

        ("Step 3.5 — YuNet Face Detection & Landmark Tracking",
         "YuNet locates the face box and 5 facial key points, locking onto the same person's face if multiple people enter the frame.",
         "self.face_detector (OpenCV FaceDetectorYN initialized from models/face_detection_yunet_2023mar.onnx, score_threshold=0.55, "
         "nms_threshold=0.3, top_k=5000) sets input size to (w, h) and runs detect(img). Each row has 15 numbers: [x, y, w, h, rx, ry, lx, ly, "
         "nx, ny, rmx, rmy, lmx, lmy, score]. If >1 face and last_bbox exists, face_dist picks nearest center to prevent jumping; else largest box. "
         "If zero faces, it clears ema_probs/last_bbox and returns apparent_expression='NO_FACE' with uniform 1/7 probabilities.",
         "backend/services/webcam_service.py -> _init_face_detector() (lines 272-290), process_frame() (lines 526-578)",
         "Input: BGR ndarray | Output: bbox [fx,fy,fw,fh], 5 landmarks, face_score"),

        ("Step 3.6 — Geometric Alignment: aligned_square_crop()",
         "Using the left and right eye coordinates, an affine warp levels head tilt horizontally and cuts a padded square face crop.",
         "aligned_square_crop(img, best_face, pad, min_side=10) reads right/left eye (rex, rey, lex, ley), computes in-plane roll "
         "angle = degrees(atan2(ley - rey, lex - rex)). If |angle| > 45 deg it sets angle = 0.0 (avoids wild flips on outlier landmarks). It rotates "
         "the full frame around eye midpoint (cx, cy) using cv2.warpAffine(..., BORDER_REPLICATE), transforms box center via the affine matrix, "
         "computes square side = int(max(w, h) * pad) (pad=1.10 for ResNet-18 checkpoints matching RAF-DB framing, 1.25 otherwise), pads edges with "
         "BORDER_REPLICATE if needed, and slices rotated[y0:y0+side, x0:x0+side].",
         "live_resnet18_demo.py -> aligned_square_crop() (lines 49-81); webcam_service.py:584-588",
         "Input: BGR frame + YuNet 15-vec | Output: Level square BGR face crop"),

        ("Step 3.7 — Quality Gate & Mild Gamma Lighting Check",
         "Blurry or low-confidence crops reuse the last good prediction instead of letting corrupted pixels flip the emotion label.",
         "Converts face_crop to gray and computes Laplacian variance (cv2.Laplacian(gray, CV_64F).var()). Flags: is_blurry (<30.0), "
         "is_low_conf (score < 0.50), is_too_small (w<36 or h<36), is_off_angle (|tilt| >= 45 deg), low_light (<35 mean luminance), "
         "overexposed (>225), low_contrast (std<15). Mild gamma LUT adjusts dark (mean_lum<60 -> gamma=1.35) or bright (>195 -> gamma=0.75) "
         "crops (NOT CLAHE, because comment notes CLAHE shadows nasolabial folds and causes false disgust). If is_blurry or is_low_conf and "
         "last_good_prediction exists, it returns held prediction at 0.95x confidence without calling the CNN.",
         "backend/services/webcam_service.py -> process_frame() (lines 590-700)",
         "Input: BGR crop | Output: gamma-adjusted face_enhanced or early held response"),

        ("Step 3.8 — Resize, BGR->RGB & ImageNet Normalization",
         "The BGR crop is converted to RGB, resized to checkpoint-exact pixels, scaled to [0, 1], and normalized by channel mean and std.",
         "For resnet18_raf / resnet18_ravdess, preprocessing metadata stored inside the .pt checkpoint drives exact transforms via load_model(): "
         "cv2.cvtColor(BGR->RGB) -> PIL -> Resize(eval_resize=146, bilinear, antialias=True) -> CenterCrop(image_size=128) -> ToTensor() -> "
         "Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]). For efficientnet_current: cv2.resize(96, 96) -> BGR->RGB -> /255.0 -> "
         "same ImageNet Normalize.",
         "backend/services/webcam_service.py -> _load_resnet18_raf() (298-305), process_frame() (703-717); live_resnet18_demo.py -> load_model()",
         "Input: BGR face_enhanced | Output: Float tensor (1, 3, 128, 128) [or 96x96]"),

        ("Step 3.9 — CNN Forward Pass & Class-Name Mapping BY NAME",
         "The CNN outputs 7 raw logits; ResNet-18 logits are remapped BY CLASS NAME (never hardcoded index) into our canonical 7-emotion order.",
         "With torch.inference_mode(), self.backbone(tensor) returns 7 logits. Critically, best_resnet18.pt uses RAF-DB class order "
         "['Surprise','Fear','Disgust','Happiness','Sadness','Anger','Neutral'], whereas our system expects canonical config order "
         "{0:anger, 1:disgust, 2:fear, 3:joy, 4:neutral, 5:sadness, 6:surprise}. _load_resnet18_raf() resolves each checkpoint class string "
         "through RAF_TO_CANONICAL + aliases ({'happy':3,'joy':3,'sad':5}) into self._raf_index_map ([6,2,1,3,5,0,4] for RAF; identity "
         "[0,1,2,3,4,5,6] for RAVDESS), permuting raw_logits[raf_idx] -> model_logits[canon_idx].",
         "backend/services/webcam_service.py -> _load_resnet18_raf() (lines 307-323), process_frame() (lines 706-727)",
         "Input: Tensor (1, 3, H, W) | Output: model_logits ndarray (7,) in canonical order"),

        ("Step 3.10 — Calibration Offset, Softmax, EMA & Hysteresis",
         "Optional personal neutral subtraction, softmax, adaptive temporal smoothing, and 3-frame stability prevent flickering labels.",
         "(1) If calibrating (25 frames), collects logits and sets baseline_logits = mean_resting - onehot(neutral), subtracting 0.70 * baseline "
         "thereafter. Note hard-coded AU rules were removed here (Phase-2 ablation +0.0996 gain). (2) raw_probs = softmax(adjusted_logits). "
         "(3) Adaptive EMA: alpha = clip(0.20 + 0.35 * raw_conf, 0.18, 0.55); smoothed = alpha * raw_probs + (1 - alpha) * prev. (4) Log-center + "
         "temperature scaling (T=1.0 for ResNet-18, 0.85 for EffNet). (5) Hysteresis: a candidate label only replaces current_label if "
         "top_prob > curr_prob + 0.08 across candidate_count >= 2 steps (3 frames total; prevents 1-frame flicker).",
         "backend/services/webcam_service.py -> process_frame() (lines 735-825)",
         "Input: model_logits (7,) | Output: calibrated_probs (7,), predicted_expression"),

        ("Step 3.11 — Uncertainty Rule & JSON Response",
         "Predictions with weak confidence or a close top-2 tie are flagged as uncertain instead of masquerading as reliable results.",
         "Sorts calibrated_probs to get p_top1, p_top2, and conf_margin = p_top1 - p_top2. Assigns tier: Robust (conf>=0.65 & margin>=0.15), "
         "Moderate (conf>=0.45 & margin>=0.08), else Subtle / Ambiguous. Sets is_uncertain = True if p_top1 < 0.38 or conf_margin < 0.08 "
         "(reason not empirically sweep-logged beyond '# per Phase 4 protocol'; likely heuristic over uniform 1/7=14.3% vs confident margin [VERIFY]). "
         "Returns JSON with bbox normalized [0..1], landmarks normalized [0..1], apparent_expression, calibrated_confidence, probabilities, tier, "
         "is_uncertain, uncertainty_reason, quality_status, disclaimer, and rolling 10-entry history.",
         "backend/services/webcam_service.py -> process_frame() (lines 831-913)",
         "Input: calibrated_probs (7,) | Output: JSON dict sent over WS or HTTP"),

        ("Step 3.12 — Frontend Mirrored Overlay & Left-to-Right Label",
         "JavaScript horizontally flips box and landmark X coordinates onto the unmirrored canvas so graphics line up with the mirrored video while text reads normally.",
         "style.css mirrors #webcam-video via scaleX(-1) ONLY, explicitly commenting NOT to CSS-mirror #webcam-canvas. Therefore in "
         "renderWebcamInference(), box X becomes bx = canvas.width - (bbox.x + bbox.width) * canvas.width and each landmark becomes "
         "px = canvas.width - lx * canvas.width. Rounded bbox corners, 5 sky-blue landmark circles, and ctx.fillText(tagText, bx+9, tagY+18) "
         "draw cleanly—so text like 'JOY 98.4%' is never backwards! Why infer on unmirrored pixels? Because training images (RAF-DB/RAVDESS) "
         "never assume CSS mirror reversal, and live-vs-disk evaluation must match 100%.",
         "frontend/js/app.js -> renderWebcamInference() (lines 1417-1590); frontend/css/style.css:1842-1843",
         "Input: Server JSON + mirrored <video> | Output: Live box, 5 landmarks, text badge, probability bars"),
    ]
    for title, plain, tech, loc, io in steps:
        add_concept_triad(doc, title, plain, f"{tech} [{io}]", loc)


# ---------------------------------------------------------------------------
# SECTION 4: THE MODELS EXPLAINED
# ---------------------------------------------------------------------------
def build_sec4(doc: Document) -> None:
    add_h1(doc, "4. The Models Explained (Architectures, Training & Metrics)", page_break=True)

    add_h2(doc, "4.1 ResNet-18 (Our Default Webcam Backbone)")
    add_concept_triad(
        doc,
        "ResNet-18 Architecture & Layer Freezing",
        "ResNet-18 is an 18-layer convolutional network with skip connections that prevent vanishing gradients; freezing early layers keeps general edge/texture filters intact while adapting high-level facial features.",
        "Standard torchvision.models.resnet18 with a 7-class Linear head has exactly 11,180,103 learnable parameters (plus 9,620 BatchNorm running "
        "buffer entries = 11,189,723 state_dict elements). It consists of stem conv1+bn1 and four residual stages (layer1..layer4), where each "
        "BasicBlock adds its input directly to two 3x3 conv outputs: y = F(x, W) + x. During mixed fine-tuning (--freeze-until layer3), "
        "apply_freeze() disables requires_grad for conv1, bn1, layer1, layer2, and layer3 (2,782,784 frozen parameters = 24.9%), training ONLY "
        "layer4 + fc (8,397,319 trainable parameters = 75.1%), while set_train_eval_modes() keeps frozen BatchNorm layers in .eval() mode so "
        "small-batch RAVDESS noise does not corrupt RAF-DB running means/variances. Why ResNet-18? Light enough for ~31-33 ms CPU inference (~30 FPS) "
        "while preserving rich spatial expression filters vs heavier ResNet-50 or ViT transformers.",
        "train_mixed.py -> apply_freeze() (lines 457-463), set_train_eval_modes() (lines 465-475); exact param counts verified by PyTorch",
    )

    add_h2(doc, "4.2 efficientnet_current (Why Our Old Baseline Failed)")
    add_concept_triad(
        doc,
        "Legacy EfficientNet-B0 Baseline",
        "efficientnet_current is a smaller 96x96-pixel EfficientNet-B0 checkpoint trained before our unified ResNet-18 pipeline that suffered catastrophic domain shift on both RAF-DB and our personal clips.",
        "Measured from models/video_classifier/best_webcam_model.pth, efficientnet_current uses EfficientNet-B0 with a custom head "
        "(Dropout 0.35 -> Linear 1280->256 -> SiLU -> Dropout 0.25 -> Linear 256->7), totaling 4,337,283 parameters (4,379,348 state_dict entries), "
        "at img_size=96 and epoch=34 (calibrated_temperature=0.85 default). train_webcam_phase3.py references a pre-cleanup "
        "models/video_classifier/face_crops_cache directory (no longer on disk; [VERIFY: exact dataset composition originally cached inside "
        "face_crops_cache]). In results/webcam_phase3.json on 699 held-out cache crops it reached 38.63% acc (0.3814 macro-F1), yet its neutral-"
        "bias probe predicted disgust at 64.11%. Under unified WebcamService, it collapsed to 20.4% acc (0.1544 macro-F1) on RAF-DB test "
        "(N=3,068) and 15.61% pooled acc (143/916 frames, 1/9 clips—Neutal.mp4 only) on personal webcam clips, over-predicting neutral/disgust/joy.",
        "backend/services/webcam_service.py:140-160; train_webcam_phase3.py:35-65; results/webcam_phase3.json; results/eval_efficientnet_current.json",
    )

    add_h2(doc, "4.3 Training resnet18_raf on RAF-DB (train_raf_resnet18.py)")
    add_concept_triad(
        doc,
        "Baseline In-the-Wild Face Expression Training",
        "resnet18_raf fine-tuned ImageNet-pretrained ResNet-18 at 128x128 px on clean static RAF-DB face photos using differential learning rates and CosineAnnealingLR.",
        "Dataset: RAF-DB (DATASET/) with 2 exact within-train duplicate images removed -> Train=10,428, Validation=1,841 (15% stratified), "
        "Test=3,068 across 7 classes ['Surprise','Fear','Disgust','Happiness','Sadness','Anger','Neutral']. Augmentations: Resize(128,128), "
        "RandomHorizontalFlip(0.5), RandomRotation(10 deg), ColorJitter(0.1). Eval: Resize(146) -> CenterCrop(128) -> ImageNet Norm. Loss: "
        "CrossEntropyLoss (label_smoothing=0.0, class_weighting=False). Optimizer: AdamW (weight_decay=1e-4) with backbone_lr=1e-4 and "
        "head_lr=1e-3, CosineAnnealingLR(T_max=10). Early stopping (patience=3 on val macro-F1): saved best at Epoch 6 (val loss 0.7405, val acc "
        "78.87%, val macro-F1 0.6915), stopped after Epoch 9. Held-out RAF-DB Test: Accuracy 79.53%, Macro-F1 0.6721, Balanced Acc 64.88%.",
        "train_raf_resnet18.py -> Config (48-113), make_transforms() (425-448), fit() (726-850); external_models/resnet18_raf/training.log",
    )

    add_h2(doc, "4.4 Mixed Fine-Tuning: resnet18_ravdess (prepare_ravdess.py & train_mixed.py)")
    add_img(doc, "report/figures/fig_diag_ravdess_prep.png",
            "Figure 4.1: RAVDESS video extraction pipeline (modality 01, speech 01, calm dropped, middle 30-70% @ 5 fps, quality gate).", width_in=5.7)
    add_img(doc, "report/figures/fig_diag_training_pipeline.png",
            "Figure 4.2: Mixed RAF-DB + RAVDESS balanced fine-tuning (extra_share=0.2, freeze-until layer3, best Epoch 7).", width_in=5.7)
    add_concept_triad(
        doc,
        "RAVDESS Data Prep & Mixed-Domain Fine-Tuning Pipeline",
        "To teach the static-photo ResNet-18 what real talking webcam faces look like without forgetting RAF-DB, we extracted 3,970 aligned speech frames from RAVDESS actors and mixed 6 RAVDESS frames with 26 RAF-DB frames in every batch of 32.",
        "1) prepare_ravdess.py: Reads 7-part filename mm-cc-ee-ii-ss-rr-aa.mp4 inside Video speech dataset/. Keeps ONLY full-AV (mm=='01') speech "
        "(cc=='01') clips, drops emotion calm (ee=='02', 80 clips skipped), skips 600 non-AV/song files and 2 clips with no valid faces -> 518 "
        "clips kept. Assigns Actors 01-08 -> train and Actors 09-10 -> val (speaker-disjoint!). Extracts middle 30%-70% frame window at 5 fps "
        "(compute_sample_indices), runs YuNet + aligned_square_crop + blur gate (Laplacian var >= 30.0, conf >= 0.55, w/h >= 64 -> 3,970 total "
        "crops: 3,184 train / 786 val).\n"
        "2) train_mixed.py: Initializes from best_resnet18.pt, freezes through layer3, mixes batch_size=32 using extra_share=0.2 (6 RAVDESS + 26 "
        "RAF-DB) via separate inverse-frequency WeightedRandomSamplers (_build_mixed_loaders). Shared MixedAugmenter simulates webcam optics: "
        "RandomResizedCrop(scale 0.75-1.0), HFlip, Rot 12 deg, brightness/contrast/gamma jitter, warm/cool shift, downscale-upscale softness, "
        "Gaussian blur, JPEG quality 35-85 re-encoding, sensor noise, plus 30% synthetic dark eyeglass rims + lens glare (_apply_glasses_occlusion). "
        "Trains with AdamW (backbone_lr=2e-5 for layer4, head_lr=2e-4 for fc, wd=1e-4) and CrossEntropyLoss(label_smoothing=0.1). Early stopping "
        "tracks avg_val = 0.5*(RAF_val_F1 + RAVDESS_val_F1), peaking at Epoch 7 (RAF val 0.8749, RAVDESS val 0.5354, composite 0.7051).",
        "prepare_ravdess.py -> parse_ravdess_filename(), compute_sample_indices(), _passes_quality_gate(); train_mixed.py -> MixedAugmenter, _build_mixed_loaders(), train_and_evaluate(); results/prepare.log & train_ravdess.log",
    )

    add_h2(doc, "4.5 Evaluation Metrics Explained with Worked Examples")
    add_p(doc,
          "Suppose 10 test samples have 8 Neutral and 2 Fear. A lazy dummy classifier predicts Neutral for ALL 10 samples: "
          "Accuracy = 8/10 = 80% (looks impressive!), yet Precision(Fear)=0/0=0, Recall(Fear)=0/2=0, F1(Fear)=0. Meanwhile for Neutral: "
          "TP=8, FP=2, FN=0 -> Precision = 8/10 = 0.80, Recall = 8/8 = 1.00, F1 = 2*0.80*1.00 / (0.80+1.00) = 0.889. Unweighted Macro-F1 = "
          "(0.889 + 0.0) / 2 = 0.444! That immediately exposes neutral-class collapse on MELD (where 47.1% of samples are neutral) AND on sparse "
          "emotion classes like fear/disgust. A 7x7 Confusion Matrix M[i,j] counts true class i predicted as class j; row-normalizing M[i,j] / row_sum[i] "
          "turns counts into per-class recall proportions so diagonal cells show accuracy for each individual emotion.")

    add_h2(doc, "4.6 MELD Multimodal Pipeline (Audio, Text, Video, Fusion, Ablations & Calibration)")
    add_img(doc, "report/figures/fig_diag_meld_fusion.png",
            "Figure 4.3: MELD 16-D stream projection, masked Transformer self-attention, entropy gates, consensus residual, and calibration.", width_in=5.7)
    meld_parts = [
        ("Audio Pipeline (Wav2Vec 2.0 + StrongerAudioClassifier)",
         "Resamples audio to 16 kHz mono, trims leading/trailing silence (top_db=30), runs frozen facebook/wav2vec2-base mask-mean pooling to 768-D, and classifies via StrongerAudioClassifier.",
         "StrongerAudioClassifier (768->512 ReLU+BN+Drop0.3 -> 256 ReLU+Drop0.3 -> 7; 527,879 params) trained with inverse-frequency WeightedRandomSampler. "
         "Why? Unweighted Audio Baseline hit 49.23% accuracy by over-predicting neutral (0.1771 macro-F1). Balanced sampler lifted MELD Test (N=2,610) macro-F1 "
         "to 0.2209 (32.87% acc, weighted-F1 0.3580); weighted-CE got 0.2110, Focal Loss 0.1972.",
         "backend/services/audio_service.py -> predict_from_audio(); backend/models/audio_model.py -> StrongerAudioClassifier"),

        ("Text Pipeline (all-MiniLM-L6-v2 vs TF-IDF)",
         "Extracts 384-D contextual embeddings from normalized dialogue using frozen all-MiniLM-L6-v2 and runs a scikit-learn calibrated classifier.",
         "TextPreprocessingUtils expands contractions (e.g., \"He's\" -> \"He is\"). Contextual pipeline tokenizes max_length=128 through "
         "sentence-transformers/all-MiniLM-L6-v2 mean-pooling + CalibratedClassifierCV (models/text_classifier/best_text_model_contextual.pt). On MELD Test "
         "(N=2,610), this raised accuracy from 40.15% (0.2989 macro-F1, TF-IDF 1-3 n-grams 12k) to 42.87% (0.3326 macro-F1, 0.4725 weighted-F1).",
         "backend/services/text_service.py -> predict_from_text()"),

        ("Video Pipeline (ResNet-50 8-Frame Temporal Pooling)",
         "Samples 8 evenly spaced frames from uploaded MP4s, center/upper crops the head, applies CLAHE lighting equalization, and pools 2048-D ResNet-50 embeddings.",
         "ResNet-50 (fc=Identity) outputs eight 2048-D frame vectors averaged across time -> VideoClassifier MLP (2048->256->7, 526,343 params). "
         "Offline checkpoint log on 35 MELD clips scored 37.14% acc (0.3684 macro-F1; CM not stored), while on the 105-sample 16-D fusion test tensor "
         "unimodal video drops to 8.57% raw / 6.67% through the fusion head due to messy Friends camera cuts and background faces.",
         "backend/services/video_service.py -> predict_from_video(); backend/models/video_model.py -> VideoClassifier"),

        ("Fusion Strategy: Hybrid Feature + Consensus Logit Late Fusion",
         "Our fusion works at decision/probability-logit level (hybrid late fusion), combining 16-D summary vectors from each modality using a 2-layer Transformer with missing-modality masks.",
         "Notice how fuse_predictions() builds a 16-D vector per modality: [7 probs, 7 centered log-probs, 1 normalized entropy certainty, 1 quality_discount]. "
         "AdvancedCrossModalAttentionFusionNet (167,113 params) projects each 16-D vector to 96-D + learned modality embeddings, runs 2-layer 4-head "
         "TransformerEncoder with key_padding_mask=~mask (so absent streams never attend), computes sigmoid reliability gates from [attended; cert; qual], "
         "blends neural_logits and consensus_logits with learned clamp(consensus_weight, 0.1, 0.9), and outputs final probabilities. Offline L-BFGS calibration "
         "yielded T*=0.5953 (reducing ECE 0.2130 -> 0.0734, Brier 0.8811 -> 0.7785, NLL 1.9835 -> 1.7244). At live runtime, MultimodalFusionService adds "
         "concordance boosting (1.15-1.30x) + adaptive sharpening T_adaptive=0.24-0.28 [VERIFY: exact choice rationale for runtime 0.24-0.28 vs offline T*=0.5953].",
         "backend/models/fusion_model.py -> AdvancedCrossModalAttentionFusionNet; backend/services/fusion_service.py -> _build_16d_feature(), fuse_predictions()"),
    ]
    for title, pl, te, cl in meld_parts:
        add_concept_triad(doc, title, pl, te, cl)


# ---------------------------------------------------------------------------
# SECTION 5: DATASETS
# ---------------------------------------------------------------------------
def build_sec5(doc: Document) -> None:
    add_h1(doc, "5. Datasets: Composition, Splits, Ethics & Non-Leakage", page_break=True)
    add_p(doc,
          "A critical viva topic is data separation: examiners want proof that test sets and personal webcam clips were NEVER leaked into training.")
    add_csv(doc, "Table 5.1: Complete Datasets Inventory (From report/tables/datasets_summary.csv)",
            "report/tables/datasets_summary.csv", [1.05, 1.25, 1.15, 1.35, 0.65, 1.00], font_pt=8.2)
    add_img(doc, "report/figures/fig_class_dist_datasets.png",
            "Figure 5.1: Sample distribution across classes and splits for RAF-DB, RAVDESS crops, and MELD.", width_in=5.8)

    add_bullet(doc, "RAF-DB (DATASET/): ",
               "15,337 unconstrained web facial photos used (after removing 2 duplicate RGB rows within train): 10,428 train / 1,841 val / 3,068 official test. "
               "Academic non-commercial license. Used for training & validating resnet18_raf and as the 80% anchor batch in train_mixed.py.")
    add_bullet(doc, "RAVDESS Speech Subset (Video speech dataset/ -> ravdess_crops/): ",
               "Published RAVDESS has 24 actors (CC BY-NC-SA 4.0); our local folder contains 1,200 MP4s from Actors 01-10 [VERIFY: why 10-actor download subset only]. "
               "prepare_ravdess.py kept 518 speech-only clips (3,970 aligned crops), strictly splitting Actors 01-08 (3,184 train) from Actors 09-10 (786 val) so zero faces overlap.")
    add_bullet(doc, "MELD (MELD.Raw/): ",
               "13,708 TV dialogue utterances from Friends (Train 9,989 / Dev 1,109 / Test 2,610; 47.1% neutral skew). Unimodal audio & text test scores run on "
               "all 2,610 test items; multimodal fusion ablations evaluate on the balanced 105-sample subset (15 per class) in data/fusion_test_dataset.pt.")
    add_bullet(doc, "Personal Evaluation Clips (New folder/): ",
               "9 private webcam MP4 clips (916 frames total, plus reserved ' (3)' clips). NEVER used in training, augmentation, or threshold tuning of weights—strictly "
               "evaluation-only—because training on those 9 clips would memorize one person's background/face and falsely inflate results.")


# ---------------------------------------------------------------------------
# SECTION 6: RESULTS AND HOW TO READ THEM
# ---------------------------------------------------------------------------
def build_sec6(doc: Document) -> None:
    add_h1(doc, "6. Results and How to Read Confusion Matrices", page_break=True)

    add_csv(doc, "Table 6.1: Webcam Backbone Summary (From report/tables/backbone_comparison.csv)",
            "report/tables/backbone_comparison.csv",
            [0.95, 0.70, 0.35, 0.52, 0.52, 0.52, 0.52, 0.48, 0.55, 0.67, 0.67], font_pt=7.2,
            col_indices=[0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            rename_headers=["Backbone", "Arch", "Px", "RAF Acc", "RAF F1", "RAV Acc", "RAV F1", "Clips", "Pooled", "p50 ms", "p95 ms"])

    add_csv(doc, "Table 6.2: Per-Clip Breakdown Across All 9 Evaluation Clips (916/916 Face Frames Detected)",
            "report/tables/per_clip_results.csv",
            [0.85, 0.50, 0.38, 0.55, 0.50, 0.55, 0.50, 0.62, 0.50, 0.75, 0.75], font_pt=7.2,
            col_indices=[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11],
            rename_headers=["Clip", "Truth", "Frm", "EffMaj", "EffAcc", "RAFMaj", "RAFAcc", "RAVMaj", "RAVAcc", "RAVWrong", "Share"])

    add_img(doc, "report/figures/fig_cm_webcam_resnet18_ravdess.png",
            "Figure 6.1: Webcam 9-clip (916 frames) confusion matrix for default resnet18_ravdess (57.31% pooled, 5/9 clips).", width_in=4.8)
    add_img(doc, "report/figures/fig_webcam_backbone_comparison.png",
            "Figure 6.2: Per-clip and pooled progression across EfficientNet-B0 (15.6%), ResNet-18 RAF (44.0%), and ResNet-18 RAF+RAVDESS (57.3%).", width_in=5.6)

    add_h2(doc, "6.1 Plain-English Reading of Key Confusion Matrix Patterns")
    add_bullet(doc, "Why Anger reads as Surprise (61.39%) & Neutral (38.61%): ",
               "In Anger.mp4 (101 frames), raised/furrowed eyebrows and tense eyes look visually close to surprised widening in static web-scraped photos without "
               "vocal shout intensity or multi-second onset.")
    add_bullet(doc, "Why Sadness reads 100.0% as Neutral: ",
               "In sadness.mp4 (93 frames), subtle downward gaze and slack lips lack dramatic crying distortion, falling squarely inside RAVDESS/RAF neutral bounds.")
    add_bullet(doc, "Why Joy (100.0%) and Disgust (100.0% / 79.78%) succeed: ",
               "Teeth/cheek zygomaticus elevation for joy and nose-wrinkle / upper-lip curl for disgust create strong geometric edge contrast preserved after affine alignment.")
    add_bullet(doc, "Why Fear diverges (43.75% Fear (2).mp4 vs 93.98% disgust on Fear.mp4): ",
               "Fear and disgust share stretched lips and flared nose-bridge wrinkles, causing small lighting/head differences to tip between them.")
    add_bullet(doc, "Why RAF-DB F1 rose (0.6721 -> 0.7077) while RAVDESS val stayed 0.5354 (train-val gap 0.2205): ",
               "Mixed batch normalization + label smoothing regularized minority classes on RAF-DB (and kept RAF train-val gap at -0.0257), while 8 training actors "
               "in RAVDESS aren't diverse enough to fully generalize to unseen Actors 09-10 (+0.2205 gap). Nine webcam clips (1/9 -> 4/9 -> 5/9) are statistically "
               "thin (just 9 independent clips, though 916 correlated frames), so every clip swing shifts accuracy by 11.1%.")

    add_h2(doc, "6.2 Unified CPU Latency & MELD Results")
    add_csv(doc, "Table 6.3: Unified 200-Frame CPU Latency (report/tables/latency_benchmark.csv — Earlier Single Runs Non-Comparable)",
            "report/tables/latency_benchmark.csv",
            [1.05, 0.80, 0.35, 0.50, 0.45, 0.50, 0.50, 0.50, 0.50, 0.50, 0.80], font_pt=7.5,
            col_indices=[0, 1, 2, 7, 8, 9, 10, 11, 12, 13, 14],
            rename_headers=["Backbone", "Arch", "Px", "Mean", "Std", "p50 ms", "p95 ms", "p99 ms", "Min", "Max", "FPS(p50)"])
    add_img(doc, "report/figures/fig_latency_comparison.png",
            "Figure 6.3: Unified 200-frame CPU latency on Disgust (2).mp4 showing ~31.5-33.6 ms median across all backbones.", width_in=5.2)

    add_csv(doc, "Table 6.4: MELD Unimodal & 7-Configuration Multimodal Fusion Results (report/tables/meld_model_results.csv)",
            "report/tables/meld_model_results.csv",
            [1.35, 1.35, 0.95, 0.42, 0.58, 0.58, 0.62, 0.60], font_pt=7.4,
            col_indices=[0, 2, 3, 4, 5, 6, 7, 8],
            rename_headers=["Model / Config", "Architecture", "Eval Set", "N", "Acc", "MacroF1", "W-F1", "ECE"])
    add_img(doc, "report/figures/fig_ablation_meld_7config.png",
            "Figure 6.4: 7-configuration MELD ablation benchmark (Text-only & A+T reach 37.14%, Trimodal achieves best ECE 0.0734).", width_in=5.5)
    add_img(doc, "report/figures/fig_calibration_reliability.png",
            "Figure 6.5: Reliability curve before vs after temperature calibration (ECE reduced from 0.2130 to 0.0734 at T*=0.5953).", width_in=5.0)


# ---------------------------------------------------------------------------
# SECTION 7: DESIGN DECISIONS & "WHY" ANSWERS
# ---------------------------------------------------------------------------
def build_sec7(doc: Document) -> None:
    add_h1(doc, "7. Design Decisions and Verified \"Why\" Answers", page_break=True)
    add_p(doc,
          "Examiners probe why each engineering choice was made. Below are all ten core design choices backed strictly by logs/code—or flagged "
          "as [VERIFY] when an empirical rationale was not written down.")

    decisions = [
        ("1. Why unify live endpoints and offline scripts onto WebcamService?",
         "Because earlier /frame, /ws, and evaluation scripts ran slightly different crops and heuristics, causing fake regressions and debugging confusion. "
         "Evidence: results/webcam_fix_log.md Session 1 documents unifying /frame, /frame-json, /ws, /ws/webcam, /multimodal, and benchmark_webcam.py onto "
         "WebcamService.process_frame(), achieving 100% (916/916) label parity on 320x180 JPEG85 inputs."),
        ("2. Why use UNMIRRORED inference while mirroring only the user preview?",
         "Training photos (RAF-DB and RAVDESS) were not recorded inside CSS mirrors, so flipping inference inputs alters asymmetric mouth/brow gradients and "
         "breaks parity against disk evaluation. Evidence: frontend/css/style.css:1831 mirrors #webcam-video via scaleX(-1) only, comment at lines 1842-1843 "
         "explicitly leaves #webcam-canvas unmirrored, and app.js:1367 captures raw unmirrored pixels onto snapCanvas."),
        ("3. Why remove the hard-coded facial Action Unit (AU) geometry rules?",
         "Hand-written landmark threshold multipliers overrode learned CNN posteriors and hurt accuracy on real variation. "
         "Evidence: results/webcam_fix_log.md + results/phase2_ablations.json prove removing hard-coded AU rules raised held-out MELD validation macro-F1 "
         "from 0.2872 to 0.3868 (+0.0996 gain); webcam_service.py:760-763 documents zeroing those deltas."),
        ("4. Why use inverse-frequency class-balanced sampling (WeightedRandomSampler)?",
         "Both MELD (47.1% neutral vs <2.8% disgust/fear) and RAF-DB skew heavily toward happy/neutral faces; without balanced sampling, models optimize raw "
         "accuracy by ignoring rare emotions. Evidence: train_mixed.py:424-431 builds inverse-frequency samplers, and train.py --strategy sampler raised MELD "
         "audio macro-F1 from 0.1771 to 0.2209 (report/tables/meld_model_results.csv)."),
        ("5. Why freeze early ResNet-18 stages (--freeze-until layer3) during mixed fine-tuning?",
         "With only 3,184 training crops from 8 RAVDESS actors, unfreezing all 11.18M parameters risks catastrophic forgetting of general RAF-DB facial filters "
         "and overfitting speaker identities. Freezing conv1..layer3 (24.9% params) and keeping their BatchNorm statistics fixed while adapting layer4 + fc "
         "(75.1%, 8,397,319 params) improved RAF-DB macro-F1 from 0.6721 to 0.7077 (train_mixed.py:457-474, results/train_ravdess.log). Note: the specific choice "
         "of extra_share=0.2 and lr=2e-5/2e-4 has no sweep log on disk [VERIFY: confirm extra_share=0.2 sensitivity]."),
        ("6. Why extract RAVDESS speech-only clips instead of song clips?",
         "Conversational webcam users speak rather than sing sustained pitch notes, and emotional singing exaggerates mouth opening far differently from facial cues "
         "during speech. Evidence: prepare_ravdess.py:67 filters channel=='01' (speech) and modality=='01' (full-AV), skipping 600 song/audio-only/video-only files."),
        ("7. Why are only Actors 01-10 in RAVDESS used instead of all 24 actors?",
         "Standard published RAVDESS contains 24 actors, whereas our workspace folder Video speech dataset/ contains Actors 01-10 (1,200 files). "
         "Reason not documented in README/logs; likely reason: partial local download constrained by disk space or bandwidth [VERIFY: confirm why Actors 11-24 were omitted]."),
        ("8. Why flag predictions as \"uncertain\" (is_uncertain) instead of always outputting a confident label?",
         "Silent uncertainty misleads users when expressions are resting or subtle; flagging low confidence (p_top1 < 0.38) or close top-2 ties (margin < 0.08) "
         "honors ethical transparency. Evidence: webcam_service.py:831-852 implements this check citing 'Phase 4 protocol'; however, the exact numeric cutoffs 0.38 and 0.08 "
         "lack an ablation table on disk; likely reason: heuristic spacing above uniform 1/7 = 14.3% [VERIFY: verify 0.38/0.08 threshold choice]."),
        ("9. Why process everything strictly locally on CPU with volatile RAM live buffers?",
         "Facial video and voice audio are sensitive biometric data; keeping frames in RAM with zero disk logging on live /ws or /frame routes eliminates cloud transmission "
         "latency, API fees, and third-party server privacy risk (debug saves occur only upon explicit opt-in via POST /predict/webcam/debug-capture)."),
        ("10. Why build the serving layer with FastAPI instead of Flask?",
         "Reason not explicitly benchmarked in README/logs; likely reason: FastAPI provides native ASGI full-duplex WebSockets (/predict/webcam/ws alongside REST), "
         "Pydantic request validation (BackboneSwitchPayload, SessionResetPayload), and async OpenAPI docs out of the box [VERIFY: FastAPI vs Flask choice]."),
    ]
    for q, ans in decisions:
        add_h3(doc, q)
        add_p(doc, ans, size=11.2, sa=4.0)


# ---------------------------------------------------------------------------
# SECTION 8: HONEST LIMITATIONS & WHAT NOT TO CLAIM
# ---------------------------------------------------------------------------
def build_sec8(doc: Document) -> None:
    add_h1(doc, "8. Honest Limitations and What NOT to Claim", page_break=True)
    add_p(doc,
          "Viva panels respect technical maturity far more when you state weaknesses accurately rather than overselling metrics.")

    claim_rows = [
        ("9 Eval Clips Sample Scope", "\"Our 9 personal webcam clips served as both our backbone selection set and our evaluation benchmark, and all 9 depict one speaker acting expressions (916 correlated frames).\"",
         "\"Works for any user in the wild at 57.3% accuracy\" — DO NOT claim multi-subject generalization."),
        ("Weak Classes on Webcam", "\"Anger (0%—often read 61.4% as surprise), sadness (0%—100% neutral), surprise (39.3%), and Fear.mp4 (0%, read as disgust) remain serious failure modes.\"",
         "\"Accurately classifies all 7 emotions on webcam\" — highlight that Joy & Disgust improved dramatically while Anger/Sadness need future temporal work."),
        ("Studio-to-Webcam Domain Shift", "\"Even after mixing RAVDESS Actors 01-08, Actors 09-10 validation macro-F1 was 0.5354 (train-val gap 0.2205), showing cross-actor domain shift persists.\"",
         "\"Solves domain shift completely\" — present it as partial mitigation (+41.7 percentage points over baseline EffNet-B0)."),
        ("Live UI Screenshot Provenance", "\"Live webcam tab screenshot ui_live_webcam.png was captured in STANDBY mode in headless Chrome (no physical camera), and quantitative tests ran via recorded New folder/*.mp4 clips.\"",
         "\"Benchmarked on a multi-hour live multi-person user trial\" — state recorded MP4 parity test explicitly."),
        ("MELD Trimodal vs Text-Only", "\"On N=105 balanced MELD test, Text-only and Audio+Text hit 37.14% accuracy while trimodal hit 36.19%; fusion's real win is halving ECE from 0.2130 to 0.0734.\"",
         "\"Multimodal fusion always beats every single modality in accuracy\" — explain video noise dilution vs calibration gain."),
        ("Features NOT Implemented Yet", "\"Personal-face fine-tuning, HSEmotion runtime deployment in the switcher, flip test-time augmentation, and multi-frame LSTM/Transformer smoothing over sequences were NOT built.\"",
         "Never pretend 25-frame additive neutral offset calibration is gradient-based personal fine-tuning or temporal sequence modeling."),
    ]
    add_table(doc, "Table 8.1: Viva Honesty Checklist (What to Say Aloud vs. What NEVER to Claim)",
              ["Topic", "DO Say (100% Honest & Defensible)", "NEVER Claim This"], claim_rows, [1.25, 2.75, 2.45], font_pt=8.4)


# ---------------------------------------------------------------------------
# SECTION 9: VIVA QUESTION BANK (100 QUESTIONS ACROSS GROUPS a..i)
# ---------------------------------------------------------------------------
def build_sec9(doc: Document) -> None:
    add_h1(doc, "9. Viva Question Bank (100 Questions Grouped a-i)", page_break=True)
    add_p(doc,
          "Each question includes Difficulty (Easy/Medium/Hard), Likelihood (Very likely/Likely/Possible), a crisp 2-4 sentence Answer ready to "
          "speak aloud, optional deeper technical intuition, and where to point in the repository or guide.")

    groups = [
        ("Group (a): Fundamentals & ML Basics (Q1-Q11)", [
            ("What is automated emotion recognition and why is it challenging?",
             "It classifies human facial, vocal, or textual cues into discrete affect categories (here 7 classes: anger, disgust, fear, joy, neutral, sadness, surprise). It is challenging because expressions vary hugely across people, lighting, poses, and cultural habits, and visible muscles do not always match internal feelings.",
             "Our UI explicitly labels outputs as 'apparent facial expressions' rather than ground-truth internal psychology.", "Sec 1 / webcam_service.py:910", "Easy", "Very likely"),
            ("What is deep learning versus classical machine learning?",
             "Classical ML (like our TF-IDF Logistic Regression text baseline) needs hand-engineered features first, whereas deep learning (CNNs like ResNet-18 and transformers like Wav2Vec 2.0) learns hierarchical representations directly from raw pixels or waveforms.",
             "On MELD text, learned MiniLM neural embeddings outperformed 12,000 hand-tuned TF-IDF features (42.87% vs 40.15%).", "Sec 4.6 / text_service.py", "Easy", "Likely"),
            ("How does a Convolutional Neural Network (CNN) extract facial features?",
             "A CNN slides learnable 2D filter kernels across the image to detect edges in early layers, eyes/mouth shapes in middle layers, and expression patterns in deep layers. Weight sharing across spatial locations makes it far more parameter-efficient than flattening 128x128x3 images into an MLP.",
             "Downsampling via strided convolutions builds spatial translation robustness while cutting compute cost.", "Sec 4.1 / ResNet-18", "Easy", "Very likely"),
            ("What is transfer learning and how did you apply it?",
             "Transfer learning reuses weights pretrained on massive external corpora (like ImageNet for ResNet-18/50 or speech audio for Wav2Vec 2.0) instead of starting from random initialization. It drastically reduces required labeled training data and avoids overfitting on small datasets like RAVDESS.",
             "We actually chained transfer twice: ImageNet -> RAF-DB (resnet18_raf) -> mixed RAF-DB+RAVDESS (resnet18_ravdess).", "Sec 4.3-4.4 / train_mixed.py", "Easy", "Very likely"),
            ("What is the difference between feature extraction and fine-tuning?",
             "In pure feature extraction we freeze the entire backbone and train only a shallow head (as we do with frozen ResNet-50 + VideoClassifier 2048->7 and frozen Wav2Vec2 + StrongerMLP). In fine-tuning we unfreeze some or all convolutional blocks (like layer4+fc in train_mixed.py) with low learning rates.",
             "ResNet-18 RAF trained 100% of weights (11.18M), whereas RAVDESS fine-tuning unfroze 75.1% (8.40M) after layer3.", "Sec 4.1 / train_mixed.py:457", "Medium", "Very likely"),
            ("What is overfitting and where did you observe it in this project?",
             "Overfitting happens when a model memorizes training nuances or speaker identities instead of generalizing to unseen subjects. We observed it directly in RAVDESS mixed training, where the train-minus-validation macro-F1 gap reached 0.2205 across only 8 training actors.",
             "Conversely, RAF-DB held a train-val gap of -0.0257 thanks to larger 10,428-image diversity and augmentation.", "Sec 6.1 / webcam_fix_log.md:6", "Easy", "Very likely"),
            ("Why do we maintain separate Training, Validation, and Test splits?",
             "Training optimizes weights, Validation selects epochs/hyperparameters (like Epoch 6 on RAF or Epoch 7 on RAVDESS), and Test provides an untouched final estimate. Reusing test labels for tuning leaks information and creates overly optimistic accuracy.",
             "Even with strict test isolation on RAF-DB (N=3,068), our 9 personal clips were reused for selection+reporting—which we disclose in Sec 8.", "Sec 5 / train_raf_resnet18.py", "Easy", "Very likely"),
            ("What is data leakage and how did you prevent it?",
             "Data leakage occurs if duplicate or correlated samples cross between train and validation/test splits. We ran MD5/pixel deduplication on RAF-DB (removing 2 within-train duplicates) and split RAVDESS strictly by actor ID (Actors 01-08 train vs 09-10 val) so the same face never appears in both.",
             "Frame-level random splitting of RAVDESS videos would falsely inflate validation accuracy toward ~95%+ via identity memorization.", "Sec 4.3-4.4 / prepare_ravdess.py", "Medium", "Very likely"),
            ("Why did removing 2 duplicates in RAF-DB matter?",
             "train_raf_resnet18.py audits exact pixel hashes across splits and fails if cross-split duplicates exist. Removing 2 duplicate images inside the training folder ensured class counts (10,428 train / 1,841 val) did not double-weight identical examples.",
             "", "Sec 4.3 / training.log:3-5", "Medium", "Possible"),
            ("What is a random seed and which seeds did your runs use?",
             "A fixed random seed initializes deterministic weight splits, batch shuffles, and augmentations so experiments are reproducible. Our ResNet-18 RAF, RAVDESS mixed training, and MELD Fusion used seed=42, while legacy Phase-3 EfficientNet used seed=101.",
             "", "report/tables/hyperparameters.csv", "Easy", "Possible"),
            ("Can emotion recognition work purely without GPU hardware?",
             "Yes—inference using lightweight ResNet-18 at 128x128 pixels plus YuNet runs at ~31.5-33.6 ms p50 (~30 FPS) purely on our Intel Core i7-12650H CPU. However, training from raw multi-hour video datasets on CPU is slow (~3 min/epoch for RAF-DB).",
             "", "Sec 6.2 / training.log", "Easy", "Likely"),
        ]),

        ("Group (b): Computer Vision & Preprocessing (Q12-Q22)", [
            ("How does YuNet face detection differ from older Haar Cascades?",
             "YuNet (face_detection_yunet_2023mar.onnx) is a tiny anchor-free deep detector (~75.8K params) built into OpenCV DNN. Unlike fragile Haar Cascades, it outputs both bounding boxes and 5 exact landmarks (two eyes, nose tip, two mouth corners) reliably under tilt and lighting changes.",
             "", "Sec 3 Step 3.5 / webcam_service.py:276", "Easy", "Very likely"),
            ("Why extract 5 landmarks instead of just using the bounding box?",
             "A bounding box alone can be rotated when a user tilts their head sideways. Eye coordinates allow aligned_square_crop() to level the face horizontally around the eye midpoint so the CNN sees canonical upright geometry.",
             "", "Sec 3 Step 3.6 / live_resnet18_demo.py:49", "Medium", "Very likely"),
            ("What happens in aligned_square_crop() if YuNet detects an extreme >45 degree head tilt?",
             "As a safety guard inside live_resnet18_demo.py, if abs(angle) > 45 degrees it skips rotation (sets angle = 0.0) instead of guessing, preventing upside-down crops from occasional landmark glitches. Off-angle frames are also flagged in quality_status.",
             "", "Sec 3 Step 3.6 / live_resnet18_demo.py:53", "Medium", "Likely"),
            ("What padding mode avoids black borders around rotated face crops?",
             "OpenCV warpAffine() and copyMakeBorder() use cv2.BORDER_REPLICATE to extend edge skin/background pixels rather than sharp black zeros, preventing artificial high-contrast borders that trick CNN edge filters.",
             "", "live_resnet18_demo.py:59,74", "Medium", "Possible"),
            ("Why do we use pad=1.10 for ResNet-18 backbones vs pad=1.25 for EfficientNet?",
             "ResNet-18 was trained on tightly framed RAF-DB face crops (and prepared via RAVDESS with tight framing), whereas wider pad=1.25 includes slightly more forehead/chin context used by the legacy path.",
             "", "webcam_service.py:585", "Medium", "Possible"),
            ("How do you convert between OpenCV BGR and PyTorch RGB?",
             "OpenCV imdecode() loads images as Blue-Green-Red (BGR), while TorchVision ResNet-18 and PIL expect Red-Green-Blue (RGB). Feeding BGR directly swaps red lips/skin into blue and ruins predictions, so process_frame() explicitly runs cv2.cvtColor(..., cv2.COLOR_BGR2RGB).",
             "", "Sec 3 Step 3.8 / webcam_service.py:704", "Easy", "Very likely"),
            ("Why normalize pixels using ImageNet mean and std instead of just dividing by 255?",
             "Backbones pretrained on ImageNet expect zero-centered channel distributions normalized by mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]. Matching checkpoint normalization exactly avoids distribution shift before the first convolutional layer.",
             "", "Sec 3 Step 3.8 / Formula F7", "Easy", "Likely"),
            ("Why Resize(146) followed by CenterCrop(128) at evaluation?",
             "Resizing the shortest side to 146 px (~128 / 0.877) and taking a centered 128x128 crop keeps training and validation framing identical to standard TorchVision protocols while trimming outer crop margins. Note: exact sweep rationale vs 224 px isn't documented [VERIFY].",
             "", "train_raf_resnet18.py:429-434", "Medium", "Likely"),
            ("Why does WebcamService use mild gamma LUT instead of CLAHE lighting enhancement?",
             "Comments in webcam_service.py:619-620 note that aggressive local CLAHE amplifies shadow creases along nasolabial folds and induces false Disgust predictions; instead we apply gentle global gamma (1.35 if mean luminance < 60, 0.75 if > 195). Notice MELD offline ResNet-50 video still uses CLAHE.",
             "", "webcam_service.py:619-636 vs video_service.py:95", "Hard", "Likely"),
            ("How does the Laplacian variance blur gate work?",
             "Applying a 2D second-derivative Laplacian filter highlights sharp edges (eye rims, teeth, brows); blurry motion frames have near-zero edge intensity variance. When var < 30.0 in live mode (or < 30.0 in prepare_ravdess.py), blurry frames are rejected or hold last_good_prediction.",
             "", "Sec 3 Step 3.7 / Formula F14", "Medium", "Very likely"),
            ("What realistic webcam augmentations exist inside MixedAugmenter?",
             "Besides RandomResizedCrop, HFlip, and +/-12 deg rotation, MixedAugmenter applies brightness/contrast/gamma shifts, warm-cool color balance, downscale-upscale webcam softness, Gaussian blur, JPEG quality 35-85 recompression, Gaussian sensor noise, and 30% synthetic dark eyeglass rims + glare.",
             "", "Sec 4.4 / train_mixed.py:273-370", "Medium", "Likely"),
        ]),

        ("Group (c): Model Architecture & Training Mechanics (Q23-Q34)", [
            ("How do residual skip connections (F(x) + x) help ResNet-18 train smoothly?",
             "During backpropagation, gradients flow directly through the identity shortcut (+ x) without shrinking exponentially through repeated weight multiplications. This lets deeper residual stages learn clean incremental refinements instead of degrading.",
             "", "Sec 4.1 / Formula F8", "Easy", "Very likely"),
            ("How many parameters are in ResNet-18 vs EfficientNet-B0?",
             "Using exact PyTorch parameter counts on our checkpoints, 7-class ResNet-18 has 11,180,103 parameters (8,397,319 trainable in layer4+fc when frozen through layer3), while efficientnet_current has 4,337,283 parameters.",
             "", "Sec 4.1-4.2 (PyTorch computed)", "Medium", "Likely"),
            ("How does Batch Normalization behave differently during frozen fine-tuning?",
             "Normally BatchNorm updates running mean and variance during training. In train_mixed.py, set_train_eval_modes() pins frozen conv1..layer3 BatchNorm modules in .eval() mode so noisy 6-sample RAVDESS batches cannot corrupt RAF-DB normalization statistics.",
             "", "Sec 4.1 / train_mixed.py:465-475", "Hard", "Likely"),
            ("What does Softmax do mathematically?",
             "Softmax exponentiates each raw logit z_c and divides by the sum over all 7 classes, turning unconstrained positive/negative logits into probabilities strictly between 0 and 1 that sum to 1.0.",
             "Logits can be scaled by temperature T before softmax to soften or sharpen confidence.", "Formula F1 / webcam_service.py:765", "Easy", "Very likely"),
            ("What is Cross-Entropy Loss with Label Smoothing (0.1)?",
             "Standard cross-entropy pushes true-class probability toward 1.0 and others to 0.0, encouraging overconfident logits. Label smoothing (eps=0.1 in train_mixed.py; 0.05 in MELD fusion) sets target 0.9 + 0.1/7 on the true class and 0.1/7 elsewhere, improving robustness on noisy crowdsourced labels.",
             "", "Formula F2 / train_mixed.py:768", "Medium", "Likely"),
            ("What is Focal Loss and why did Audio/Phase-3 experiments test it?",
             "Focal Loss multiplies cross-entropy by (1 - p_t)^gamma (gamma=2.0), down-weighting easy well-classified majority examples so gradients focus on hard or rare classes like fear and disgust.",
             "On MELD audio, WeightedRandomSampler ultimately beat Focal Loss in Macro-F1 (0.2209 vs 0.1972).", "Formula F3 / audio_model.py:41", "Medium", "Likely"),
            ("Why use AdamW instead of standard SGD or vanilla Adam?",
             "Adam adapts per-parameter step sizes using first and second gradient moments, while AdamW decouples L2 weight decay (wd=1e-4) from the adaptive gradient update so regularization applies uniformly across sparse and dense layers.",
             "", "Formula F11 / train_mixed.py:477", "Medium", "Likely"),
            ("Why use two different learning rates (2e-5 vs 2e-4) in train_mixed.py?",
             "Convolutional stage layer4 already holds strong pretrained facial filters, so a 10x smaller LR (2e-5) fine-tunes features gently without destroying them, while the classification head fc (2e-4) adapts faster to balanced class boundaries. [VERIFY: exact LR grid sweep not logged].",
             "", "train_mixed.py:486-488", "Medium", "Very likely"),
            ("How does CosineAnnealingLR work in train_raf_resnet18.py and train.py?",
             "It smoothly lowers learning rates along a half-cosine curve from initial LR toward zero (or eta_min=1e-5 in MELD fusion) over T_max epochs, helping optimization settle into flatter, better-generalizing minima.",
             "", "train_raf_resnet18.py:565 / train.py:285", "Medium", "Possible"),
            ("How does early stopping prevent wasted epochs or overfitting?",
             "Validation Macro-F1 is checked every epoch; if it fails to improve over patience=3 consecutive epochs (in both train_raf_resnet18.py and train_mixed.py), training stops and retains the best checkpoint (Epoch 6 for RAF-DB, Epoch 7 for Mixed RAVDESS).",
             "", "train_raf_resnet18.py:807 / train_mixed.py:873", "Easy", "Likely"),
            ("Why choose Macro-F1 over Accuracy as our primary selection metric?",
             "Accuracy is dominated by majority classes (like 47.1% neutral in MELD or happy/neutral in RAF-DB). Macro-F1 averages F1 across all 7 emotions equally, forcing models to recognize rare classes like fear and disgust to achieve a high score.",
             "", "Sec 4.5 / Formula F4", "Easy", "Very likely"),
            ("What does 'train-val gap 0.2205' in RAVDESS tell us?",
             "At best Epoch 7, training Macro-F1 on Actors 01-08 exceeded validation Macro-F1 on unseen Actors 09-10 by 0.2205, proving the network partially memorized the 8 training actors' facial identities/styles.",
             "", "Sec 6.1 / webcam_fix_log.md:6", "Hard", "Likely"),
        ]),

        ("Group (d): Project-Specific Architecture, Config & Endpoints (Q35-Q48)", [
            ("Why was remapping RAF-DB classes BY NAME critical in WebcamService?",
             "best_resnet18.pt outputs classes in RAF-DB's original order ['Surprise','Fear','Disgust','Happiness','Sadness','Anger','Neutral'], whereas MELD/our backend uses alphabetical ['anger','disgust','fear','joy','neutral','sadness','surprise']. Remapping by string name (_raf_index_map) prevents predicting 'anger' as 'surprise' by index bug!",
             "", "Sec 3 Step 3.9 / webcam_service.py:307-320", "Medium", "Very likely"),
            ("What happens during the 2-second / 25-frame personal neutral calibration button?",
             "Clicking CALIBRATE FACE records 25 frame logit vectors while the user holds a relaxed face, computes baseline_logits = mean_resting - onehot(neutral), and subtracts 0.70 * baseline_logits on later frames to offset permanent resting brow/mouth biases—without storing photos or altering weights.",
             "", "Sec 3 Step 3.10 / webcam_service.py:736-756", "Medium", "Very likely"),
            ("How does confidence-adaptive EMA smoothing prevent jitter while staying responsive?",
             "EMA blends alpha * raw_probs + (1 - alpha) * prev_ema, where alpha = clip(0.20 + 0.35 * raw_conf, 0.18, 0.55). High-confidence expressions react quickly (up to 55% weight per frame), while ambiguous frames stay smooth (18-20%).",
             "", "Sec 3 Step 3.10 / webcam_service.py:770-778", "Medium", "Likely"),
            ("What is the 3-frame hysteresis state machine?",
             "To eliminate single-frame flickering between two tied classes, a new emotion only replaces the active displayed label after beating the current class probability by > 0.08 across 2 consecutive confirmation updates (candidate_count >= 2, ~3 frames total).",
             "", "Sec 3 Step 3.10 / webcam_service.py:792-820", "Medium", "Likely"),
            ("What exact condition triggers is_uncertain = True?",
             "It triggers if top-1 probability p_top1 < 0.38 OR if the margin between top-1 and top-2 classes (p_top1 - p_top2) < 0.08, painting an 'UNCERTAIN' amber warning badge instead of a false confident emotion. [VERIFY: empirical rationale for 0.38/0.08].",
             "", "Sec 3 Step 3.11 / webcam_service.py:846-852", "Easy", "Very likely"),
            ("What happens if two people sit in front of the webcam simultaneously?",
             "YuNet detects all faces and warns '{N} faces detected; locked onto primary user face.' Spatial tracking (face_dist) compares centers against client['last_bbox'] so detection locks smoothly onto the tracked user instead of jumping back and forth.",
             "", "webcam_service.py:533-541, 580-581", "Easy", "Very likely"),
            ("What happens when NO face is visible to the camera?",
             "process_frame() clears ema_probs and last_bbox so stale state never leaks to the next person, returning face_detected=False, apparent_expression='NO_FACE', uniform 1/7 probabilities (0.1429), and a clear UI prompt.",
             "", "webcam_service.py:556-578", "Easy", "Very likely"),
            ("How does hot-swapping backbones via POST /predict/webcam/backbone work cleanly?",
             "switch_backbone() resolves the requested ID (resnet18_ravdess, resnet18_raf, efficientnet_current) via config.yaml, reloads weights and checkpoint-embedded preprocessing (128 px vs 96 px), and clears client_states so old EMA buffers cannot pollute the new model.",
             "", "webcam_service.py:257-270 / webcam.py:54", "Medium", "Likely"),
            ("What is the difference between /predict/webcam/ws, /frame-json, and /frame?",
             "All three invoke the exact same WebcamService.process_frame(). /ws (and /ws/webcam) keeps one persistent full-duplex socket for fast streaming, /frame-json accepts HTTP JSON with base64 strings, and /frame accepts multipart image UploadFiles.",
             "", "backend/api/webcam.py:68-173", "Easy", "Likely"),
            ("What does GET /health return?",
             "Returns system status, device ('cpu'), loaded flags for audio/text/video/fusion models, active webcam backbone, and checkpoint integrity status—checked by Integration Test 01.",
             "", "backend/main.py / report/tables/api_endpoints.csv", "Easy", "Likely"),
            ("What exact batch mix ratio did train_mixed.py use, and how is it built?",
             "With batch_size=32 and extra_share=0.2, _build_mixed_loaders computes n_extra=6 RAVDESS balanced crops and n_raf=26 RAF-DB balanced crops per step, pairing separate WeightedRandomSamplers so rare emotions stay balanced in both sources.",
             "", "Sec 4.4 / train_mixed.py:606-634", "Hard", "Likely"),
            ("What does prepare_ravdess.py skip and what count remains?",
             "Out of 1,200 MP4 files in Video speech dataset/, it skips 600 non-AV/song clips, drops 80 calm clips, and rejects 2 clips with zero valid sharp faces, retaining 518 speech clips yielding 3,970 aligned crops (3,184 train / 786 val).",
             "", "Sec 4.4 / results/prepare.log", "Medium", "Likely"),
            ("How was 100% live-vs-offline parity actually verified?",
             "eval_clips.py and test_parity_harness.py simulated WebRTC browser capture (320x180 JPEG quality 85) across all 916 frames of the 9 clips, sending frames through the HTTP API and comparing against direct disk evaluation: 916/916 frames (100%) produced identical labels.",
             "", "Sec 4.1 / results/webcam_fix_log.md:4", "Medium", "Very likely"),
            ("How is live webcam + microphone multimodal streaming triggered?",
             "When the user checks the microphone toggle in the Webcam Studio, app.js uses MediaRecorder every timer tick + a 320x180 JPEG snap to POST /predict/webcam/multimodal, combining live facial probabilities with Wav2Vec 2.0 audio (plus optional Whisper/speech transcript).",
             "", "app.js:1678-1812 / webcam_service.py:959", "Medium", "Possible"),
        ]),

        ("Group (e): Multimodal MELD Pipeline, Calibration & Ablations (Q49-Q59)", [
            ("How does the MELD Audio pipeline handle raw MP4/WAV files and silence?",
             "AudioInferenceService loads audio via soundfile/librosa, downmixes stereo to mono, resamples to 16 kHz, removes DC offset, peak-normalizes, trims silence below top_db=30 (preserving >=100 ms), truncates at 30s max, and computes masked mean-pooling of Wav2Vec2 768-D hidden states.",
             "", "Sec 4.6 / backend/services/audio_service.py:62-142", "Medium", "Very likely"),
            ("Why did unweighted Audio Baseline give 49.23% accuracy yet terrible 0.1771 Macro-F1?",
             "MELD contains 47.1% neutral clips, so the baseline MLP learned to predict neutral almost everywhere. Switching to WeightedRandomSampler in StrongerAudioClassifier (527,879 params) forced rare-class recall up, raising macro-F1 to 0.2209 (26% relative gain).",
             "", "Sec 4.6 / report/tables/meld_model_results.csv", "Medium", "Very likely"),
            ("How does the Contextual Text pipeline beat TF-IDF Logistic Regression?",
             "After normalizing contractions (\"I'm\" -> \"I am\"), frozen all-MiniLM-L6-v2 produces 384-D dense semantic sentence vectors capturing synonyms and word order that 12,000 sparse bag-of-n-grams miss, lifting MELD test accuracy 40.15% -> 42.87% and macro-F1 0.2989 -> 0.3326.",
             "", "Sec 4.6 / backend/services/text_service.py", "Easy", "Likely"),
            ("How does MELD uploaded Video prediction work, and why is it weak on MELD?",
             "VideoInferenceService samples 8 uniform frames per clip, center/upper crops, applies CLAHE, averages ResNet-50 2048-D vectors, and feeds VideoClassifier (526,343 params). Because MELD TV clips contain cuts, small side profiles, and non-speaking actors, the 16-D stream achieves only 6.67% fused / 8.57% raw accuracy on N=105.",
             "", "Sec 4.6 / video_service.py & meld_model_results.csv", "Medium", "Likely"),
            ("Is your multimodal fusion Early Fusion, Late Fusion, or Hybrid?",
             "It is a Hybrid Attention Late Fusion network: instead of concatenating raw pixels + 768-D audio + 384-D text vectors early, each modality first produces a compact 16-D summary ([7 probs, 7 centered logits, normalized entropy certainty, quality]), which then interacts through a 2-layer Transformer plus weighted consensus logits.",
             "", "Sec 4.6 / fusion_model.py:98-218", "Hard", "Very likely"),
            ("How do the 16 features per modality encode reliability before attention even runs?",
             "Dimensions 0:7 hold probabilities, 7:14 hold mean-centered log-probabilities, dim 14 holds certainty = 1 - H(p)/log(7) (low when distribution is flat/unsure), and dim 15 holds quality_discount (e.g., 0.85 for dark video or quiet/short audio).",
             "", "backend/services/fusion_service.py:100-115", "Hard", "Likely"),
            ("What happens when the user uploads ONLY Audio+Text without Video?",
             "A boolean presence mask [True, True, False] sets key_padding_mask=True on the missing Video token inside nn.TransformerEncoder AND zeros out its gate contribution, normalizing Audio + Text weights to sum to 1.0 dynamically without crashing or injecting garbage.",
             "", "fusion_model.py:179-203 / fusion_service.py:234-246", "Medium", "Very likely"),
            ("What is consensus_weight (gamma) inside AdvancedCrossModalAttentionFusionNet?",
             "A learned scalar clamped to [0.1, 0.9] blending the Transformer neural classifier logits (1 - gamma) with the attention-gate-weighted sum of raw unimodal logits (gamma * consensus_logits), stabilizing predictions on scarce multimodal training data (210 train / 105 val / 105 test).",
             "", "fusion_model.py:162, 206-215", "Hard", "Possible"),
            ("What is Expected Calibration Error (ECE) and why does it matter?",
             "ECE bins predictions (15 bins) and measures the weighted average gap between stated confidence (e.g., 85%) and actual empirical accuracy (e.g., 60%). Overconfident wrong answers destroy trust in human-AI systems; temperature scaling reduced our fusion ECE from 0.2130 down to 0.0734.",
             "", "Sec 6.2 / Formula F22 / metrics.py", "Medium", "Very likely"),
            ("Explain the difference between offline T*=0.5953 and live runtime T_adaptive=0.24-0.28.",
             "train.py fits post-hoc L-BFGS scalar temperature T*=0.5953 on validation logits (producing the N=105 ECE=0.0734 benchmark). Live MultimodalFusionService adds concordance scaling + adaptive T (0.28 consensus / 0.24-0.28 partial / 0.26 conflicting) for sharper UI display [VERIFY: rationale for tighter runtime T vs T*].",
             "", "train.py:354 vs fusion_service.py:196-221", "Hard", "Possible"),
            ("What did the 7-configuration MELD ablation prove?",
             "Text-only (37.14%, ECE 0.1903) and Audio+Text (37.14%, macro-F1 0.3670, ECE 0.1146) slightly outperformed trimodal A+V+T in raw accuracy (36.19%) because noisy video features pulled down clean text, yet trimodal A+V+T achieved the lowest calibration error (ECE=0.0734) and balanced Brier/NLL.",
             "", "Sec 6.2 / Table 6.4 / Fig 6.4", "Hard", "Very likely"),
        ]),

        ("Group (f): Results, Interpretation & Literature Comparison (Q60-Q70)", [
            ("Why did all 3 webcam backbones have ~31.5-33.6 ms p50 latency despite ResNet-18 being larger than EfficientNet-B0?",
             "On 320x180 CPU frames, shared OpenCV YuNet face detection, affine rotation, crop slicing, and Python pre/post-processing dominate frame runtime far more than 96 px EfficientNet vs 128 px ResNet-18 forward passes—giving almost identical ~30 FPS speed across all three.",
             "", "Sec 6.2 / Table 6.3", "Medium", "Very likely"),
            ("Why do earlier single-run latency numbers (like 102.33 ms in webcam_fix_log) not match 31.50 ms?",
             "Background CPU load, thermal throttling, and clip lengths varied across separate historical terminal runs. Only Table 6.3 / latency_benchmark.csv measured all three backbones back-to-back on the exact same 200 frames of Disgust (2).mp4 under uniform idle CPU load.",
             "", "Sec 6.2 / report/tables/latency_benchmark.csv", "Medium", "Likely"),
            ("Why did ResNet-18 RAVDESS beat ResNet-18 RAF on RAF-DB macro-F1 too (0.7077 vs 0.6721)?",
             "RAVDESS clips have balanced class representation across 7 emotions, plus mixed training used class-balanced samplers and label smoothing (0.1)—lifting weak minority-class F1 on RAF-DB test while maintaining overall accuracy (79.27% vs 79.53%).",
             "", "Sec 6.1 / Table 6.1", "Medium", "Very likely"),
            ("Which emotion class is strongest on RAF-DB and RAVDESS validation, and which is weakest?",
             "According to report/tables/rafdb_and_ravdess_metrics.csv, Joy is strongest (F1 = 0.9182 on RAF-DB test and 0.9039 on RAVDESS val for resnet18_ravdess). On RAVDESS val, Neutral (0.3714) and Fear (0.4404) are weakest due to subtle neutral/fear facial ambiguity during speech.",
             "", "report/tables/rafdb_and_ravdess_metrics.csv", "Medium", "Likely"),
            ("Why do published papers report ~90-92% on RAF-DB while our ResNet-18 reaches ~79.5%?",
             "Verified state-of-the-art RAF-DB models like POSTER++ (92.21%), DDAMFN (91.35%), and EAC (89.99-90.35%) use heavy Vision Transformers, landmark dual-branches, or ResNet-50 backbones at 224x224 on GPUs. Our lightweight 128x128 ResNet-18 traded ~12 points of static peak accuracy for ~30 FPS CPU inference and dynamic webcam generalization.",
             "", "Literature Ref [6][7][9][10]", "Hard", "Very likely"),
            ("How do published MELD models get ~62-67% while our fusion model scores 36.19%?",
             "Published MELD state of the art (TelME ~66.4-67.4%, UniMSE 65.51%, MM-DFN 62.49%) evaluates on the imbalanced 2,610-utterance test set (where predicting 47% neutral boosts accuracy automatically) using multi-turn conversational dialogue context across entire scenes. Our fusion benchmark evaluates single isolated utterances on a strictly class-balanced 105-sample subset (15 per class, random chance = 14.3%).",
             "", "Sec 5 & 6.2 / Literature Ref [21][23][24]", "Hard", "Very likely"),
            ("What does Brier Score measure alongside ECE?",
             "Brier Score is the mean squared error between the 7 predicted probabilities and the one-hot ground-truth vector—combining both calibration and sharpness. Optimization reduced multimodal Brier score from 0.8811 to 0.7785 (NLL 1.9835 -> 1.7244).",
             "", "Formula F22 / optimization_comparison.json", "Hard", "Possible"),
            ("Why is weighted-F1 on MELD Text (0.4725) higher than macro-F1 (0.3326)?",
             "Weighted-F1 weights each class's F1 by its support count in the 2,610 test utterances, so high accuracy on frequent neutral and joy pulls Weighted-F1 upward, while Macro-F1 weights rare disgust and fear equally.",
             "", "Table 6.4 / meld_model_results.csv", "Medium", "Likely"),
            ("Why does MELD Video report 37.14% (N=35) in one table row and 6.67% (N=105) in another?",
             "The 37.14% figure comes from an offline checkpoint report on 35 MELD samples (whose confusion matrix wasn't saved), whereas 6.67% (8.57% raw X_v argmax) is measured directly on the 105-sample balanced multimodal test tensor data/fusion_test_dataset.pt.",
             "", "Table 6.4 / report/assets_manifest.md:FIG-12", "Hard", "Possible"),
            ("What happened in tests/run_tests.py --all and how many checks passed?",
             "All 5 unit test suites (audio, text, video, fusion, plus 22 checks inside test_webcam.py) and all 11 FastAPI integration tests passed (16/16 PASS, zero skipped or relaxed assertions).",
             "", "report/tables/test_cases.csv / Sec 10", "Easy", "Very likely"),
            ("What is a '1-of-9 clip sensitivity'? ",
             "With only 9 discrete evaluation MP4 files, getting even 1 single extra clip right shifts clip-level pass rate by 1/9 = 11.11 percentage points—making clip count far noisier than 3,068-image RAF-DB test evaluations.",
             "", "Sec 6.1 & Sec 8", "Medium", "Likely"),
        ]),

        ("Group (g): Hostile & Tricky Questions Teachers Love (Q71-Q84)", [
            ("Is your system 100% accurate?",
             "No—and any system claiming 100% on real facial affect is overfitting or fabricating data. Our best webcam backbone achieves 79.27% on RAF-DB test, 56.87% on unseen RAVDESS actors, and 57.31% pooled (5/9 clips) on personal clips, with honest 'Uncertain' warnings when confidence or margin drops.",
             "", "Sec 6 / Sec 8", "Easy", "Very likely"),
            ("Why only 5/9 clips correct on webcam? Isn't 57% too low?",
             "We must judge 57.31% against where we started and why the failures happen: the baseline EfficientNet-B0 scored only 15.61% (1/9 clips—Neutal only), and RAF-only ResNet-18 hit 44.0% (4/9 clips). Mixed RAVDESS adaptation tripled frame accuracy over baseline while fixing pipeline bugs and showing clearly why static single-frame models miss subtle anger and sadness.",
             "", "Sec 1.1 & Sec 6.1", "Hard", "Very likely"),
            ("Did you test on your training data?",
             "Never. RAF-DB uses the untouched official 3,068-image test split with cross-split duplicate verification; RAVDESS splits strictly by actor ID (01-08 train vs 09-10 val); MELD tests on the 2,610 test split and 105-sample fusion test tensor; and New folder clips were never touched during training.",
             "", "Sec 5 / train_raf_resnet18.py / prepare_ravdess.py", "Easy", "Very likely"),
            ("Did you tune or select hyperparameters on your 9 webcam test clips?",
             "Important honest distinction: training weights never saw the 9 clips, and hyperparameters came from RAF+RAVDESS validation (best Epoch 7 = 0.7051). HOWEVER, those 9 clips were used both to compare backbones and promote resnet18_ravdess to default as well as to report final webcam metrics—so we explicitly disclose they are not an untouched multi-user test set.",
             "Note extra reserved clips with suffix ' (3)' remain untouched in New folder/ for clean final checking.", "Sec 8 / webcam_fix_log.md:17", "Hard", "Very likely"),
            ("Why does your model fail completely on Anger (0%)?",
             "On Anger.mp4, tight furrowed brows and tense eyes under normal desk lighting look structurally similar to Surprise (61.39% predictions) or Neutral (38.61%) to a static 2D CNN without voice loudness or temporal onset cues. Solving this requires dynamic temporal sequence modeling or opt-in vocal/multimodal fusion.",
             "", "Sec 6.1 / Table 6.2", "Medium", "Very likely"),
            ("Will this work equally well for other people outside your team?",
             "Partially: it learns generalizable facial structure from 10,428 RAF-DB strangers plus Actors 01-08, yet our 9-clip test covered only one speaker, and the 0.2205 RAVDESS actor gap proves unseen individuals still experience domain shift (lighting, skin tone, resting face shape). Using the 25-frame neutral calibration helps normalize individual resting baselines.",
             "", "Sec 8 / webcam_service.py:736", "Medium", "Very likely"),
            ("Is this detecting real felt emotions or acted facial expressions?",
             "Strictly apparent outward expressions—never true internal feelings. RAVDESS consists of paid actors reading scripts, MELD comes from TV actors (Friends), and our 9 clips are posed expressions; our UI displays an explicit non-clinical apparent-expression disclaimer on every response.",
             "", "webcam_service.py:576, 910", "Easy", "Very likely"),
            ("What about privacy, consent, and GDPR/biometric concerns?",
             "Live webcam frames and microphone buffers execute strictly inside volatile CPU RAM on 127.0.0.1:8000 and are garbage-collected immediately after JSON prediction—nothing touches disk unless a developer explicitly posts to /predict/webcam/debug-capture. Personal datasets and raw videos are also git-ignored.",
             "", "Sec 4.6 / webcam_service.py:499", "Easy", "Very likely"),
            ("What is YOUR own engineering contribution versus pretrained models and public datasets?",
             "We reused open pretrained feature extractors (ImageNet ResNet-18/50, Wav2Vec2, MiniLM, YuNet) and public datasets, which is standard modern practice. Our direct engineering work was: diagnosing & proving 916/916 live-vs-offline parity, fixing CSS-only mirroring vs unmirrored inference, removing harmful AU rules (+0.0996 macro-F1), writing prepare_ravdess.py + mixed trainer train_mixed.py with synthetic glasses augmentation, balancing MELD audio sampling, designing AdvancedCrossModalAttentionFusionNet with 16-D entropy/quality gates & calibration, and building the full FastAPI + Web Studio + 16-test verification suite.",
             "", "Sec 2 Code Map / git log", "Hard", "Very likely"),
            ("Why didn't you use a huge Vision Transformer (like POSTER++ or Swin/ViT) instead of ResNet-18?",
             "Larger transformers push CPU per-frame latency far higher on laptop hardware without CUDA GPUs, hurting real-time responsiveness. ResNet-18 gave us 32.98 ms p50 (~30 FPS) alongside YuNet detection while holding 79.27% RAF-DB test accuracy.",
             "", "Sec 4.1 & Sec 6.2 / Tool Justification", "Medium", "Very likely"),
            ("If you had two more months, how would you concretely improve this system?",
             "First, gather a multi-subject multi-lighting webcam test benchmark to eliminate single-speaker evaluation bias. Second, train a lightweight temporal sequence head (e.g., TCN/attention over 16 frames) instead of single-frame EMA so neutral vs sadness/anger dynamics separate. Third, evaluate HSEmotion/EmotiEffNet checkpoints inside our runtime switcher alongside speaker-turn context on MELD.",
             "", "Sec 8 (Unimplemented Future Work)", "Medium", "Very likely"),
            ("What happens if a user wears glasses, sits in dim lighting, has two faces behind them, or leaves the camera?",
             "Glasses: train_mixed.py explicitly augmented 30% of training crops with dark eyeglass rims + specular glare to resist rim occlusion. Dim light: mean luminance < 60 triggers 1.35 gamma brightening (low_light warning < 35). Two faces: spatial face_dist tracks last_bbox with a multi-face warning. No face: resets EMA buffer and returns NO_FACE cleanly.",
             "", "webcam_service.py:533-664 / train_mixed.py:289", "Medium", "Very likely"),
            ("Why did Text-only (37.14%) beat your Trimodal Fusion (36.19%) in accuracy on MELD?",
             "In TV dialogue (MELD), words carry crisp emotional polarity while video tracks frequently show occluded or neutral listener faces (video 16-D scored 6.67%). Rather than blindly trusting noisy video, trimodal fusion sacrificed <1% raw point accuracy on 105 samples to cut Expected Calibration Error in half (0.1903 text-only -> 0.0734 trimodal).",
             "", "Sec 4.6 & Sec 6.2 / Fig 6.4", "Hard", "Very likely"),
            ("Wait—did you actually capture live webcam video for your report UI screenshot?",
             "No—because headless Chrome verification ran without an attached physical webcam hardware device, ui_live_webcam.png shows the real Web Studio UI in verified STANDBY mode ('Camera: Inactive'), while frame accuracy and latency numbers were rigorously measured on recorded evaluation MP4 clips in New folder/ via the identical API pipeline.",
             "", "Sec 8 / Sec 10", "Hard", "Likely"),
        ]),

        ("Group (h): Ethics, Bias, Privacy & Sustainability (Q85-Q92)", [
            ("How can facial emotion recognition models exhibit demographic or lighting bias?",
             "If training data underrepresents darker skin tones, eyewear, older faces, or regional expressiveness—and if RAVDESS contains only North American English actors—accuracy drops systematically on underrepresented users. That is why we must disclose dataset demographics (Actors 01-10) and warn against high-stakes automated decisions.",
             "", "Sec 5 & Sec 8", "Medium", "Likely"),
            ("Why should affective computing NEVER be used for automated hiring, policing, or medical diagnosis?",
             "Because apparent facial cues have ~40%+ error even on benchmarks and vary wildly with culture, pain, tiredness, or resting anatomy. Our output includes an explicit disclaimer confirming it does not measure internal psychological states.",
             "", "webcam_service.py:910", "Easy", "Very likely"),
            ("How does your architecture support environmental sustainability?",
             "Rather than training trillion-parameter multimodal models on GPU farms, we freeze heavy pretrained encoders (Wav2Vec2, MiniLM, ResNet-50), freeze 24.9% of ResNet-18 weights up to layer3, and run real-time ~30 FPS inference purely on a standard 15W/45W laptop CPU.",
             "", "Sec 4.1 / environment.csv", "Medium", "Likely"),
            ("Explain your Git hygiene rules regarding personal biometric files.",
             ".gitignore excludes raw datasets (DATASET/, MELD.Raw/, ravdess_crops/), private New folder/ recordings, live_debug/ captures, and heavy checkpoints so personal faces and licensed third-party media never leak into version history.",
             "", ".gitignore / commit 52afcab", "Easy", "Possible"),
            ("How do you protect uploaded audio/video files on the MELD endpoints?",
             "Uploaded files are handled via temporary server buffers, validated against empty/corrupt inputs (blank text or empty forms return HTTP 400 Bad Request, Integration Tests 04 and 11), and deleted immediately after inference.",
             "", "backend/utils/file_utils.py / test_cases.csv", "Medium", "Possible"),
            ("How could an organization verify your system's calibration over time?",
             "Our Research & Benchmarks UI tab and evaluate.py export ECE, Brier score, NLL, and reliability bins on held-out calibration datasets so silent overconfidence drift is detected early.",
             "", "Sec 4.6 / metrics.py", "Medium", "Possible"),
            ("What does PO8 (Ethics) require in your project implementation?",
             "Zero falsified metrics or fake screenshots, proper IEEE attribution of datasets/libraries, honest reporting of failed classes (anger/sadness 0%), and volatile RAM privacy on live streams.",
             "", "Report Declaration & Ch 6", "Easy", "Possible"),
            ("Is the software completely free of proprietary runtime fees?",
             "Yes—all frameworks (PyTorch, TorchVision, OpenCV, FastAPI, Transformers, scikit-learn, Librosa) are open source (MIT/BSD/Apache 2.0) and run offline locally with INR 0 API cost.",
             "", "bill_of_materials.csv", "Easy", "Possible"),
        ]),

        ("Group (i): Teamwork & Project Management (Q93-Q100)", [
            ("How did the team divide technical ownership across the project?",
             "Responsibilities split cleanly along subsystem boundaries: Member 1 on MELD fusion, calibration, FastAPI, and tests; Member 2 on YuNet tracking, live/offline parity, prepare_ravdess.py, and mixed ResNet-18 training; Member 3 on Wav2Vec2 imbalance experiments, MiniLM NLP, Web Studio SPA, and benchmarking.",
             "", "Sec 1.3 Table 1.1", "Easy", "Very likely"),
            ("How many commits track your actual development history and what phases occurred?",
             "Git history records the evolution across refactoring into 4 canonical root modules (ac707de), Phase 1-3 webcam benchmarks & focal/EMA/alignment upgrades (1a38ba0..e2a87ff), parity + mirror + ResNet-18 RAF integration (4e26e3c..1612f58), RAVDESS prep & mixed training + promotion (52afcab..8871ddc), cleanup (08227b5), and report packaging (278a762).",
             "", "git log --oneline -n 60", "Medium", "Likely"),
            ("What was the single highest-impact bug fix your team found together?",
             "Discovering live-vs-offline crop/parity mismatches plus removing the hardcoded AU geometry rules—which immediately fixed resting false-surprise predictions and jumped MELD validation Macro-F1 by +0.0996 (0.2872 -> 0.3868) before we even trained RAVDESS.",
             "", "results/webcam_fix_log.md:4", "Medium", "Very likely"),
            ("What risk management strategy handled the failure of baseline EfficientNet-B0?",
             "Instead of deleting the baseline, we kept it selectable inside WebcamService alongside resnet18_raf and promoted resnet18_ravdess ONLY after it passed all quantitative promotion gates (majority 5/9 > 4/9, pooled 57.3% > 44.0%, RAF-DB F1 +0.0356, p95 <= 150 ms) with automatic fallback if a checkpoint is missing.",
             "", "webcam_fix_log.md:17 / webcam_service.py:115", "Medium", "Likely"),
            ("How did you prevent breaking existing modules while iterating rapidly?",
             "Every milestone ran tests/run_tests.py --all (5 unit suites incl. 22 webcam parity/alignment/switch checks + 11 live FastAPI integration calls) plus test_parity_harness.py.",
             "", "tests/run_tests.py", "Easy", "Likely"),
            ("Why keep legacy checkpoints like best_webcam_model.pth and best_text_model.joblib in the switcher/fallback?",
             "Keeping baselines callable lets us demonstrate quantitative before-vs-after progression live during viva and ensures graceful service startup fallback if an external model file is relocated.",
             "", "webcam_service.py:234 / text_service.py:61", "Medium", "Possible"),
            ("What project management artifact ties code modules to your academic schedule?",
             "Our Work Breakdown Structure (WBS) across 6 work packages and the 10-phase work breakdown Gantt timeline across relative semester weeks without fixed calendar dates (report/figures/fig_diag_wbs.png and fig_gantt_git_timeline.png).",
             "", "Report Ch 1 Fig 1.1-1.2", "Easy", "Possible"),
            ("If an examiner asks to inspect how tests pass right now, what single command do you run?",
             "Run .\\.venv\\Scripts\\python.exe main.py --test (or .\\.venv\\Scripts\\python.exe tests\\run_tests.py --all) to execute all unit and live ASGI integration checks end to end.",
             "", "main.py:101 / tests/run_tests.py", "Easy", "Very likely"),
        ]),
    ]

    q_counter = 1
    for grp_title, qlist in groups:
        add_h2(doc, grp_title)
        for qt, sa, dp, ptr, df, lk in qlist:
            add_q_item(doc, q_counter, qt, sa, dp, ptr, df, lk)
            q_counter += 1


# ---------------------------------------------------------------------------
# SECTION 10: LIVE DEMO SCRIPT & FALLBACKS
# ---------------------------------------------------------------------------
def build_sec10(doc: Document) -> None:
    add_h1(doc, "10. Live Demo Script, Commands & Failure Fallbacks", page_break=True)

    add_h2(doc, "10.1 Exact PowerShell Commands to Launch Live", )
    add_script_box(
        doc,
        "Run in Repo Root (c:\\Users\\harsh\\Desktop\\Major project)",
        "1. Optional Quick Health Verification (before entering viva room):\n"
        "   .\\.venv\\Scripts\\python.exe main.py --test\n\n"
        "2. Start the Web Studio FastAPI Server:\n"
        "   .\\.venv\\Scripts\\python.exe main.py --serve\n"
        "   -> Open Chrome/Edge at: http://127.0.0.1:8000/\n\n"
        "3. Optional Terminal Multimodal Quick Demo (no browser needed):\n"
        "   .\\.venv\\Scripts\\python.exe main.py --demo",
        bg="F1F5F9",
    )

    demo_steps = [
        ("1. Home / Landing Overview", "Briefly point out the 7-emotion schema and five local services loaded on CPU.", "Unified Multimodal dashboard loads preset MELD cards without internet."),
        ("2. Live Webcam Tab -> Calibrate", "Click Real-Time Webcam -> select resnet18_ravdess (default) -> click Start Camera -> click CALIBRATE FACE and hold relaxed face 2 sec (25 frames).", "Bounding box + 5 cyan landmarks track eye-aligned face; 'CALIBRATED FOR USER' appears; resting face locks calmly to NEUTRAL."),
        ("3. Smile (Joy)", "Smile clearly toward camera for 2 seconds (hysteresis needs ~3 stable frames).", "Top label flips smoothly to JOY (>85%), tier 'Robust', Joy bar rises without backward text."),
        ("4. Neutral & Surprise", "Return relaxed -> NEUTRAL; raise eyebrows + open mouth -> SURPRISE.", "Demonstrates reversible smooth hysteresis transitions (~30 FPS, ~33 ms latency pill)."),
        ("5. Subtle / Ambiguous -> Uncertain", "Make a tiny half-smile or slightly confused look.", "Triggers UNCERTAIN amber tag when top-1 < 0.38 or margin < 0.08—say 'Notice our ethical safety flag!'"),
        ("6. Eyeglasses Test (if applicable)", "Put on glasses to demonstrate rim + glare robustness from MixedAugmenter (30% glasses augmentation).", "YuNet still locks 5 landmarks and keeps tracking."),
        ("7. Video / Audio / Text Studio -> Multimodal Preset", "Open Multimodal Unified tab -> click MELD Dia0 Utt0 preset (Friends coffee mugs clip) -> Run Analysis.", "Shows Audio (Wav2Vec2), Text (MiniLM), Video (ResNet-50), attention gate weights A/T/V, conflict/consensus status, and calibrated confidence."),
        ("8. Benchmarks & Research Tab", "Open Benchmarks tab briefly to show real stored MELD ablation table & calibration curve live inside the app.", "Reinforces transparency and empirical rigor."),
    ]
    add_table(doc, "Table 10.1: Step-by-Step Live Demo Choreography",
              ["Step", "What to Click & Say", "Expected Screen Output"], demo_steps, [1.25, 2.75, 2.45], font_pt=8.4)

    add_h2(doc, "10.2 \"If It Fails\" Troubleshooting Table")
    fails = [
        ("Browser blocks camera permission", "Click lock icon left of URL bar -> Allow Camera -> Reload; or verify Windows Privacy Settings -> Camera access ON.", "Switch immediately to Video Upload tab or 30s screenshot fallback."),
        ("Displays 'NO FACE DETECTED' continuously", "Face camera directly within 0.4-1.0 m, increase desk light above mean luminance 35, avoid extreme >45 deg side profile.", "Click Restart Camera button or lower glare behind head."),
        ("Wrong label stuck (e.g., resting face reads Surprise)", "Ensure resnet18_ravdess (not efficientnet_current) is active in dropdown; click CALIBRATE FACE for 25 frames or Reset Calibration + Reset Session.", "Explain resting neutral calibration explicitly!"),
        ("Server fails to start (WinError 10048 port busy)", "Another uvicorn instance is already holding port 8000; either open http://127.0.0.1:8000 directly or run: .\\.venv\\Scripts\\python.exe main.py --serve --port 8001.", "Check task manager if zombie python holds camera."),
        ("Slow initial prediction (>1-2 s on first click)", "First audio/video upload warms lazy CPU Torch graphs and Wav2Vec2/ResNet-50 caches; subsequent requests drop to ~180-800 ms (and webcam runs at ~33 ms).", "Run 'main.py --demo' before panel walks in to pre-warm OS file cache."),
    ]
    add_table(doc, "Table 10.2: Emergency Demo Troubleshooting & Fixes",
              ["Failure Symptom", "Immediate Live Fix", "Backup Action"], fails, [1.55, 3.05, 1.85], font_pt=8.3)

    add_h2(doc, "10.3 30-Second Safe Fallback Using Captured Screenshots")
    add_p(doc,
          "If hardware webcam or live server fails completely, open report/Project_Report.pdf Chapter 4 (or Figures 10.1-10.3 below) and walk the panel through "
          "these three real screenshots in 30 seconds: (1) Live Webcam Studio (noting standby status + backbone selector), (2) Video/Audio/Text Studio, and "
          "(3) Unified Multimodal dia0_utt0 output with attention weights and Benchmarks reliability plot.")
    add_img(doc, "report/figures/ui_live_webcam.png", "Figure 10.1 (Fallback 1): Live Webcam Studio showing ResNet-18 (RAF+RAVDESS) default & status controls.", width_in=4.8)
    add_img(doc, "report/figures/ui_multimodal_unified.png", "Figure 10.2 (Fallback 2): Unified MELD Multimodal dia0_utt0 prediction with attention gates.", width_in=4.8)
    add_img(doc, "report/figures/ui_results_benchmarks.png", "Figure 10.3 (Fallback 3): Live UI Benchmarks & Calibration curve view.", width_in=4.8)


# ---------------------------------------------------------------------------
# SECTION 11: CHEAT SHEETS
# ---------------------------------------------------------------------------
def build_sec11(doc: Document) -> None:
    add_h1(doc, "11. One-Glance Cheat Sheets (Numbers, Formulas, Glossary & Rapid-Fire)", page_break=True)

    add_h2(doc, "11.1 Numbers to Remember (100% Real Verified Figures)")
    nums = [
        ("Webcam Pooled Frame Acc (9 clips, 916 frames)", "EffNet: 15.61% (143/916, 1/9) | R18-RAF: 44.00% (403/916, 4/9) | R18-RAVDESS: 57.31% (525/916, 5/9)"),
        ("RAF-DB Test (N=3,068)", "EffNet: 20.4% / 0.1544 F1 | R18-RAF: 79.53% / 0.6721 F1 | R18-RAVDESS: 79.27% / 0.7077 F1"),
        ("RAVDESS Val (Actors 09-10, N=786)", "3,970 total crops (518/1,200 clips); 3,184 train / 786 val | R18-RAVDESS: 56.87% Acc / 0.5354 F1 (gap 0.2205)"),
        ("Unified 200-Frame CPU Latency", "EffNet (96px): 33.63/35.82 ms (29.73 FPS) | R18-RAF (128px): 31.50/33.26 ms (31.75 FPS) | R18-RAV: 32.98/34.59 ms (30.32 FPS)"),
        ("Exact Model Parameter Counts", "ResNet-18: 11,180,103 total (8,397,319 train >layer3) | EffNet-B0: 4,337,283 | Audio StrongerMLP: 527,879 | Fusion Opt: 167,113"),
        ("MELD Unimodal & Fusion (N=2,610 / N=105)", "Audio Bal Macro-F1: 0.1771->0.2209 | Text MiniLM: 42.87% / 0.3326 | Text-Only(105): 37.14% | Opt Fusion(105): 36.19% / 0.3574 / ECE 0.0734 (down from 0.2130, T*=0.5953)"),
        ("Parity, AU Rule Ablation & Tests", "Live-vs-disk parity: 916/916 (100% @ 320x180 JPEG85) | AU rules removed: MELD val F1 0.2872->0.3868 | Tests: 5/5 unit + 11/11 API = 16/16 PASS"),
    ]
    add_table(doc, "Table 11.1: Must-Know Verified Quantitative Numbers", ["Metric / Domain", "Exact Value(s) to Quote"], nums, [2.15, 4.30], font_pt=8.4)

    add_h2(doc, "11.2 Hyperparameter Reference Table (report/tables/hyperparameters.csv)")
    add_csv(doc, "Table 11.2: Exact Training Hyperparameters Across All Subsystems",
            "report/tables/hyperparameters.csv", [1.05, 1.25, 1.00, 1.15, 0.95, 1.05], font_pt=7.5)

    add_h2(doc, "11.3 Complete Formula Sheet (All 22 Equations Reused from report/formulas/)")
    formulas = [
        ("F1 Softmax", "report/formulas/F1_softmax.png", "z_c: raw logits (7 classes); converts logits into probabilities summing to 1."),
        ("F2 Label-Smoothed CE", "report/formulas/F2_cross_entropy_label_smoothing.png", "eps=0.1 (mixed) or 0.05 (fusion); softens 1-hot target to curb overconfidence."),
        ("F3 Focal Loss", "report/formulas/F3_focal_loss.png", "alpha_y: class weight, gamma=2.0 focusing factor down-weighting easy samples."),
        ("F4 Classification Metrics", "report/formulas/F4_classification_metrics.png", "TP/FP/FN per class c; unweighted Macro-F1 averages all 7 classes equally."),
        ("F5 Confusion Matrix", "report/formulas/F5_confusion_matrix.png", "M_ij count true i predicted j; row-norm M~_ij gives per-class recall."),
        ("F6 Class-Balanced Weight", "report/formulas/F6_class_balanced_weight.png", "w_c = N / (C * N_c); inverse frequency weight for WeightedRandomSampler."),
        ("F7 Input Normalization", "report/formulas/F7_input_normalization.png", "mu=[0.485,0.456,0.406], std=[0.229,0.224,0.225] channel-wise ImageNet RGB scaling."),
        ("F8 ResNet Block", "report/formulas/F8_resnet_residual_block.png", "y = F(x, {W_i}) + W_s x; residual skip path stops vanishing gradients."),
        ("F9 2D Conv + ReLU", "report/formulas/F9_conv_relu.png", "Spatial filter W*X + b followed by max(0, .) activation."),
        ("F10 Batch Norm", "report/formulas/F10_batch_norm.png", "Normalizes activations by batch moments; frozen in ResNet stages <= layer3."),
        ("F11 AdamW Update", "report/formulas/F11_adamw_update.png", "Adaptive 1st/2nd moment step + decoupled weight decay lambda=1e-4."),
        ("F12 Mixed Batch Rule", "report/formulas/F12_mixed_batch_composition.png", "B=32, extra_share=0.2 -> n_extra=6 RAVDESS + n_raf=26 RAF-DB crops/batch."),
        ("F13 Aligned Square Crop", "report/formulas/F13_aligned_square_crop.png", "Eye angle atan2(dy, dx), |angle|<=45, square side max(w,h)*pad (1.10/1.25)."),
        ("F14 Blur Gate", "report/formulas/F14_blur_quality_gate.png", "Var(Laplacian(Gray(crop))) < 30.0 triggers blur rejection / prediction hold."),
        ("F15 Majority & Pooled Acc", "report/formulas/F15_majority_and_pooled_acc.png", "Pooled sums correct frames / 916; majority checks modal label per clip (5/9)."),
        ("F16 Uncertain Rule", "report/formulas/F16_uncertain_rule.png", "p_top1 < 0.38 or (p_top1 - p_top2) < 0.08 flags UNCERTAIN output."),
        ("F17 Percentile Latency", "report/formulas/F17_percentile_latency.png", "Ordered frame ms timings L_(k) give p50, p95, and FPS = 1000 / p50."),
        ("F18 EfficientNet Scaling", "report/formulas/F18_efficientnet_compound_scaling.png", "Compound phi scales depth d, width w, resolution r jointly (96 px legacy)."),
        ("F19 Adaptive EMA", "report/formulas/F19_ema_smoothing.png", "alpha = clip(0.20 + 0.35 * raw_conf, 0.18, 0.55) smooths frame jitter."),
        ("F20 MELD Audio Pool", "report/formulas/F20_meld_audio_pooling_mlp.png", "Masked mean pool of Wav2Vec2 hidden states -> StrongerAudioClassifier MLP."),
        ("F21 MELD Fusion Eq.", "report/formulas/F21_meld_fusion_equations.png", "Transformer attention on 16-D stream vectors + entropy gates + consensus residual."),
        ("F22 Calibration Metrics", "report/formulas/F22_calibration_metrics.png", "15-bin ECE (|acc - conf| gap) and multi-class Brier mean squared prob error."),
    ]
    for title, fpath, sym in formulas:
        add_p(doc, f"{title}: {sym}", bold=False, size=10.5, sa=1.0)
        add_img(doc, fpath, title, width_in=4.2)

    add_h2(doc, "11.4 Technical Glossary & Abbreviations (65 Essential One-Liners)")
    glossary = [
        ("AdamW", "Adaptive Moment Estimation optimizer with decoupled L2 weight decay (wd=1e-4)."),
        ("Affine Alignment", "2D rotation+translation around eye midpoint leveling head roll before cropping."),
        ("all-MiniLM-L6-v2", "Compact 384-D distilled Sentence-Transformer producing text embeddings."),
        ("ASGI", "Asynchronous Server Gateway Interface powering FastAPI HTTP + WebSocket routes."),
        ("AU (Action Unit)", "Anatomical facial muscle movement descriptor; hard-coded AU rules were removed."),
        ("AudioClassifier", "Baseline MELD 768->256->7 MLP achieving 0.1771 Macro-F1 (neutral collapse)."),
        ("Backbone Switcher", "POST /predict/webcam/backbone hot-swapping 3 CNN checkpoints safely."),
        ("BatchNorm (BN)", "Layer stabilizing activations via batch mean/variance (frozen through layer3)."),
        ("BGR vs RGB", "OpenCV decodes JPEG as Blue-Green-Red while PyTorch ResNet requires Red-Green-Blue."),
        ("BORDER_REPLICATE", "OpenCV edge-pixel replication preventing artificial black borders during rotation."),
        ("Brier Score", "Mean squared probability error measuring calibration plus confidence sharpness."),
        ("CalibratedClassifierCV", "Scikit-learn cross-validated probability calibrator wrapping text predictions."),
        ("Calibration Offset", "25-frame user resting neutral logit average subtracted at 0.70 weight."),
        ("CenterCrop(128)", "Deterministic central 128x128 crop taken after Resize(146) during eval."),
        ("Class Imbalance", "Unequal class counts (e.g., 47.1% MELD neutral) skewing unweighted models."),
        ("client_id", "Browser-generated session key keeping smoothing & baseline state isolated per user."),
        ("Consensus Residual", "Learnable weighted sum of unimodal raw logits blended with neural fusion logits."),
        ("CosineAnnealingLR", "Scheduler lowering LR smoothly along a cosine curve over training epochs."),
        ("Cross-Entropy (CE)", "Log-loss penalizing low predicted probability on the true class."),
        ("Cross-Modal Attention", "Multi-head Transformer allowing audio, text, and video tokens to query each other."),
        ("Data Leakage", "Accidental overlap between training and test subjects/images; prevented via actor/MD5 split."),
        ("Domain Shift", "Accuracy drop when test camera, lighting, or expressions differ from training datasets."),
        ("Early Stopping", "Halting training when validation Macro-F1 stagnates for patience=3 epochs."),
        ("ECE", "Expected Calibration Error: average |accuracy - confidence| gap across 15 bins."),
        ("EfficientNet-B0", "Legacy 96 px 4.34M-param checkpoint scoring 15.61% on personal webcam clips."),
        ("EMA Smoothing", "Exponential moving average blending current/prior frame probabilities dynamically."),
        ("Entropy Certainty", "1 - normalized Shannon entropy; approaches 0 on flat uncertain distributions."),
        ("eval_resize (146)", "Pre-crop resize scale matching TorchVision evaluation geometry before CenterCrop(128)."),
        ("extra_share (0.2)", "Fraction of mixed batch drawn from RAVDESS (6 of 32 samples per step)."),
        ("FastAPI", "Python web framework serving REST JSON endpoints, WebSockets, and static SPA UI."),
        ("Focal Loss", "Modulated CE loss (gamma=2.0) focusing gradients on hard/minority samples."),
        ("freeze-until layer3", "Freezing conv1..layer3 (24.9% ResNet-18 weights) while adapting layer4 + fc."),
        ("Gamma LUT", "Global pixel power-law lookup table adjusting dark/bright crops without CLAHE shadows."),
        ("GELU", "Smooth Gaussian Error Linear Unit used inside AdvancedCrossModalAttentionFusionNet."),
        ("Hysteresis", "Requires candidate emotion to lead by >0.08 across 2 updates (~3 frames) before switching."),
        ("ImageNet Norm", "Subtracting [0.485,0.456,0.406] and dividing by [0.229,0.224,0.225] RGB channels."),
        ("is_uncertain", "Boolean safety flag raised when p_top1 < 0.38 or margin (p1 - p2) < 0.08 [VERIFY]."),
        ("JPEG Quality 85", "WebRTC-matched compression factor used on 320x180 browser snaps & parity harness."),
        ("key_padding_mask", "Boolean Transformer mask ignoring missing audio/text/video modalities."),
        ("Label Smoothing", "Softening hard 1-hot targets (0.1 mixed / 0.05 fusion) to reduce overconfidence."),
        ("Laplacian Variance", "Sharpness metric flagging motion blur whenever Var(Laplacian(gray)) < 30.0."),
        ("Late / Hybrid Fusion", "Combining compact 16-D stream predictions/logits rather than raw raw pixels/waves."),
        ("LayerNorm", "Per-token feature normalization stabilizing Transformer and projection MLPs."),
        ("Logit", "Unnormalized linear class score prior to Softmax normalization."),
        ("Macro-F1", "Unweighted arithmetic mean of F1 across all 7 emotions; treats rare classes fairly."),
        ("MELD", "Multimodal EmotionLines Dataset (13,708 Friends utterances across A/T/V)."),
        ("MixedAugmenter", "Webcam realism transforms including compression, blur, sensor noise, and 30% glasses."),
        ("NLL", "Negative Log-Likelihood measuring probabilistic fit (improved 1.9835 -> 1.7244)."),
        ("NO_FACE", "Fallback label returned when YuNet sees zero faces, resetting EMA memory."),
        ("p50 / p95 Latency", "Median (50th) and tail (95th) percentile per-frame processing times (~33/35 ms)."),
        ("Parity (916/916)", "100% label match between live API frames (320x180 JPEG85) and disk clip eval."),
        ("RAF-DB", "Real-world Affective Faces Database (15,337 static web facial photos used)."),
        ("RAVDESS", "Ryerson speech video corpus; Actors 01-08 train (3,184) / 09-10 val (786 crops)."),
        ("ReLU / SiLU", "Rectified (conv nets) and Sigmoid Linear (EfficientNet head) nonlinearities."),
        ("ResNet-18", "11.18M-param residual CNN serving as both RAF baseline and RAVDESS default backbone."),
        ("ResNet-50", "2048-D frozen backbone pooling 8 uniform frames for uploaded MELD clips."),
        ("scaleX(-1)", "CSS mirror applied ONLY to preview <video> so raw pixel capture stays unmirrored."),
        ("StrongerAudioClassifier", "768->512->256->7 MLP (527,879 params) paired with balanced sampler."),
        ("T* (0.5953)", "L-BFGS learned temperature scalar cutting fusion ECE from 0.2130 to 0.0734."),
        ("TF-IDF", "Sparse 12k n-gram baseline text model (40.15% Acc vs 42.87% MiniLM)."),
        ("VideoClassifier", "2048->256->7 MLP (526,343 params) classifying pooled ResNet-50 video features."),
        ("Wav2Vec 2.0 Base", "Self-supervised speech Transformer producing 768-D acoustic embeddings."),
        ("WeightedRandomSampler", "Mini-batch sampler oversampling rare classes via inverse frequency."),
        ("WebSocket (/ws)", "Persistent bi-directional socket streaming frames with minimal protocol overhead."),
        ("YuNet", "75.8K-param OpenCV ONNX detector extracting bounding box plus 5 facial landmarks."),
    ]
    add_table(doc, "Table 11.3: Concise 65-Term Technical Glossary", ["Term", "One-Line Definition in This Project"], glossary, [1.70, 4.75], font_pt=8.2)

    add_h2(doc, "11.5 15 Rapid-Fire One-Line Viva Answers")
    rf = [
        ("How many emotions?", "7 shared classes: anger, disgust, fear, joy, neutral, sadness, surprise (calm in RAVDESS is dropped)."),
        ("Default webcam model?", "resnet18_ravdess (11.18M-param ResNet-18 fine-tuned on RAF-DB + RAVDESS @ 128 px)."),
        ("Best webcam frame accuracy?", "57.31% pooled (525/916 frames, 5/9 clips), up from 15.61% EffNet baseline."),
        ("CPU frame speed?", "~32.98 ms p50 (~30.3 FPS) and 34.59 ms p95 on Intel i7-12650H CPU only."),
        ("Face detector?", "OpenCV YuNet ONNX outputting bbox + 5 landmarks (eyes, nose, mouth corners)."),
        ("Why eye alignment?", "aligned_square_crop rotates eyes level around midpoint using BORDER_REPLICATE."),
        ("When does UNCERTAIN show?", "Whenever calibrated p_top1 < 0.38 or top-2 margin < 0.08 [VERIFY thresholds]."),
        ("Why flip CSS video only?", "Users see a mirror preview while the CNN infers on unmirrored pixels matching training data."),
        ("Number of RAVDESS crops?", "3,970 sharp middle 30-70% speech crops from 518 clips (3,184 Actors 01-08 / 786 Actors 09-10)."),
        ("Why freeze up to layer3?", "Preserves 24.9% general edge/texture weights & BN stats while adapting 8.40M layer4+fc weights."),
        ("Best MELD audio strategy?", "WeightedRandomSampler on StrongerAudioClassifier lifted Macro-F1 from 0.1771 to 0.2209."),
        ("Best MELD text model?", "all-MiniLM-L6-v2 + CalibratedClassifierCV: 42.87% Acc / 0.3326 Macro-F1 (N=2,610)."),
        ("MELD fusion ECE gain?", "Reduced from 0.2130 down to 0.0734 at T*=0.5953 on balanced N=105 test."),
        ("Hardest failure modes?", "Anger read as surprise (61.4%), sadness read as neutral (100%), and Fear.mp4 vs disgust."),
        ("Where do live frames go?", "Deleted immediately in volatile RAM after JSON response—zero disk upload unless /debug-capture called."),
    ]
    add_table(doc, "Table 11.4: 15 Rapid-Fire Examiner Drill Answers", ["Rapid Question", "Exact One-Line Spoken Answer"], rf, [1.85, 4.60], font_pt=8.4)


# ---------------------------------------------------------------------------
# SECTION 12: 3-DAY PRACTICE PLAN & SELF-TEST
# ---------------------------------------------------------------------------
def build_sec12(doc: Document) -> None:
    add_h1(doc, "12. 3-Day Practice Plan & 20-Question Mock Self-Test", page_break=True)

    plan_rows = [
        ("Day 1: Core Story & Pipeline Trace",
         "Sections 1, 2, 3 & Table 11.3 (Glossary)",
         "Speak the 60s pitch aloud 5 times; draw the 12 live-frame steps on blank paper from browser capture to CSS mirror overlay.",
         "Q1, Q3, Q7, Q8, Q12, Q13, Q16, Q21, Q35, Q36"),
        ("Day 2: Models, Metrics, MELD & Honest Numbers",
         "Sections 4, 5, 6 & Formula Sheet 11.2",
         "Practice explaining worked Macro-F1 vs Accuracy example aloud; trace 16-D MELD fusion vector & explain ECE 0.2130->0.0734 cleanly.",
         "Q23, Q27, Q30, Q41, Q49, Q52, Q55, Q60, Q62, Q64"),
        ("Day 3: Hostile Questions & Full Live Demo Run",
         "Sections 7, 8, 9g, 10 & Rapid-Fire 11.4",
         "Run 'python main.py --serve' twice end-to-end (including calibrate + uncertain + dia0_utt0 + 30s fallback); grill teammates on hostile Qs.",
         "Q71, Q72, Q74, Q75, Q79, Q80, Q83, Q84, Q86, Q95"),
    ]
    add_table(doc, "Table 12.1: Structured 3-Day Viva Preparation Schedule",
              ["Day", "What to Read", "What to Say / Drill Aloud", "Mock Questions to Quiz"], plan_rows, [1.30, 1.35, 2.25, 1.55], font_pt=8.4)

    add_h2(doc, "12.1 20-Question Randomized Mock Self-Test (Cover Answers While Drillinng!)")
    mock = [
        ("1", "Q35", "Why did we build _raf_index_map by class string name?", "Because RAF-DB puts Surprise at index 0 whereas our config puts anger at 0; index assumption scrambles labels."),
        ("2", "Q74", "Did you leak your 9 personal clips into training weights or selection?", "Never into training weights (Epoch 7 picked on RAF+RAVDESS val), but yes—we disclosed those 9 clips served as both backbone selection & test set."),
        ("3", "Q79", "What is YOUR team's original contribution beyond pretrained weights?", "Unified live/offline parity (916/916), unmirrored inference decoupling, AU-rule removal (+0.0996), RAVDESS prep & mixed trainer w/ glasses aug, balanced MELD audio, 16-D attention fusion w/ ECE 0.0734, FastAPI/UI & 16-test suite."),
        ("4", "Q40", "What condition makes a prediction UNCERTAIN?", "Calibrated p_top1 < 0.38 OR top-1 minus top-2 margin < 0.08 [VERIFY]."),
        ("5", "Q60", "Why do EffNet 96px and ResNet-18 128px run at virtually the same speed (~32-33 ms)?", "YuNet face detection, affine alignment, and Python image ops dominate fixed per-frame CPU overhead."),
        ("6", "Q15", "What padding mode prevents black corners after eye alignment?", "cv2.BORDER_REPLICATE in aligned_square_crop()."),
        ("7", "Q83", "Why did Text-only (37.14%) edge out Trimodal Acc (36.19%) on N=105?", "MELD TV video crops contain side profiles/cuts (6.67%), slightly diluting text Acc, while fusion slashes ECE from 0.1903 to 0.0734."),
        ("8", "Q19", "Why gentle gamma LUT instead of CLAHE inside WebcamService?", "Code notes CLAHE amplifies nasolabial shadows creating false Disgust; gamma adjusts global dark/bright exposure cleanly."),
        ("9", "Q51", "How much did Wav2Vec2 Macro-F1 improve with WeightedRandomSampler?", "From 0.1771 (49.23% neutral collapse) up to 0.2209 on MELD test (N=2,610)."),
        ("10", "Q64", "Explain why POSTER++ hits 92% on RAF-DB while our ResNet-18 gets 79.3%.", "POSTER++ uses heavy dual landmark-image Vision Transformers at GPU compute; our 128 px ResNet-18 optimizes 30 FPS laptop CPU + RAVDESS speech transfer."),
        ("11", "Q25", "Why set frozen Conv/BN layers to .eval() in train_mixed.py?", "Prevents small 6-sample RAVDESS batches from corrupting learned RAF-DB running mean and variance buffers."),
        ("12", "Q42", "How are batch samples divided in mixed training?", "Batch 32, extra_share=0.2 -> 6 balanced RAVDESS crops + 26 balanced RAF-DB crops per step."),
        ("13", "Q75", "Why does Anger fail completely (0%) on your Anger.mp4 clip?", "Furrowed brows/tense eyes look like Surprise (61.4%) or Neutral (38.6%) without temporal context or acoustic cues."),
        ("14", "Q55", "How does fusion gracefully handle a missing modality?", "Boolean key_padding_mask=True disables absent tokens inside Transformer attention and zeros their normalized gate."),
        ("15", "Q12", "What 15 numbers does YuNet output per detected face?", "[x, y, w, h] box, 5 (x,y) landmarks (right/left eye, nose, right/left mouth), and confidence score."),
        ("16", "Q29", "What parameter count changed when freezing through layer3?", "8,397,319 params (75.1% in layer4+fc) remained trainable out of 11,180,103 total; 2,782,784 frozen."),
        ("17", "Q71", "Wait—isn't MELD 62-67% in papers vs your 36.2%? Explain!", "Papers report imbalanced full 2,610 test (47% neutral skew) using whole conversation history; ours evaluates single isolated utterances on balanced 105-sample test (15/class, chance 14.3%)."),
        ("18", "Q10", "How did you verify 100% live API vs offline clip parity?", "test_parity_harness.py & eval_clips.py confirmed 916/916 exact predictions across 320x180 JPEG85."),
        ("19", "Q84", "How was ui_live_webcam.png captured if no camera was attached?", "In headless Chrome STANDBY mode ('Camera: Inactive'); real frame metrics ran on recorded New folder/*.mp4 clips."),
        ("20", "Q100", "What command runs all unit and live integration checks before viva?", ".\\.venv\\Scripts\\python.exe main.py --test (or tests/run_tests.py --all) -> 5/5 unit + 11/11 API PASS."),
    ]
    add_table(doc, "Table 12.2: 20-Question Randomized Mock Viva Key", ["#", "Bank Ref", "Question Prompt", "Concise Verification Answer"], mock, [0.35, 0.55, 2.35, 3.20], font_pt=8.0)


def main() -> None:
    doc = Document()
    configure_a4_layout(doc)
    setup_styles(doc)
    build_prelims(doc)
    build_sec1(doc)
    build_sec2(doc)
    build_sec3(doc)
    build_sec4(doc)
    build_sec5(doc)
    build_sec6(doc)
    build_sec7(doc)
    build_sec8(doc)
    build_sec9(doc)
    build_sec10(doc)
    build_sec11(doc)
    build_sec12(doc)

    out_docx = REPORT_DIR / "Viva_Guide.docx"
    doc.save(str(out_docx))
    print(f"Wrote {out_docx} ({out_docx.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
