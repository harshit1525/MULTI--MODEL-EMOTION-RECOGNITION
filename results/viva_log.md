# TASK: VIVA GUIDE — FINAL STATUS: COMPLETE

- Script: report/build_viva_guide.py
- Outputs: report/Viva_Guide.docx (3,021,201 bytes, TOC updated) & report/Viva_Guide.pdf (2,265,845 bytes, 49 pages verified)
- Sections Done: Prelims + Sections 1-12 (60s/5m scripts, Code Map, Live Webcam Flow, Models, Datasets, Results, Decisions, Honesty Table, 100-Q Bank, Live Demo Script, Cheat Sheets + 22 Formulas + 65 Glossary Terms + 15 Rapid-Fire, 3-Day Plan + 20-Q Self-Test)
- Question Bank: 100 questions (Q1-Q100 across groups a-i with Difficulty & Likelihood ratings)
- Checks: 49 pages, TOC_POPULATED=True, 36 inline images (diagrams/charts/UI/22 formulas), 0 TODO/TBD/lorem/FIXME

## Items Marked [VERIFY] (Confirmed Open for Student Verification)
- [VERIFY] Why only Actors 01-10 (vs full 24) exist in Video speech dataset/ (Sec 5, Sec 7)
- [VERIFY] Empirical sweep rationale for is_uncertain cutoffs (p_top1<0.38, margin<0.08) & hysteresis margin 0.08 (Sec 3, Sec 7, Q39)
- [VERIFY] Parameter sweep rationale for extra_share=0.2, lr=2e-5/2e-4, image_size=128/eval_resize=146 (Sec 4.3, Q18, Q30, Sec 7)
- [VERIFY] Exact dataset origin/composition inside deleted models/video_classifier/face_crops_cache (Sec 4.2)
- [VERIFY] Choice rationale for live runtime T_adaptive=0.24-0.28 in fusion_service.py vs offline T*=0.5953 (Sec 4.6, Q58)
- [VERIFY] FastAPI vs Flask framework selection rationale (not documented in README/logs) (Sec 7)
- [TO BE FILLED] Member 1/2/3 names on Cover & Table 1.1 team role scripts

## Rebuild Command
- .\.venv\Scripts\python.exe report\build_viva_guide.py
