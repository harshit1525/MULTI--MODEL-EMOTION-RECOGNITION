# Report Revision Discrepancies & Viva Notes Outside Edited Scope

During the final revision pass on the B.Tech Major Project report (`Major.pdf` / [`report/build_report.py`](file:///c:/Users/harsh/Desktop/Major%20project/report/build_report.py)), `Abstract`, `Table 0.1`, Chapters `1–6`, and the Chapter `3`/`5` figure/table assets were updated in accordance with [`report_revision/PLAN.md`](file:///c:/Users/harsh/Desktop/Major%20project/report_revision/PLAN.md). Sections explicitly kept untouched by rule (`Certificate`, `Declaration of Originality`, `Acknowledgement`, `1.6.1–1.6.3` / role text, `4.2 Web Studio Walkthrough` screenshots, `References`, `Appendices A–B`, `Annexures A–D`) and remaining CSV/static diagram/codebase discrepancies are logged below without altering out-of-scope files.

---

## 1. Source Provenance & Build Lock Issues

1. **Deliverables & `report/Project_Report.docx` OS File Lock:**
   - The final editable Word report and matching Word-exported PDF were generated in [`report_revision/output/Major_Report_Revised.docx`](file:///c:/Users/harsh/Desktop/Major%20project/report_revision/output/Major_Report_Revised.docx) and [`report_revision/output/Major_Report_Revised.pdf`](file:///c:/Users/harsh/Desktop/Major%20project/report_revision/output/Major_Report_Revised.pdf) (**78 PDF pages**, `3,215,550` bytes; prelims cover + `i–xiv`, body Chapters `1–6` pp. `1–55`, References/Appendices/Annexures pp. `56–63`), with synced copies at [`report/Project_Report_Revised.docx`](file:///c:/Users/harsh/Desktop/Major%20project/report/Project_Report_Revised.docx) and [`report/Project_Report_Revised.pdf`](file:///c:/Users/harsh/Desktop/Major%20project/report/Project_Report_Revised.pdf).
   - An external `WINWORD.EXE` process still holds a write lock on `report/Project_Report.docx` (`report/~$oject_Report.docx`). Close Microsoft Word if you want `build_report.py` to overwrite `report/Project_Report.docx` directly; `Major.pdf` was never overwritten.
2. **Institutional Placeholders (`[TO BE FILLED: ...]`):**
   - Because `Certificate`, `Declaration of Originality`, `Acknowledgement`, `1.6.3`, `Table 5.10` hardware cost cells, and `Annexures A–D` were preserved, fill your student names, enrollment numbers, supervisor/HOD names, institute details, JIIT/department PSOs, and INR hardware costs before final printing.
3. **`Major.pdf` vs. `report/build_report.py` Drift:**
   - The uploaded `Major.pdf` (54 PDF pages) was a manually Word-edited derivative whose `.docx` source is not committed in Git. `report/build_report.py` is the reproducible source of truth (`report/Project_Report_Revised.pdf`).

---

## 2. Out-of-Scope & Frozen Section Discrepancies

1. **Appendix A Stale Asset Counts & Repeated Parity Metric (`build_report.py:1761-1778`, untouched):**
   - Frozen `Appendix A` still states `"20 high-signal PNG figures (200 dpi), 10 curated CSV tables, and all 22 mathtext formula PNGs"` and states `"916/916 identical emotion predictions"` in prose.
   - After the final pass, `build_report.py` actually embeds **19 PNG figures**, **8 CSV tables + 16 styled tables** (`Table 0.1`, `2.1`, `3.1`, `3.2`, `3.3`, `4.2`, `5.4–5.9`, `5.11`, `A.1–A.3`), and **25 mathtext formula PNGs (`F1`–`F25`)**. Update those Appendix A bullet counts (and optionally cross-reference Section `5.1` instead of repeating `916/916`) if you unlock Appendix A before printing.
   - Similarly, [`report/assets_manifest.md`](file:///c:/Users/harsh/Desktop/Major%20project/report/assets_manifest.md) does not yet list formulas `F23`–`F25`.
2. **Chapter 4.2 `Figure 4.4` Static Web Studio Screenshot (`frontend/`):**
   - `Figure 4.4` (`ui_results_benchmarks.png`, untouched inside `4.2 Web Studio Walkthrough`) shows legacy/mock frontend numbers (`82.86%`, `Unimodal Video 85.71%`) rather than the verified Chapter 5 evaluation numbers (`Trimodal 36.19%`, `Video Only 6.67%` on `data/fusion_test_dataset.pt`). Be ready to note in viva that the frontend Benchmarks tab screenshot displayed an older cached benchmark JSON (`results.json`), whereas Chapter 5 reports the verified re-evaluated results.
3. **Frontend Tab vs. Pane Count (`Abstract` / `4.2` / `frontend/index.html`):**
   - [`frontend/index.html`](file:///c:/Users/harsh/Desktop/Major%20project/frontend/index.html) defines **7 `<section class="tab-pane">` panels** (`tab-multimodal`, `tab-audio`, `tab-video`, `tab-text`, `tab-webcam`, `tab-research`, `tab-history`) addressed by **8 `<button class="nav-tab">` sidebar links** (both `#nav-dashboard` and `#nav-multimodal` point to `data-tab="tab-multimodal"`). Chapter `4.2` refers to the `"eight primary operational interfaces"`.

---

## 3. Static Diagrams, CSV Tables, and Codebase Divergences

1. **Frozen Static Diagrams (`report/figures/`):**
   - [`report/figures/fig_diag_system_architecture.png`](file:///c:/Users/harsh/Desktop/Major%20project/report/figures/fig_diag_system_architecture.png) (Figure `3.1`) still contains `"8-Frame CLAM + Crop"` instead of `"CLAHE"`, labels text `"Calib Clf"`, and prints `Temp Scaling (T*=0.28/0.60)` and `"7-Tab Web Studio"`.
   - [`report/figures/fig_diag_ravdess_prep.png`](file:///c:/Users/harsh/Desktop/Major%20project/report/figures/fig_diag_ravdess_prep.png) (Figure `3.3`) Box 2 shows `520 kept` (filename-eligible AV speech clips before 2 dropped in quality filtering) and Box 7 correctly shows `3,970 JPG Crops` (`Clips Kept: 518 / 1,200`).
   - [`report/figures/fig_diag_meld_fusion.png`](file:///c:/Users/harsh/Desktop/Major%20project/report/figures/fig_diag_meld_fusion.png) (Figure `3.5`) displays runtime `Adaptive Temp T* in [0.24, 0.28]` rather than validation-fitted `T* = 0.5953`.
   - [`report/figures/fig_cm_webcam_resnet18_ravdess.png`](file:///c:/Users/harsh/Desktop/Major%20project/report/figures/fig_cm_webcam_resnet18_ravdess.png) (Figure `5.1`) rounds `62/101` Anger->Surprise to `61.4%`, while `Section 5.4` text prints `61.39%` (2 decimals of the same ratio).
2. **Stale CSV Files in `report/tables/` (Bypassed via Inline Styled Tables in `build_report.py`):**
   - [`report/tables/meld_model_results.csv`](file:///c:/Users/harsh/Desktop/Major%20project/report/tables/meld_model_results.csv) still lists `Text Contextual (Production)` as `all-MiniLM-L6-v2 + CalibratedClassifierCV` (actual saved checkpoint is `LogisticRegression(C=1.2, class_weight='balanced')`) and conflates `Fusion Baseline` (`0.3048 / 0.2488` from `best_fusion_model.pth` 14-D test slice) with `0.0713` stored validation ECE. Chapter 5 (`Tables 5.4–5.5`) overrides this CSV cleanly with direct verified numbers.
   - [`report/tables/hyperparameters.csv`](file:///c:/Users/harsh/Desktop/Major%20project/report/tables/hyperparameters.csv) has only 4 rows; `Table 4.2` was expanded inline to cover all 8 models.
   - [`report/tables/test_cases.csv`](file:///c:/Users/harsh/Desktop/Major%20project/report/tables/test_cases.csv) (`Unit 04`) reflects [`tests/test_fusion.py`](file:///c:/Users/harsh/Desktop/Major%20project/tests/test_fusion.py), which unit-tests the legacy `LearnedAttentionFusionNet` (14-D) rather than the production `AdvancedCrossModalAttentionFusionNet` (16-D).
3. **Minor Table Wrapping in Frozen / Narrow Columns:**
   - In `Table 1.1` (`Model Generalization`) and `Table 4.3` (`/predict/webcam/backbone`, `MultimodalPredictionResponse`), narrow column widths cause single-character mid-word wraps (`Generalizatio-n`, `backbon-e`).
4. **Codebase vs. Checkpoints / Config Divergences:**
   - [`backend/services/transcription_service.py#L21-L50`](file:///c:/Users/harsh/Desktop/Major%20project/backend/services/transcription_service.py#L21-L50): indexes MELD `dev` (`1,109`), `test` (`2,610`), and `train` (`9,989`) (`13,708` rows) by `dia{Dialogue_ID}_utt{Utterance_ID}`, retaining `11,132` unique keys due to cross-split ID collisions.
   - [`train.py`](file:///c:/Users/harsh/Desktop/Major%20project/train.py): `train_text` specifies `CalibratedClassifierCV(C=1.0, cv=3)` on 80 samples/class, whereas saved `best_text_model_contextual.pt` is `LogisticRegression(C=1.2, class_weight='balanced')`; `train_video` (`train.py#L172-L184`) is an unimplemented stub (`best_video_model.pth` training script is missing).
   - [`backend/services/fusion_service.py`](file:///c:/Users/harsh/Desktop/Major%20project/backend/services/fusion_service.py): overrides offline `T* = 0.5953` with `base_temperature = 0.28` (`T_adaptive in [0.24, 0.28]`).
   - [`backend/main.py`](file:///c:/Users/harsh/Desktop/Major%20project/backend/main.py) `/health` checks legacy checkpoint filenames; [`config.yaml`](file:///c:/Users/harsh/Desktop/Major%20project/config.yaml) still specifies `LearnedAttentionFusionNet` (`num_heads: 2`); [`backend/services/webcam_service.py`](file:///c:/Users/harsh/Desktop/Major%20project/backend/services/webcam_service.py) docstring still references Action Unit geometry rules and `T*=0.85` globally despite AU removal and ResNet-18 `T=1.0`.
