# Webcam Fix Log (Sessions 1, 1b, 5)

## Done
- **Session 1:** Unified `/frame`, `/frame-json`, `/ws`, `/ws/webcam`, `/multimodal`, `benchmark_webcam.py` on `WebcamService.process_frame` (100% parity on 320x180 JPEG85, 916/916 frames); fixed WS disconnect; fixed mirrored overlay coords; removed AU geometry rules (MELD val Macro-F1 **0.2872 -> 0.3868**).
- **Session 1b (ResNet-18 RAF Unification):** Removed duplicate `aligned_square_crop` and hardcoded `224`/normalize loader in `backend/services/webcam_service.py`; now reuses `live_resnet18_demo.py` (`aligned_square_crop`, `load_model`) reading checkpoint `preprocessing` (`eval_resize=146`, `image_size=128`) and name-mapping RAF classes (`[6, 2, 1, 3, 5, 0, 4]`). Ran `eval_clips.py --backbone resnet18_raf`.
- **Frontend & API Backbone UI:** Added `/predict/webcam/backbone` (`POST`) + backbone metadata on `/predict/webcam/status` (`GET`); added `#webcam-backbone-select`, `#webcam-model-pill`, accurate description/T* label, and `/25` calibration sync in `frontend/index.html` and `frontend/js/app.js`.
- **Session 5 (RAVDESS + Mixed Fine-Tuning Scripts):** Updated `.gitignore` (`Video speech dataset/`, `ravdess_crops/`, `DATASET/`, `external_models/`); added `prepare_ravdess.py` (modality `01` speech only, drop `02`/calm, actors `01-08` train / `09-10` val, 30%-70% @ ~5 fps, reuses `live_resnet18_demo.aligned_square_crop` + YuNet, blur/size gate) and `train_mixed.py` (checkpoint-driven preprocessing, exact seed-42 RAF split, 80/20 class-balanced mixed batching, webcam/glasses augmentations, AdamW, patience-3 early stopping on avg val macro-F1, resume, `--dry-run`, `best_resnet18_ravdess.pt`, <=30-line `train_summary.json`). Smoke tests passed (`py_compile`, filename parsing, name-mapped `--init`, dry-run, 1-epoch train/save/reload, `--resume`).

## Key Numbers
- **RAF-DB Test (`external_models/resnet18_raf/test_metrics.json`):** accuracy **0.7953**, macro-F1 **0.6721**.
- **`eval_clips.py` Totals (9 clips, 916 frames, native, smoothing=True):**
  - `efficientnet_current`: majority **1/9**, pooled acc **15.6%** (143/916), surp-vs-ang **0**, neu-vs-surp **23** (23 neu on surp).
  - `resnet18_raf`: majority **4/9**, pooled acc **44.0%** (403/916), surp-vs-ang **101** (101 surp on ang), neu-vs-surp **82** (45 neu on surp, 37 surp on neu). CPU latency: **p50=102.33 ms**, **p95=110.55 ms**.
- **Per Clip (`Truth`: `Eff` vs `R18`):** `Anger.mp4` (`anger`: neu 0.0% vs surp 0.0%); `Disgust (2).mp4` (`disgust`: neu 0.0% vs disg 78.5%); `disgust.mp4` (`disgust`: neu 0.0% vs disg 67.4%); `Fear (2).mp4` (`fear`: disg 0.0% vs surp 0.0%); `Fear.mp4` (`fear`: joy 0.0% vs surp 0.0%); `Joy.mp4` (`joy`: neu 38.3% vs joy 93.0%); `Neutal.mp4` (`neutral`: neu 100.0% vs neu 62.6%); `sadness.mp4` (`sadness`: neu 0.0% vs 0.0%); `surprise.mp4` (`surprise`: disg 0.0% vs neu 46.4%).
- **Tests (`tests/test_webcam.py` + `tests/run_tests.py --all`):** 21/21 webcam, 5/5 unit, 11/11 integration passed.

## Decisions
- Kept `webcam_backbone: "efficientnet_current"` as default in `config.yaml`; Session 5 outputs `external_models/ravdess/best_resnet18_ravdess.pt` (never overwriting `best_resnet18.pt`).

## Next Step
- User runs `prepare_ravdess.py`, `--dry-run`, and `train_mixed.py` (local or Colab), then evaluates `best_resnet18_ravdess.pt`.
