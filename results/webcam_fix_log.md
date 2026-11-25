# Webcam Fix Log (Sessions 1, 1b, 5, Task B)

## Done
- **Session 1:** Unified `/frame`, `/frame-json`, `/ws`, `/ws/webcam`, `/multimodal`, `benchmark_webcam.py` on `WebcamService.process_frame` (100% parity on 320x180 JPEG85, 916/916 frames); removed AU rules (MELD val Macro-F1 **0.2872 -> 0.3868**).
- **Session 1b + Frontend:** Unified ResNet-18 loader/crop in `WebcamService` via `live_resnet18_demo.py`; added `/predict/webcam/backbone`, `/predict/webcam/status` metadata, and `#webcam-backbone-select` UI.
- **Session 5 + Task A:** Added `prepare_ravdess.py` & `train_mixed.py`; extracted 3,970 RAVDESS crops (518 clips kept, 682 skipped); fine-tuned 8 epochs (`--freeze-until layer3`, best Epoch 7: RAF val `0.8749`, RAVDESS val `0.5354`, avg `0.7051`, train-val gap `raf=-0.0257`, `ravdess=0.2205`).
- **Task B:** Added `resnet18_ravdess` backbone (`external_models/ravdess/best_resnet18_ravdess.pt`, fallback to `resnet18_raf`) in `config.yaml`, `WebcamService`, `webcam.py`, UI, `eval_clips.py`, `tests/test_webcam.py`.

## Key Numbers
- **RAF-DB Test (`best_resnet18.pt` -> `best_resnet18_ravdess.pt`):** acc **0.7953 -> 0.7927**, macro-F1 **0.6721 -> 0.7077**.
- **`eval_clips.py` (9 clips, 916 frames, native, smoothing=True):**
  - `resnet18_raf`: maj **4/9**, pooled **44.0%** (`403/916`), surp-vs-ang **101**, neu-vs-surp **82** (`45/37`), p50/p95 **102.33 / 110.55 ms**.
  - `resnet18_ravdess`: maj **5/9**, pooled **57.3%** (`525/916`), surp-vs-ang **62**, neu-vs-surp **51** (`51/0`), p50/p95 **33.00 / 38.89 ms**.
- **Per Clip (`R18_RAF` vs `R18_RAVDESS`):** `Anger.mp4` (`surp 0.0%` vs `surp 0.0%`); `Disgust (2).mp4` (`disg 78.5%` vs `disg 100.0%`); `disgust.mp4` (`disg 67.4%` vs `disg 79.8%`); `Fear (2).mp4` (`surp 0.0%` vs `fear 43.8%`); `Fear.mp4` (`surp 0.0%` vs `disg 0.0%`); `Joy.mp4` (`joy 93.0%` vs `joy 100.0%`); `Neutal.mp4` (`neu 62.6%` vs `neu 100.0%`); `sadness.mp4` (`neu 0.0%` vs `neu 0.0%`); `surprise.mp4` (`neu 46.4%` vs `neu 39.3%`).

## Decisions
- Promoted **`resnet18_ravdess`** to default in `config.yaml` (all criteria a-f met: maj `5/9 > 4/9`, pooled `57.3% > 44.0%`, RAF-DB F1 `+0.0356`, no prior-correct clip dropped >10 pts, p95 `38.89 ms <= 150 ms`).

## Next Step
- Run live webcam demo or evaluate reserved ` (3)` clips (`eval_clips.py --backbone resnet18_ravdess --final`) when ready.
