# Webcam Fix Log (Session 1)

## Done
- **Unified Engine & Parity:** `/frame`, `/frame-json`, `/ws`, `/ws/webcam`, `/multimodal`, and `benchmark_webcam.py` all call `WebcamService.process_frame`. Fixed WS clean disconnect bug in `backend/api/webcam.py`.
- **Browser Encoding vs Native Gap:** `frontend/js/app.js` captures `320x180` JPEG `q=85` (unmirrored) with per-tab `clientId`. Endpoint parity on `320x180` JPEG `q=85` is **100.00% (916/916)** across `/frame-json`, `/frame` (100% on Anger), `/ws/webcam` (100% on Anger). Native `1280x720` vs `320x180` agrees 0.0–88.8% (`Neutal.mp4`: `1280x720` BGR = 99/99 NEUTRAL, `1280x720` JPEG95 = 87 NEUTRAL / 12 JOY, `640x360` JPEG85 = 99 DISGUST, `320x180` JPEG85 = 93 DISGUST / 6 JOY) because EfficientNet-B0 (96x96) is brittle to downscaling + JPEG artifacts.
- **Mirror Display:** `#webcam-video` mirrored via CSS `scaleX(-1)`; `#webcam-canvas` unmirrored; JS flips `bx` and landmark `px` and renders label left-to-right.
- **Hardcoded Rules:** Audited and removed AU geometry hacks (`prior_bias = np.zeros(7)`, no class biases in live path). MELD val Macro-F1 improved **0.2872 -> 0.3868** (+0.0996).
- **ResNet-18 RAF Backbone:** Added `webcam_backbone` (`efficientnet_current` default | `resnet18_raf`), `webcam_resnet18_raf_path` in `config.yaml`, `aligned_square_crop`, and name-based mapping `RAF_TO_CANONICAL`. `live_resnet18_demo.py` and `external_models/resnet18_raf/best_resnet18.pt` missing; tested with synthetic `.pt` checkpoints.
- **Clip Eval (`eval_clips.py`) & Gitignore:** Added `personal_data/`, `personal_crops/`, `external_models/` to `.gitignore`. Ran `eval_clips.py --backbone efficientnet_current` (`results/eval_efficientnet_current.json`); `resnet18_raf` skipped (missing checkpoint).

## Key Numbers
- `eval_clips.py` (`efficientnet_current`, native 1280x720, smoothing=True): `Neutal.mp4` 100.0% neutral; `Joy.mp4` 38.3% joy (top wrong neutral 61.7%); `Anger/disgust/Disgust(2)/Fear/Fear(2)/sadness/surprise` 0.0% frame acc.
- Tests (`tests/test_webcam.py` + `tests/run_tests.py --all`): 19/19 webcam unit checks, 5/5 unit suites, 11/11 integration tests passed.

## Decisions
- Kept `efficientnet_current` as default (`webcam_backbone: efficientnet_current`) until `external_models/resnet18_raf/best_resnet18.pt` is supplied.
- Kept native resolution for `eval_clips.py` and logged browser `320x180` downscale gap explicitly rather than forcing 720p over HTTP/WS.

## Next Step
- Supply `external_models/resnet18_raf/best_resnet18.pt` (and `live_resnet18_demo.py` if custom preprocessing is needed), then run `eval_clips.py --backbone resnet18_raf` and begin Session 2.
