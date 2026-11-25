# Cleanup Log (Pre-cleanup tag: `pre-cleanup`, commit `73b66ba`)

## Moved to `_cleanup_quarantine\` (11 items, 80,630,891 B / 76.90 MiB)
- `models/video_classifier/face_crops_cache/` (80,034,124 B / 76.33 MiB)
- `train_raf.err` (493,926 B), `uvicorn_err.log` (0 B), `uvicorn_out.log` (0 B)
- `evaluate_all_7_clips.py` (18,323 B), `evaluate_all_9_clips.py` (22,123 B), `evaluate_candidates.py` (19,593 B)
- `eval_all_candidates.py` (8,989 B), `eval_user_face.py` (724 B), `run_phase2_ablations.py` (8,285 B), `scratch/` (24,804 B)

## Deleted Directly (9 `__pycache__` dirs, 752,495 B / 0.72 MiB)
- `__pycache__/`, `backend/**/__pycache__/` (6 dirs), `Project_Report/__pycache__/`, `tests/__pycache__/`

## Checks Passed (all after cleanup)
- Module imports (`webcam_service`, `fusion_service`, `eval_clips`, `main`, `prepare_ravdess`, `train_mixed`, `live_resnet18_demo`): PASS
- `config.yaml` referenced paths: PASS (0 missing)
- `main.py --serve` `/health` (`healthy`) + `/predict/webcam/status` (`resnet18_ravdess`, `128px`): PASS
- `tests/test_webcam.py` and `tests/run_tests.py --all` (5/5 unit, 11/11 integration): PASS
