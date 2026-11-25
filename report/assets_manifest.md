# Report Assets Manifest

| ID | File | Caption | Source |
| :--- | :--- | :--- | :--- |
| FIG-01 | report/figures/fig_cm_webcam_efficientnet_current.png | 7x7 row-normalized confusion matrix for efficientnet_current on 9 evaluation clips (916 frames). | results/eval_efficientnet_current.json |
| FIG-02 | report/figures/fig_cm_webcam_resnet18_raf.png | 7x7 row-normalized confusion matrix for resnet18_raf on 9 evaluation clips (916 frames). | results/eval_resnet18_raf.json |
| FIG-03 | report/figures/fig_cm_webcam_resnet18_ravdess.png | 7x7 row-normalized confusion matrix for resnet18_ravdess on 9 evaluation clips (916 frames). | results/eval_resnet18_ravdess.json |
| FIG-04 | report/figures/fig_cm_rafdb_efficientnet_current.png | 7x7 confusion matrix for EfficientNet-B0 on RAF-DB test (N=3,068). | results/webcam_rafdb_reference.json |
| FIG-05 | report/figures/fig_cm_rafdb_resnet18_raf.png | 7x7 confusion matrix for ResNet-18 (RAF-DB) on RAF-DB test (N=3,068). | external_models/resnet18_raf/test_confusion_matrix.csv |
| FIG-06 | report/figures/fig_cm_rafdb_resnet18_ravdess.png | 7x7 confusion matrix for ResNet-18 (RAF+RAVDESS) on RAF-DB test (N=3,068). | external_models/ravdess/best_resnet18_ravdess.pt + DATASET/test |
| FIG-07 | report/figures/fig_cm_ravdess_val_resnet18_ravdess.png | 7x7 confusion matrix for ResNet-18 (RAF+RAVDESS) on RAVDESS validation (N=786, Actors 09-10). | external_models/ravdess/best_resnet18_ravdess.pt + ravdess_crops/val |
| FIG-08 | report/figures/fig_cm_meld_audio_balanced.png | 7x7 confusion matrix: MELD Test — Audio (Wav2Vec2 + Balanced StrongerMLP) (N=2610). | models/audio_classifier/best_audio_model_balanced.pth + audio_embeddings/test_embeddings.pt |
| FIG-09 | report/figures/fig_cm_meld_audio_baseline.png | 7x7 confusion matrix: MELD Test — Audio Baseline (Wav2Vec2 + MLP) (N=2610). | models/audio_classifier/best_audio_model.pth + audio_embeddings/test_embeddings.pt |
| FIG-10 | report/figures/fig_cm_meld_text_contextual.png | 7x7 confusion matrix: MELD Test — Text (MiniLM + Calibrated Clf) (N=2610). | models/text_classifier/best_text_model_contextual.pt + MELD.Raw/test_sent_emo.csv |
| FIG-11 | report/figures/fig_cm_meld_text_tfidf.png | 7x7 confusion matrix: MELD Test — Text Baseline (TF-IDF + LogReg) (N=2610). | models/text_classifier/best_text_model.joblib + MELD.Raw/test_sent_emo.csv |
| FIG-12 | report/figures/fig_cm_meld_video_fusion16d.png | 7x7 confusion matrix: MELD Fusion Test — Unimodal Video Stream (16-D) (N=105). | data/fusion_test_dataset.pt |
| FIG-13 | report/figures/fig_cm_meld_fusion_baseline.png | 7x7 confusion matrix: MELD Fusion Test — LearnedAttentionFusionNet (N=105). | models/fusion_classifier/best_fusion_model.pth + data/fusion_test_dataset.pt |
| FIG-14 | report/figures/fig_cm_meld_fusion_optimized.png | 7x7 confusion matrix: MELD Fusion Test — AdvancedCrossModalAttentionNet (N=105). | models/fusion_classifier/optimized_fusion_model.pth + data/fusion_test_dataset.pt |
| FIG-15 | report/figures/fig_train_curves_raf.png | Training and validation loss, accuracy, and macro-F1 per epoch for ResNet-18 on RAF-DB (best epoch 6). | external_models/resnet18_raf/training.log + training_history.csv |
| FIG-16 | report/figures/fig_train_curves_ravdess.png | Mixed RAF-DB + RAVDESS training loss and validation macro-F1 curves (best epoch 7). | results/train_ravdess.log |
| FIG-17 | report/figures/fig_train_curves_meld_audio.png | MELD Audio baseline training loss, validation accuracy, and validation macro-F1 over 26 epochs. | results/metrics/training_history.json |
| FIG-18 | report/figures/fig_f1_per_class_rafdb.png | Per-class F1 scores on RAF-DB test set (N=3,068) across the three webcam backbones. | results/webcam_rafdb_reference.json + external_models/resnet18_raf/test_classification_report.csv + external_models/ravdess/best_resnet18_ravdess.pt |
| FIG-19 | report/figures/fig_f1_per_class_ravdess_val.png | Per-class F1 scores on RAVDESS validation split (N=786) for ResNet-18 (RAF+RAVDESS). | external_models/ravdess/train_summary.json + ravdess_crops/val |
| FIG-20 | report/figures/fig_f1_per_class_meld.png | Per-class F1 scores across MELD audio, text, video, and optimized fusion models. | audio_embeddings/test_embeddings.pt + MELD.Raw/test_sent_emo.csv + data/fusion_test_dataset.pt |
| FIG-21 | report/figures/fig_webcam_backbone_comparison.png | Per-clip frame accuracy, pooled accuracy, and majority-vote clips correct across 3 webcam backbones (9 clips, 916 frames). | results/eval_efficientnet_current.json + eval_resnet18_raf.json + eval_resnet18_ravdess.json |
| FIG-22 | report/figures/fig_class_dist_datasets.png | Class distribution across splits for RAF-DB, RAVDESS crops, and MELD. | DATASET/ + ravdess_crops/ + MELD.Raw/ |
| FIG-23 | report/figures/fig_latency_comparison.png | Unified 200-frame CPU p50/p95 latency across the 3 webcam backbones (earlier single-run numbers are not comparable). | New folder/Disgust (2).mp4 + WebcamService |
| FIG-24 | report/figures/fig_ablation_meld_7config.png | 7-configuration MELD multimodal ablation benchmark showing accuracy, macro-F1, and ECE. | results/full_ablation_benchmark_results.json |
| FIG-25 | report/figures/fig_ablation_audio_improvement.png | Comparison of MELD audio classifier imbalance strategies across accuracy, macro-F1, and weighted-F1. | results/audio_classifier_improvement/comparison.csv |
| FIG-26 | report/figures/fig_ablation_fusion_optimization.png | Baseline vs. optimized multimodal fusion system across accuracy, macro-F1, ECE, Brier score, and NLL. | results/metrics/multimodal_optimization_comparison.json |
| FIG-27 | report/figures/fig_calibration_reliability.png | Reliability diagram comparing uncalibrated vs. temperature-scaled multimodal probabilities. | results/metrics/multimodal_calibration_report.json |
| FIG-28 | report/figures/fig_ablation_phase2_au_rules.png | Phase 2 ablation showing MELD validation macro-F1 improvement from 0.2872 to 0.3868 upon removing hard-coded AU geometry rules. | results/phase2_ablations.json |
| FIG-29 | report/figures/fig_gantt_git_timeline.png | Major Project Work Breakdown Timeline (Gantt chart across relative semester weeks without calendar dates). | git log sequence + WBS phases |
| FIG-30 | report/figures/fig_diag_system_architecture.png | Overall system architecture integrating MELD multimodal pipelines, WebcamService, FastAPI, and Web Studio UI. | backend/main.py + backend/services/*.py + frontend/index.html |
| FIG-31 | report/figures/fig_diag_live_inference.png | Live webcam inference pipeline from frame capture through YuNet, aligned crop, quality gate, backbone, EMA, hysteresis, and overlay. | backend/services/webcam_service.py + live_resnet18_demo.py |
| FIG-32 | report/figures/fig_diag_training_pipeline.png | Mixed RAF-DB and RAVDESS ResNet-18 fine-tuning pipeline (extra_share=0.2, freeze-until layer3, best epoch 7). | train_mixed.py + results/train_ravdess.log |
| FIG-33 | report/figures/fig_diag_ravdess_prep.png | RAVDESS speech video preparation flow (modality=01, channel=01, drop calm, middle 30-70% at 5 fps, quality gate). | prepare_ravdess.py + results/prepare.log |
| FIG-34 | report/figures/fig_diag_meld_fusion.png | MELD AdvancedCrossModalAttentionFusionNet and MultimodalFusionService entropy gating and temperature calibration. | backend/models/fusion_model.py + backend/services/fusion_service.py |
| FIG-35 | report/figures/fig_diag_api_deployment.png | FastAPI deployment and REST/WebSocket endpoint architecture on 127.0.0.1:8000. | backend/main.py + backend/api/*.py |
| FIG-36 | report/figures/fig_diag_wbs.png | Project Work Breakdown Structure (WBS) across six technical work packages. | README.md + git log + project modules |
| FIG-37 | report/figures/ui_home_landing.png | Web Studio landing view (Unified Multimodal Dashboard) at 1280x800. | frontend/index.html + main.py --serve |
| FIG-38 | report/figures/ui_multimodal_unified.png | Unified Multimodal Emotion Analysis results for MELD dia0_utt0 preset showing attention weights and diagnostics. | frontend/index.html + /predict/multimodal |
| FIG-39 | report/figures/ui_audio_studio.png | Acoustic Modality Studio (Wav2Vec2) prediction view on dia0_utt0.wav. | frontend/index.html + /predict/audio |
| FIG-40 | report/figures/ui_upload_clip_video.png | Video Upload Clip Studio (ResNet-50 8-frame temporal pooling) prediction view on dia0_utt0.mp4. | frontend/index.html + /predict/video |
| FIG-41 | report/figures/ui_text_studio.png | Linguistic Modality Studio (all-MiniLM-L6-v2) prediction view. | frontend/index.html + /predict/text |
| FIG-42 | report/figures/ui_live_webcam.png | Real-Time Webcam Detection Studio showing default ResNet-18 (RAF+RAVDESS) backbone selector and status pills. | frontend/index.html + /predict/webcam/status |
| FIG-43 | report/figures/ui_results_benchmarks.png | Benchmarks & Ablation view showing empirical MELD ablation table and reliability diagram. | frontend/index.html (#tab-research) |
| FIG-44 | report/figures/ui_history.png | Session Prediction History and Audit Log view. | frontend/index.html (#tab-history) |
| TAB-01 | report/tables/latency_benchmark.csv | Unified 200-frame CPU latency benchmark (p50/p95/mean/FPS) across all 3 backbones in one run on Disgust (2).mp4. | New folder/Disgust (2).mp4 + WebcamService |
| TAB-02 | report/tables/backbone_comparison.csv | Comparison of all three webcam vision backbones across RAF-DB test, RAVDESS val, 9 webcam clips, and unified 200-frame CPU latency. | results/eval_*.json + external_models/ + latency_benchmark |
| TAB-03 | report/tables/per_clip_results.csv | Per-clip ground truth, majority vote, and frame accuracy across all 9 evaluation clips for all 3 backbones. | results/eval_efficientnet_current.json + eval_resnet18_raf.json + eval_resnet18_ravdess.json |
| TAB-04 | report/tables/rafdb_and_ravdess_metrics.csv | Class-wise precision, recall, F1, and support on RAF-DB test and RAVDESS validation. | results/webcam_rafdb_reference.json + external_models/resnet18_raf/test_classification_report.csv + report_cache |
| TAB-05 | report/tables/datasets_summary.csv | Summary of all datasets used across training and evaluation (source, size, splits, classes, license). | MELD.Raw/ + DATASET/ + ravdess_crops/ + New folder/ |
| TAB-06 | report/tables/hyperparameters.csv | Training hyperparameters across ResNet-18 (RAF+RAVDESS), ResNet-18 (RAF-DB), EfficientNet-B0, and MELD Fusion. | train_mixed.py + external_models/resnet18_raf/config.json + train_webcam_phase3.py + train.py |
| TAB-07 | report/tables/environment.csv | Software and hardware execution environment specifications. | platform + Get-CimInstance + .venv package imports |
| TAB-08 | report/tables/api_endpoints.csv | Complete list of FastAPI REST and WebSocket endpoints. | backend/main.py + backend/api/*.py |
| TAB-09 | report/tables/test_cases.csv | Input vs. Expected Output vs. Actual Output verification results from tests/run_tests.py --all (5/5 unit, 11/11 integration PASS). | tests/run_tests.py --all + tests/*.py |
| TAB-10 | report/tables/meld_model_results.csv | MELD unimodal, audio improvement, and 7-configuration fusion evaluation results. | report_cache + results/full_ablation_benchmark_results.json + audio_classifier_improvement/comparison.csv |
| TAB-11 | report/tables/bill_of_materials.csv | Project Bill of Materials listing existing hardware, software, pretrained weights, and datasets (unknown costs marked [TO BE FILLED]). | environment + requirements.txt + models/ + datasets |
| TAB-12 | report/tables/risk_register.csv | Project risk register derived from real engineering, domain-shift, latency, privacy, and imbalance issues in webcam_fix_log.md and results/. | results/webcam_fix_log.md + backend/services/webcam_service.py |
| F1 | report/formulas/F1_softmax.png | Softmax class probability mapping | backend/services/webcam_service.py:765, audio_service.py:142, video_service.py:141 |
| F2 | report/formulas/F2_cross_entropy_label_smoothing.png | Cross-entropy loss with label smoothing (eps=0.1) | train_mixed.py:686,768 |
| F3 | report/formulas/F3_focal_loss.png | Class-weighted focal loss for imbalanced MELD/webcam training | backend/models/audio_model.py:41-59, train_webcam_phase3.py:163-192 |
| F4 | report/formulas/F4_classification_metrics.png | Accuracy, per-class precision, recall, F1, and macro-F1 | train_mixed.py:492-505, train_webcam_phase3.py:212-232 |
| F5 | report/formulas/F5_confusion_matrix.png | 7x7 confusion matrix and row-normalized proportions | train_mixed.py:515-521, backend/utils/metrics.py:55 |
| F6 | report/formulas/F6_class_balanced_weight.png | Inverse-frequency class-balanced WeightedRandomSampler weight | train_mixed.py:424-431 |
| F7 | report/formulas/F7_input_normalization.png | Channel-wise ImageNet RGB normalization from checkpoint preprocessing | train_mixed.py:383, live_resnet18_demo.py:44, webcam_service.py:137 |
| F8 | report/formulas/F8_resnet_residual_block.png | ResNet residual skip-connection block (ResNet-18 / ResNet-50) | train_mixed.py:29, live_resnet18_demo.py:32, video_service.py:33 |
| F9 | report/formulas/F9_conv_relu.png | 2D spatial convolution and ReLU activation | torchvision.models.resnet18, backend/models/audio_model.py:13 |
| F10 | report/formulas/F10_batch_norm.png | Batch Normalization (frozen stats in ResNet-18 stages, active in StrongerAudioClassifier) | train_mixed.py:465-474, backend/models/audio_model.py:30 |
| F11 | report/formulas/F11_adamw_update.png | AdamW update with decoupled weight decay and differential learning rates | train_mixed.py:477-489 |
| F12 | report/formulas/F12_mixed_batch_composition.png | Mixed RAF-DB + RAVDESS mini-batch composition rule | train_mixed.py:618-621 |
| F13 | report/formulas/F13_aligned_square_crop.png | Eye-line affine rotation around eye midpoint and aspect-preserving square crop (pad=1.10 / 1.25) | live_resnet18_demo.py:49-81, backend/services/webcam_service.py:585-586 |
| F14 | report/formulas/F14_blur_quality_gate.png | Variance of Laplacian blur quality gate (threshold 30.0) | prepare_ravdess.py:131, backend/services/webcam_service.py:592-593 |
| F15 | report/formulas/F15_majority_and_pooled_acc.png | Clip-level majority vote and pooled frame accuracy | eval_clips.py:113-115 |
| F16 | report/formulas/F16_uncertain_rule.png | Top-1 confidence and margin uncertainty flagging rule | backend/services/webcam_service.py:832-846 |
| F17 | report/formulas/F17_percentile_latency.png | Median (p50) and tail (p95) per-frame inference latency | eval_clips.py:207-208 |
| F18 | report/formulas/F18_efficientnet_compound_scaling.png | EfficientNet-B0 compound scaling formulation (used by efficientnet_current) | backend/services/webcam_service.py:140-144, train_webcam_phase3.py:35 |
| F19 | report/formulas/F19_ema_smoothing.png | Confidence-adaptive exponential moving average temporal smoothing | backend/services/webcam_service.py:770-775 |
| F20 | report/formulas/F20_meld_audio_pooling_mlp.png | MELD Wav2Vec2 masked mean pooling and StrongerAudioClassifier MLP | backend/services/audio_service.py:130-142, backend/models/audio_model.py:21-39 |
| F21 | report/formulas/F21_meld_fusion_equations.png | MELD Cross-Modal Attention Fusion with entropy reliability gates and consensus residuals | backend/models/fusion_model.py:172-216, backend/services/fusion_service.py:112,249-262 |
| F22 | report/formulas/F22_calibration_metrics.png | Expected Calibration Error (ECE) and multi-class Brier Score | backend/utils/metrics.py:10-48 |
