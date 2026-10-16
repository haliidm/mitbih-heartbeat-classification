# Internal train / validation / test results

Saved training artifacts; no retraining or evaluation was run for this documentation update.

Original S/V/F/Q-centered windows use a 70/10/20 random split (seed 42). Supplemental rhythm-N windows are then added. Each model has 13,224 training, 1,892 validation, and 3,778 test windows (18,894 total).

Training windows update parameters. Validation pointwise macro-F1 selects the checkpoint. The selected checkpoint is loaded for the held-out test-window evaluation after training. Full-database inference is a separate subsequent evaluation.

| Window | Selected epoch | Validation pointwise macro-F1 | Test pointwise macro-F1 |
|---|---:|---:|---:|
| 16 s | 50 | 98.83% | 98.72% |
| 32 s | 48 | 99.48% | 99.52% |
| 64 s | 49 | 99.78% | 99.76% |

Macro-F1 above averages N/S/V/F/Q, excluding background. It measures per-sample segmentation, not beat-event accuracy or R-peak F1.

## Window-center class distribution

| Window | Split | Windows | N | S | V | F | Q |
|---|---|---:|---:|---:|---:|---:|---:|
| 16 s | train | 13,224 | 20 | 1,941 | 5,068 | 576 | 5,619 |
| 16 s | validation | 1,892 | 6 | 274 | 729 | 72 | 811 |
| 16 s | test | 3,778 | 5 | 566 | 1,439 | 155 | 1,613 |
| 32 s | train | 13,224 | 20 | 1,941 | 5,068 | 576 | 5,619 |
| 32 s | validation | 1,892 | 6 | 274 | 729 | 72 | 811 |
| 32 s | test | 3,778 | 5 | 566 | 1,439 | 155 | 1,613 |
| 64 s | train | 13,224 | 20 | 1,941 | 5,068 | 576 | 5,619 |
| 64 s | validation | 1,892 | 6 | 274 | 729 | 72 | 811 |
| 64 s | test | 3,778 | 5 | 566 | 1,439 | 155 | 1,613 |

Counts describe the center beat used to sample each window, not the number of all beats inside those windows. Per-class labeled-sample support for the test set is stored in pointwise_classification_metrics.json.

Each model folder includes split_manifest.json, rhythm_n_windows.json, selected_validation_metrics.json, and pointwise_classification_metrics.json. Window membership differs from patient independence: same-record and overlapping signal context can appear across splits. Test windows are not directly used for parameter updates or checkpoint selection, but this protocol is not a patient-disjoint benchmark.
