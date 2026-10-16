# External Generalization Evaluation

European ST-T and Icentia11k are external datasets relative to the three released MIT-BIH-trained models. Neither dataset is used for training, fine-tuning, or validation-based checkpoint selection of these weights. The purpose is to assess cross-database generalization.

This release includes completed European ST-T result tables. Icentia11k support is provided as evaluation code only; no Icentia result tables, progress logs, or partial experiments are included.

## European ST-T

Download: https://physionet.org/content/edb/1.0.0/

```bash
python -m mitbih_aami.evaluate_records --database-dir ./european-st-t-database-1.0.0 --checkpoint ./checkpoints/ecg_cnn_transformer_32s.pt --output-dir ./runs/stt_32s --tolerance-ms 150
```

Select the 16-s or 64-s checkpoint to evaluate another model. Input is resampled to 360 Hz. Use a separate output directory for each checkpoint.

## Icentia11k (evaluation code)

Download: https://physionet.org/content/icentia11k-continuous-ecg/1.0/

Expected layout: `icentia11k-continuous-ecg-1.0/p00/p00000/p00000_s00.hea`, with matching `.dat` and `.atr` files.

```bash
python -m mitbih_aami.evaluate_icentia --database-dir ./icentia11k-continuous-ecg-1.0 --patient-start 0 --patient-end 49 --tolerance-ms 150 --output-dir ./runs/icentia_external_00000_00049
```

The range is inclusive. The evaluator loads all three released checkpoints and writes separate reports. Each original record segment is processed independently; boundaries are not concatenated. Missing annotation files are listed and excluded. Reusing an unchanged output directory resumes completed record results; use a new directory if code, data, or weights change.

Reference Q beats participate in location matching and R-peak scoring but are excluded from class scoring. Reference N/S/V predicted as Q remain classification errors. The class matrix retains prediction F/Q columns, while class metrics are reported for N/S/V.

All generated outputs remain in ignored `runs/` directories and are not automatically published. The supplied preprocessing is bandpass filtering followed by whole-record z-score; this release does not claim automatic artifact repair. V5 rules are documented in [postprocessing_v5.md](postprocessing_v5.md).
