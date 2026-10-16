# V5 Beat Postprocessing

Implementation: `mitbih_aami/beat_postprocess.py` and `label_cleanup.py`. The same V5 decoder is used by the desktop viewer and current evaluators. All decisions run on the full-record labels before display windows are cut.

## 1. Form regions and remove very short regions

A region is a continuous run of nonzero labels. An N→S→V transition without background between labels does **not** create three beats. V5 passes `max_gap_ms=0`: it does not fill background gaps. Remove every nonzero region shorter than 100 ms, including internal and boundary regions. Durations are converted to samples using rounding.

## 2. Remove contextual short regions

A normal-width region is 180–500 ms inclusive. Estimate typical RR as the median of center-to-center intervals between normal-width regions, keeping only intervals of 300–2000 ms. At least three such intervals are required; otherwise skip this step.

Remove a region shorter than 180 ms only if all these conditions hold:

- It has an immediate region on each side, both of normal width.
- The center distance between those two outer regions is 300–2000 ms.
- That distance lies within ±25% of the typical RR.
- At least one background gap from the short region to its neighbors is strictly less than 180 ms.

All decisions use the same region list before this step's deletions. The first and last regions are not candidates for this contextual rule. Regions shorter than 100 ms have already been removed by step 1.

## 3. Locate beats and split long regions

For a region no longer than 500 ms, use its midpoint, `(start + end - 1) // 2`, as the predicted beat position (end is exclusive).

For a region longer than 500 ms, search the absolute amplitude of the preprocessed ECG for local peaks:

| Parameter | Value |
|---|---|
| Minimum peak separation | 200 ms |
| Local amplitude scale | 95th percentile of absolute ECG amplitude inside the region |
| Minimum peak height | 0.45 × local scale |
| Minimum prominence | max(0.3 normalized amplitude, 0.35 × local scale) |

If at least two peaks remain inside the region, split at the midpoints between adjacent peaks and use each peak as the corresponding beat position. Otherwise retain a single region and use the region midpoint. Splitting is not forced by equal-duration division. Split pieces are not passed through the earlier short-region filters again.

## 4. Assign a class and display it

Within each final region, if the proportion of N-labeled samples is **strictly greater than 10%**, assign N. Otherwise choose the most frequent nonzero class. Ties use the smallest class index (N=1, S=2, V=3, F=4, Q=5). The App paints the entire final region with this assigned class.

## 5. Keep decoding separate from evaluation

Predicted-region decisions do not use ATR annotations. Reference peak refinement and evaluation matching are separate operations. Matching tolerance (for example ±150 ms) changes scoring, not the predicted labels or V5 thresholds.

Current preprocessing is 0.67–40 Hz bandpass filtering followed by full-record z-score normalization, with resampling where necessary. Experimental Icentia artifact exclusion and missing-value interpolation are not included in this release.
