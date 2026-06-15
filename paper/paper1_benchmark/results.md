# Paper 1 — Results (draft)

**Venue strategy:** see `abstract.md` (not locked). Citations in (Author,
Year) form; entries in `introduction.md` appendix.
**Status:** first full draft — author to revise. **All numbers below are
copied from the committed paper tables** (`paper/tables/T1–T7`,
`results/stage1|3|4/`); they are the canonical values, not re-derived.
Table/figure callouts use the existing labels.

---

## 4. Results

### 4.1 Detector benchmark (Table T2, Fig. F3)

Across five seeds per detector (30 trained models), the six detectors
separate into three tight tiers on the stratified test set (Table T2;
overall mAP@50 / mAP@50-95, supervision metrics):

| Rank | Detector | mAP@50 | mAP@50-95 | 100 m mAP@50 | 120 m mAP@50 |
|------|----------|--------|-----------|--------------|--------------|
| 1 | RT-DETR-L | 0.963 ± 0.002 | **0.876 ± 0.002** | 0.969 | 0.958 |
| 2 | YOLO26-m | 0.958 ± 0.002 | 0.866 ± 0.001 | 0.963 | 0.953 |
| 3 | YOLOv10-m | 0.954 ± 0.001 | 0.858 ± 0.000 | 0.960 | 0.947 |
| 4 | YOLO11-m | 0.935 ± 0.001 | 0.843 ± 0.001 | 0.943 | 0.933 |
| 5 | YOLO12-m | 0.935 ± 0.002 | 0.841 ± 0.002 | 0.943 | 0.931 |
| 6 | YOLOv9-m | 0.933 ± 0.001 | 0.837 ± 0.001 | 0.941 | 0.928 |

Three findings stand out. **(1) RT-DETR-L wins on every metric**, leading the
strongest YOLO competitor (YOLO26-m) by ~0.5 mAP@50 / ~1.0 mAP@50-95 points;
the real-time transformer thus tops the modern YOLO line on top-down traffic.
**(2) The ranking is stable**: per-seed standard deviation is ≤0.002
everywhere, so the inter-tier gaps exceed run-to-run variance — a separation
that single-seed studies cannot establish (Wang et al., 2020). **(3) Altitude
matters consistently**: 100 m outperforms 120 m by ~1.5–2 mAP@50 points for
every detector, confirming that the lower flight height's larger pixel
footprint aids detection across architectures.

### 4.2 Accuracy–efficiency trade-off (Table ST1, Fig. F3)

The accuracy ranking is not free (Table ST1 / Fig. F3, the
accuracy-vs-FPS Pareto front). RT-DETR-L is the most accurate but the
heaviest and slowest (32.8 M params, 421.7 GFLOPs, 55.4 FPS at 1280 on the
RTX 5090), whereas YOLOv10-m is the throughput leader (16.5 M params, 256.0
GFLOPs, 161.4 FPS) at a ~1.7-point mAP@50 cost. YOLO26-m is the most balanced
operating point — second on accuracy at 138.4 FPS. For offline trajectory
extraction, where throughput is not the binding constraint, RT-DETR-L's
accuracy makes it the natural choice for the downstream stages, and we adopt
it as the detector for Stages 2–4.

### 4.3 Per-class difficulty (Table T3)

Per-class AP (Table T3) reveals where the difficulty lies. Small vehicles
(motorcycles, SV) are by far the hardest class at the strict AP@.5:.95
(RT-DETR-L: 0.787, vs. 0.891 for LV and 0.949 for MV), even though their
loose-IoU AP@.5 is high (0.963). The gap is therefore one of **localisation
precision**, not detection — motorcycles are found but their small footprint
penalises tight-IoU matching. This per-class signal also separates the
mid-tier detectors: YOLO11/12/v9-m tie on the easy dominant MV class
(AP@.5:.95 ≈ 0.885) and differ chiefly on LV and SV, confirming that the
overall ranking is driven by the hard classes. The finding has direct
downstream relevance, since motorcycles dominate the object count in this
traffic and feed the Paper 2 modelling arc.

### 4.4 Tracker benchmark (Table T4)

Coupling RT-DETR-L to four trackers over the four annotated sequences
(class-agnostic, clip-pooled; Table T4), the post-linker leaders are
ByteTrack (HOTA 0.891, IDF1 0.871) and BoT-SORT (HOTA 0.881, IDF1 0.874),
ahead of Deep OC-SORT (0.856 / 0.846) and OC-SORT (0.845 / 0.835). Because
all four consume identical RT-DETR-L detections, raw DetA is ≈0.82–0.84 for
all of them, so **the spread is pure association quality**. Two effects are
clear: the offline linker lifts every tracker (e.g. ByteTrack raw→linked HOTA
0.823→0.891) and collapses identity switches (OC-SORT 3433→1062; ByteTrack
634→495); and the motion/IoU-based ByteTrack and BoT-SORT consistently beat
the OC-SORT family on association from this overhead viewpoint. Consistent
with the ReID null result (decision log D3), vehicle appearance adds little
from the top-down view.

### 4.5 Trajectory quality (Tables T5, T5b, T7)

After per-clip pixel-to-world calibration (Table T7; reprojection RMSE ≈ 0
with four GCPs, scale cross-validating to ~0.031 m/px at 100 m and ~0.035–
0.040 m/px at 120 m), we score world-space trajectories along the three axes
(Table T5, smoothed series).

- **Axis 1 (identity):** BoT-SORT and ByteTrack lead on completeness
  (0.953 / 0.950), fragments per GT track (2.09 / 2.22), and ID switches
  (132 / 149), versus the OC-SORT family (completeness ~0.91, switches
  202–249).
- **Axis 2 (kinematic accuracy):** positional ADE is near-zero for all
  trackers (≈0.003–0.008 m) because the ground truth shares RT-DETR
  detection provenance from semi-automatic seeding; the discriminating
  signal is therefore in the **differentiated** channels — speed MAE
  (ByteTrack 0.029 m/s, best) and heading MAE (ByteTrack 0.29°, best) —
  exactly the point that trajectory quality is not captured by box overlap.
- **Axis 3 (plausibility):** ByteTrack matches the GT ceiling on the
  acceleration-in-bounds fraction (0.977) and mean jerk (22.0 m/s³),
  while the OC-SORT family is markedly jerkier (28.1 / 27.9 m/s³).

**The central finding** is that the trajectory-quality ranking *agrees* with
the box-level ranking: the trackers that lead on HOTA/IDF1 also lead on
identity persistence *and* physical plausibility. We had pre-registered the
possibility of a Kalman-smoothing inversion — that OC-SORT's filter might
clean up its trajectories enough to overtake on plausibility (decision log
D4) — but Table T5b shows this does **not** occur: Savitzky–Golay smoothing
lifts every source toward the GT ceiling and *preserves* the ordering in both
raw and smoothed columns. OC-SORT's weaker association injects more kinematic
noise than its filter removes. Box metrics are therefore a *validated cheap
proxy* for trajectory quality in this setting — a positive result that holds
because we tested it rather than assuming it.

### 4.6 Ground-truth-free generalisation (Table T6, Fig. F9)

To test reusability, we applied the Axis-3 plausibility metrics and
self-consistency proxies to two new, unlabelled 5-min 120 m intersection
clips with no ground truth (Table T6). The labelled-benchmark ranking
**replicates with zero ground truth**: ByteTrack and BoT-SORT lead on
self-consistency (segments per track ~2.1 vs. ~3.5–3.9 for the OC-SORT
family; impossible-jump splits 80–94 vs. 200–207) and on plausibility (raw
acceleration-ok 0.86 vs. 0.77; smoothed jerk ~21 vs. ~26). The framework thus
ranks trackers correctly without annotation, establishing an
annotation-light protocol for evaluating UAV trajectory pipelines on new
footage.

---

## Drafting notes (not part of the section)

- **All values pasted from `paper/tables/*.tex`** (T2 via `aggregated.csv`;
  ST1 via `efficiency.csv`/`inference_speed.csv`; T3, T4, T5, T5b, T6, T7).
  **Caution:** T4/T6 use the clip-pooled `ALL` row (TrackEval combined
  aggregation) — a naive mean-of-per-clip gives different (lower) HOTA, so
  keep the pooled values and the "clip-pooled" wording aligned with T4's
  caption.
- **FPS is hardware-specific** (RTX 5090, fp32, 1280, batch-1): state the
  measurement conditions wherever FPS appears so reviewers can interpret it.
- **Axis-2 ADE ≈ 0 needs the provenance caveat every time** it is mentioned
  (GT shares RT-DETR detections) — otherwise a reviewer reads it as
  implausibly perfect. Already stated in §4.5; keep it.
- **Figures to place:** F3 (Pareto) in §4.2; F2 (scene gallery) supports
  §4.1/§3.1; F4 (box-vs-trajectory) and F7 (roundabout spaghetti) in §4.5;
  F5 (smoothing) with T5b; F6a/F6b (plausibility ECDFs) in §4.5; F8
  (calibration overlay) with T7; F9 (GT-free bars) in §4.6.
- **Supplementary tables** ST2–ST6 (architectures, training protocol, tracker
  hparams) belong in an appendix; T8 (finetune-vs-pretrained) appears to be a
  Nadir-YOLO/Paper-1.5 artefact — **exclude from Paper 1**.
- Decide whether §4.5's "validated cheap proxy" framing is stated as a
  headline finding in the abstract (currently implied) — it is the cleanest
  one-line takeaway of the paper.
