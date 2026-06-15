# Paper 1 — Dataset and Methodology (draft)

**Venue strategy:** see `abstract.md` (not locked). Citations in (Author,
Year) form; entries + flags in `introduction.md` appendix.
**Status:** first full draft — author to revise. Numbers grounded in the
committed pipeline (`configs/`, `src/`, `scripts/`, `data/splits/`,
`results/`).

---

## 3. Dataset and Methodology

Our methodology proceeds in five stages: data preparation (§3.1), detector
benchmarking (§3.2), semi-automatic video annotation (§3.3), tracker
benchmarking (§3.4), and trajectory-quality evaluation (§3.5). Figure F1
summarises the pipeline; all stages run locally on a single RTX 5090 (32 GB)
under Windows 11, with no cloud compute.

### 3.1 Dataset and preparation

**Acquisition.** Top-down (near-nadir) video was recorded over Thai roads at
two flight altitudes, 100 m and 120 m, spanning scenes from free-flow
carriageways to a complex roundabout, at 4K resolution and 29.97 fps. Frames
for detector training were extracted at ~10 s intervals so that intra-clip
temporal leakage between train/val/test is minimal.

**Annotation and classes.** Vehicles are labelled with axis-aligned
bounding boxes (HBB) in three classes, ordered large-to-small:
`0 = LV` (large vehicles: trucks, buses, trailers), `1 = MV` (medium
vehicles: cars, vans, pickups, SUVs), and `2 = SV` (small vehicles:
motorcycles). We use HBB rather than oriented boxes (OBB) because, from a
top-down view, orientation is recoverable from trajectory direction; HBB has
first-class support across the detector and tracker stacks; standard tracking
metrics (HOTA/MOTA/IDF1) are HBB-native; and articulated vehicles (trailers)
break a single rigid OBB (decision log D1).

**Composition and splits.** The detection dataset comprises 3,010 annotated
frames (1,497 at 100 m, 1,513 at 120 m), partitioned 70/15/15 into
train/val/test (2,107 / 452 / 451 frames) with a fixed seed (42) for
reproducibility. The class distribution is heavily imbalanced and reflects
real Thai traffic: across the training split, medium vehicles dominate
(~57% of objects), small vehicles (motorcycles) are the plurality of road
users at ~41%, and large vehicles are rare (~2%). Per-altitude object counts
are reported in Table T1. Despite the LV scarcity, full-data training learns
it cleanly, so the imbalance does not preclude a fair benchmark.

**Pipeline and provenance.** Per-altitude exports are altitude-prefixed for
provenance (`01_rename_with_altitude_prefix.py`) and merged into a single
stratified split (`00_prepare_splits.py`) producing a flat `dataset.yaml`,
an `altitude_index.json`, and `split_metadata.json`. The altitude index
enables stratified per-altitude evaluation downstream from a single combined
model. A duplicate-label audit found and removed ~0.8% exact-duplicate boxes
(~98% of them MV) in the source export; all detector evaluations use the
deduplicated ground truth (decision log D5).

### 3.2 Stage 1 — Detector benchmarking

**Detectors.** We benchmark six modern detectors at the medium size for a
fair capacity-matched comparison: YOLOv9-m (Wang et al., 2024a), YOLOv10-m
(Wang et al., 2024b), YOLO11-m (Jocher and Qiu, 2024), YOLO12-m (Tian et al.,
2025), YOLO26-m (Sapkota et al., 2025), and the real-time transformer
RT-DETR-L (Zhao et al., 2024). All are driven through a single Ultralytics
entry point, so no per-detector code branching confounds the comparison.

**Training protocol.** Each detector is trained for up to 100 epochs (early
stopping with patience 20) at 1280×1280 input resolution — chosen to retain
small vehicles at both altitudes — with a cosine learning-rate schedule and
Ultralytics' native augmentation. The optimiser is selected automatically
(AdamW). Batch size is fixed per detector rather than auto-tuned, because
auto-batch would confound the cross-detector comparison with stochastic
batch-size variance: batch 8 for all detectors except YOLO12-m at batch 4,
whose area-attention memory peaks at 1280 exceed what auto-tuning would
stabilise. We do **not** use `batch=-1`.

**Variance-aware evaluation.** Each detector is trained over five random
seeds (0–4), yielding 30 trained models, and we report the mean and standard
deviation across seeds — a level of variance reporting rarely present in the
UAV-detection literature but necessary given that inter-detector gaps can be
smaller than run-to-run variance (Wang et al., 2020). We follow Strategy C
(decision log D2): one model is trained on the combined 100 m + 120 m set and
evaluated **stratified by altitude** via the altitude index, maximising
training data, yielding a single deployable model, and preserving
per-altitude diagnostics. Metrics are mAP@50 and mAP@50-95, reported overall,
per altitude, and per class.

### 3.3 Stage 2 — Semi-automatic video annotation

Tracker and trajectory evaluation require frame-accurate ground-truth tracks
on held-out sequences, which we produce with a human-in-the-loop pipeline
over four clips held out from detector training. The best detector
(RT-DETR-L) coupled with ByteTrack generates pseudo-tracks
(`02_generate_pseudo_tracks.py`), which a three-pass offline linker stitches
into longer fragments (`02b_link_tracks.py`), cutting raw track counts by
~80%. The linker is a fragment-stitcher, not a tracker — it depends on
ByteTrack's frame-to-frame association. Seeds are packed into CVAT-importable
MOT-1.1 archives (`02d_pack_cvat_mot.py`), a human corrects every track over
five review passes per a written protocol, and the export is converted back
to repo-convention ground truth (`02e_unpack_cvat_mot.py`). All four clips
are corrected over the full 1,799-frame span, yielding 40, 218, 61, and 115
ground-truth tracks respectively across free-flow and complex-geometry
scenes.

### 3.4 Stage 3 — Tracker benchmarking

**Trackers and coupling.** Four trackers are benchmarked through one BoxMOT
harness: ByteTrack (Zhang et al., 2022), BoT-SORT (Aharon et al., 2022),
OC-SORT (Cao et al., 2023), and Deep OC-SORT (Maggiolino et al., 2023). To
isolate association quality from detection quality, all four consume an
identical set of RT-DETR-L detections — "detect once, track four times" — so
that DetA is shared and any spread is attributable purely to association.
Appearance association uses a vehicle re-identification encoder (VeRi-776
CLIP-ReID, FP16); we found this changed every aggregate metric by ≤0.003
versus a person-ReID baseline, a null result indicating that overhead
association is motion/IoU-dominated and top-down appearance is weakly
discriminative (decision log D3 follow-up).

**Robustness patch.** Deep OC-SORT's Kalman filter occasionally raised a
non-positive-definite Cholesky failure on dense scenes; we apply a runtime
patch (`src/trackers/boxmot_patches.py`) that projects the covariance onto
the nearest PSD matrix only when a factorisation would otherwise crash,
introducing no hyperparameter change and leaving the locked environment
untouched.

**Evaluation.** Each tracker is scored against Stage 2 ground truth with
HOTA, MOTA, and IDF1 via TrackEval, in two arms — raw and post-linker — to
quantify the linker's contribution (decision log D3, resolved to Option B,
keeping the comparative benchmark and dual evaluation). The evaluator was
validated against synthetic ground truth (perfect tracks score 1.0; ID
switches, false negatives, and cross-sequence combination all behave
correctly).

### 3.5 Stage 4 — Trajectory-quality evaluation

The paper's central contribution is a framework that evaluates trajectories
in world coordinates rather than boxes in image space.

**Pixel-to-world calibration.** For each clip we fit a ground-plane
homography from manually identified road-marking ground-control points
(`00_calibrate.py`, `src/trajectory/calibration.py`), reporting reprojection
RMSE and a held-out scale-consistency check that flags inconsistent points.
The recovered scale cross-validates by altitude (~0.030 m/px at 100 m,
~0.040 m/px at 120 m). GCP specifications are committed; fitted homographies
are regenerable.

**Trajectory extraction.** MOT outputs are projected to world metres and
differentiated into kinematic features — speed, acceleration, heading, jerk —
using a Savitzky–Golay smoothing-differentiation filter (Savitzky and Golay,
1964; default window 15, polynomial order 2, adapted down for short tracks),
with trajectories split at long temporal gaps and physically impossible
jumps (`smoothing.py`, `kinematics.py`, `04_extract_trajectories.py`).

**Three-axis metrics.** Each trajectory set is scored along three axes
(`src/evaluation/trajectory_metrics.py`):
- **Axis 1 — identity persistence:** track completeness, fragments per GT
  track, and ID switches.
- **Axis 2 — kinematic accuracy vs. GT:** ADE/FDE and positional RMSE in
  metres, speed MAE, and heading MAE, under per-frame greedy
  nearest-neighbour GT↔prediction matching in world metres (5 m gate).
- **Axis 3 — intrinsic physical plausibility (GT-free):** the fraction of
  acceleration samples within a physical bound (|a| ≤ 8 m/s²), jerk
  distribution, path curvature, and speed statistics; headings and curvature
  are treated as undefined below a 1 m/s motion threshold.

All metrics are reported for both raw and Savitzky–Golay-smoothed series,
which lets us test whether smoothing reorders the tracker ranking. We
pre-registered the hypothesis that Kalman-based trackers might invert their
ranking under smoothing (decision log D4).

**GT-free generalisation.** To demonstrate reusability, we additionally apply
the Axis-3 plausibility metrics and self-consistency proxies to two new,
unlabelled 5-min 120 m intersection clips with no ground truth, testing
whether the framework reproduces the labelled-benchmark ranking without any
annotation.

### 3.6 Implementation and reproducibility

All experiments run on one RTX 5090 (32 GB) under native Windows 11 (not
WSL), Python 3.12, PyTorch 2.11.0 + CUDA 12.8 (Blackwell sm_120). The
environment is pinned (`requirements.lock.txt`); fixed seeds, committed
configs, and committed result CSVs make the benchmark reproducible. Code and
configurations are released openly.

---

## Drafting notes (not part of the section)

- **Numbers to lock against tables at proof:** §3.1 split counts (2,107 /
  452 / 451) and class fractions are from `data/splits/split_metadata.json`;
  Stage-3 ReID delta (≤0.003) and GT track counts (40/218/61/115) from
  CLAUDE.md/`results/stage3/`. Cross-check the final T1/T3 tables.
- **Acquisition specifics to fill from field notes:** exact dates, drone
  model/camera, GSD per altitude, clip durations and count, weather/time of
  day — currently described generically. The author has these.
- **Strategy C / batch sizing / HBB:** all cite the decision log; for a CV or
  remote-sensing venue, consider promoting the batch-sizing and resolution
  rationale into the text (reviewers there probe these); for a transportation
  venue, the current brief treatment is likely sufficient.
- **A_MAX = 8 m/s², gate = 5 m, SG(15,2):** these are the committed defaults
  in `trajectory_metrics.py` / `smoothing.py`. State them as design choices
  with brief justification; a reviewer may ask for sensitivity.
- **Figure/table hooks:** F1 (pipeline), T1 (dataset), T2 (detectors),
  T3 (per-class) belong with §3.1–3.2; T4 with §3.4; T5/T7 with §3.5.
