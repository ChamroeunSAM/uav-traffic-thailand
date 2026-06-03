# Decision Log

A running record of methodological decisions, the alternatives considered, and the
reasoning behind each choice. New decisions append to the bottom; do not rewrite
history.

---

## D1 — Use horizontal bounding boxes (HBB) over oriented bounding boxes (OBB)

**Date:** 2026-05-08
**Status:** Accepted

**Decision.** Annotate and detect vehicles with axis-aligned (horizontal) bounding
boxes rather than oriented (rotated) bounding boxes.

**Reasoning.**
- Annotation cost is markedly lower for HBB, letting us label more frames and
  cover more scenes within the project budget.
- The mainstream detector and tracker stacks we are benchmarking (YOLO26,
  YOLOv12, RF-DETR, ByteTrack, BoT-SORT, OC-SORT, Deep OC-SORT) have first-class
  HBB support; OBB support is uneven and would force per-tool workarounds.
- Most ground-truth tracking and trajectory metrics (HOTA, MOTA, IDF1) are
  defined over HBB; switching to OBB would require custom adaptations.
- For our downstream traffic-engineering use (vehicle counts, speeds, lane
  occupancy via projected centroids), HBB is sufficient — orientation can be
  recovered from the trajectory direction rather than from the box.

**Trade-offs.** HBB inflates apparent vehicle area for diagonally oriented
vehicles, which can hurt IoU at high thresholds; we mitigate by reporting
mAP@.5 alongside mAP@.5-.95 and by using centroid-based association in
trajectory metrics where appropriate.

---

## D2 — Strategy C: combined-altitude training with stratified evaluation

**Date:** 2026-05-08
**Status:** Accepted

**Decision.** Train a single detector on the combined 90 m + 120 m dataset
(`configs/data/dataset_combined.yaml`) and evaluate it stratified by altitude
(separate metrics on the 90 m and 120 m splits) rather than training a
separate detector per altitude (Strategy A) or training only on one altitude
and testing on the other (Strategy B).

**Reasoning.**
- Combined training maximizes data utilization and yields a single deployable
  model, which matches the operational reality of a UAV that flies a mix of
  altitudes within and across missions.
- Vehicle scale at 90 m vs. 120 m differs but overlaps; a combined model
  benefits from scale augmentation implicit in the data rather than learning
  two narrow scale distributions.
- Stratified evaluation preserves the diagnostic value of per-altitude metrics
  (so we can still report whether the model degrades at higher altitude /
  smaller objects) without paying the data-fragmentation cost of Strategy A.
- Strategy B (cross-altitude generalization) would be a secondary ablation,
  not the primary result, and is out of scope for the headline benchmark.

**Trade-offs.** A combined model may underperform a per-altitude specialist
on each altitude in isolation; we accept this for the operational and
data-efficiency benefits, and we leave per-altitude fine-tuning as future
work if stratified results show large gaps.


---
## 2026-05-08 — Dependency conflict: regex pinning between transformers and boxmot

**Issue:** During environment setup, transformers 5.8.0 (pulled in by RF-DETR)
performs a runtime check requiring `regex >= 2025.10.22`. boxmot has a
metadata-level pin requiring `regex < 2025.0.0`. Pip's resolver picked the older
version to satisfy boxmot's metadata, which broke transformers and rfdetr at
import time.

**Resolution:** Upgraded `regex` to `2026.4.4`, which satisfies transformers'
runtime requirement. boxmot's pin is metadata-only with no runtime enforcement,
so it still imports and runs correctly. Pip emits a metadata-conflict warning
about boxmot, which is harmless and can be ignored.

**Verification:** `import transformers`, `import rfdetr`, `import boxmot` all
succeed. Full GPU verification (Step 9 of environment setup) passed cleanly.

**Reproducibility note:** When recreating the environment from `requirements.lock.txt`,
the regex version is already locked at 2026.4.4. No additional steps needed.
---

## 2026-05-08 — Fixed batch size for benchmark comparability

**Decision:** Use fixed batch sizes (YOLO-m: 8, RF-DETR-m: 4) for all
benchmark training runs, not auto-batch (`batch=-1`).

**Rationale:** Auto-batch produces different effective batch sizes across
runs depending on system VRAM availability at the moment of detection.
Different batch sizes affect gradient noise, effective learning rate, and
convergence dynamics, which contaminates cross-detector comparison. Fixed
batch sizes ensure that performance differences between detectors reflect
the detectors themselves, not stochastic batch-size variation.

**Trade-off:** Slightly slower per-run training time. Acceptable.

**Pilot exception:** The pilot config uses `batch: -1` (auto) for speed.
Pilot results are not used in the benchmark — only for pipeline validation.
---

## 2026-05-08 — Stage 1 detector scope: six detectors + per-detector batch sizing

**Scope expansion (RF-DETR → RT-DETR; add YOLOv9 and YOLOv10):** Stage 1
detector benchmark expanded from four to six detectors to strengthen the
peer-reviewed citation foundation and broaden architectural coverage.

Final detector roster:

| Detector | Architecture | Reference |
|---|---|---|
| YOLOv9-m | CNN + PGI + GELAN | Wang et al. 2024 (peer-reviewed) |
| YOLOv10-m | NMS-free, dual assignments | Wang et al. NeurIPS 2024 |
| YOLO11-m | CNN baseline (anchor-free, NMS) | Ultralytics (software citation) |
| YOLO12-m | Area-attention augmentation | NeurIPS 2025 |
| YOLO26-m | NMS-free end-to-end | Ultralytics (third-party reviews) |
| RT-DETR-L | Transformer encoder-decoder | Zhao et al. CVPR 2024 |

**Why RT-DETR replaces RF-DETR:** RF-DETR's pretrained variants top out
at 704 px native resolution (RFDETRLarge); the training-time weight loader
in `rfdetr/training/module_model.py:166` calls `load_state_dict` directly
without PE interpolation, so resolution=1280 fails with a position-embedding
size-mismatch RuntimeError. Upstream issue #497 (Dec 2025) confirms this is
a known gap. Patching would create upstream-fork maintenance burden.
RT-DETR-L (Zhao et al. CVPR 2024) uses CNN-based feature extractors with no
fixed-PE-grid constraint, integrates cleanly through Ultralytics' YOLO()
auto-detection (returns RTDETR class), and trains at 1280 with no resolution
constraints. Validated end-to-end via 5-epoch extpilot on placeholder data:
159s runtime, 11.8GB peak VRAM.

**Per-detector fixed batch sizes (rejecting auto-batch):** Auto-batch picks
the largest batch that fits at run launch, depending on transient VRAM
state. This produces seed-to-seed batch-size variance that contaminates
the comparison — different effective batch sizes mean different gradient
noise, effective learning rate, and convergence dynamics. Q1-venue
reviewers will flag this. We use fixed per-detector batch sizes,
documented in the methods section:

| Detector | Batch | Rationale |
|---|---|---|
| YOLOv9-m | 8 | extpilot 16GB/batch=4 → 10.5GB peak; fits 32GB at batch=8 |
| YOLOv10-m | 8 | extpilot 16GB/batch=4 → 11.2GB peak; fits 32GB at batch=8 |
| YOLO11-m | 8 | extpilot 16GB/batch=8 → 21GB peak (spillover); fits 32GB |
| YOLO12-m | **4** | area-attention OOMs at batch=8 even on 16GB; batch=4 → 22GB peak |
| YOLO26-m | 8 | extpilot 16GB/batch=4 → 12.5GB peak; fits 32GB at batch=8 |
| RT-DETR-L | 8 | extpilot 16GB/batch=4 → 11.8GB peak; fits 32GB at batch=8 |

YOLO12-m is the methodologically uncomfortable case — its area-attention
operator is materially more memory-hungry than the other variants. Two
options were considered:

1. Drop YOLO12-m's batch to 4 across the benchmark, keep all others at 8.
   *Confound:* batch-size becomes part of the architecture comparison.
2. Drop ALL detectors to batch=4 to match.
   *Cost:* slower training, smaller effective batch, may underperform what
   each detector can do at its native batch.

We chose (1) and disclose the asymmetry explicitly in the methods. The
gradient-noise difference between batch=4 and batch=8 is small relative to
the architecture differences, and seed-averaging (5 seeds) absorbs much of
the residual variance.

**Hardware target and cost estimate:** Full benchmark on a 32GB rental
GPU (e.g. RTX 4090, A100-40GB, L40):
- 6 detectors × 5 seeds = 30 training runs
- Estimated 45–60 GPU-hours total (per-detector estimates in
  `scripts/run_benchmark.py:ROUGH_HOURS_PER_RUN`)
- Approximate cost: $80–$100 at typical rental rates ($1.5–2/hour for 32GB)

**RF-DETR wrapper retained:** `src/detectors/rfdetr_wrapper.py` is
unchanged and unused in Stage 1; kept for potential future research with
either upstream RF-DETR fixes (issue #497) or a custom PE-interpolation
training pipeline.
---

## 2026-05-16 — New PC (RTX 5090) validation + YOLO12m batch ceiling confirmed

**Context.** Project transferred to RTX 5090 build (32 GB VRAM) to
resolve the YOLO12m VRAM spillover constraint observed on 16 GB hardware.

**Pipeline validation.** Full train + stratified eval pipeline validated
end-to-end on placeholder data on the new hardware. YOLO12m at
`imgsz=1280`, `batch=4` peaks at 22.1 GB on the 5090 with no spillover,
~10 s/epoch (vs ~13 min/epoch on the old 16 GB PC — the spillover
bottleneck that drove the hardware upgrade is resolved for the batch=4
configuration).

**Empirical batch=8 ceiling test for YOLO12m.** Directly tested whether
YOLO12m could be standardized to `batch=8` (matching the other five
detectors) on the 32 GB hardware. Result: `batch=8` at `imgsz=1280`
peaks at **43.6 GB** — exceeding the 5090's ~31.8 GB physical VRAM by
~12 GB, forcing hard spillover into shared system memory and ~40×
per-epoch slowdown (395 s vs 10 s).

**Conclusion.** YOLO12m's `batch=4` setting is an intrinsic constraint of
its area-attention operator at 1280 resolution, not a small-GPU
artifact. Batch=8 does not physically fit on 32 GB hardware. The
asymmetric batch-size design (YOLO12m=4, all others=8) is empirically
justified. Training YOLO12m at batch=4 ensures its results are free of
memory-spillover artifacts — a fairer representation of its detection
capability than forcing batch=8 would be.

**Methods-section framing.** State the 43.6 GB measurement explicitly.
The asymmetry preempts reviewer questions with data rather than
defending an arbitrary choice.

**Other transfer-hardening fixes (this commit).**
- Ultralytics MLflow callback disabled programmatically in
  `src/detectors/ultralytics_wrapper.py` (the prior fix was a
  machine-local `Ultralytics/settings.json` edit; it crashes on Windows
  paths and we don't use MLflow in this project).
- `scripts/setup_environment.ps1` uses `pip install --no-deps -r
  requirements.lock.txt` (fully-resolved lock; avoids re-litigating the
  boxmot/regex resolver conflict).
- `docs/machine_transfer.md` added with the validated transfer
  checklist; flags that splits must be regenerated with `--force` after
  transfer because the generated `dataset.yaml` contains absolute paths.

**Benchmark implication.** Full 6-detector × 5-seed benchmark runs
locally on owned 5090, free, well under a day of compute once real
annotations arrive. No GPU rental needed.
---

## 2026-05-18 — Altitude correction: "90m" was actually 100m

**Issue:** Team communication clarified that UAV footage labeled "90m"
throughout the project was in fact collected at 100m altitude. The 120m
height is confirmed correct and unchanged.

**Scope of correction:** All forward-looking references updated 90m →
100m: source data folder (data/annotations/90m → 100m), file prefixes
(90m_ → 100m_), parsing/eval/benchmark scripts, configs (dataset_90m.yaml
→ dataset_100m.yaml), and forward-looking docs. Generated artifacts
(merged/, splits/) regenerated from corrected source.

**Historical record preserved:** Earlier decision-log entries and all
prior git commit messages retain "90m" as an accurate record of the
project's understanding at the time. They are deliberately NOT rewritten.
This entry documents the correction; the prior naming was an internal
labeling error, not a data-collection error — the footage itself is
valid, only its altitude label was wrong.

**Action items outside this commit:**
- methodology.docx must be manually updated by the author (Word doc,
  not edited programmatically)
- Real annotations, when they arrive, will use the correct 100m_ prefix
  via the updated 01_rename_with_altitude_prefix.py
---

## D3 — Stage 3 framing: dual evaluation (raw + post-linker) with fallback to single end-to-end

**Date:** 2026-06-02
**Status:** **Resolved 2026-06-03 — Option B locked** (dual evaluation,
full 4-tracker comparative benchmark). The mini-experiment was run; see
the Resolution block at the end of this entry. Original text below is
preserved as written.

**Decision.** Stage 3 will evaluate each of the four trackers (ByteTrack,
BoT-SORT, OC-SORT, Deep OC-SORT) *twice* against the Stage 2 ground truth:

1. **Raw tracker output** → HOTA / MOTA / IDF1. Provides comparability
   with the mainstream MOT literature, which uniformly reports raw-output
   metrics.
2. **Tracker + linker output** (overlap-merge + gap-link + per-frame dedup
   from `scripts/02b_link_tracks.py`) → same metrics. Provides relevance
   to the offline trajectory analysis that Paper 1 (Stage 4) and Paper 2
   actually need.

The gap between the two columns is itself a paper finding: it quantifies
the linker's contribution and identifies which tracker is the cleanest
*feedstock* for offline pipelines.

**Reasoning.**
- The traditional MOT benchmarks optimize for *online* tracker performance,
  which matters for autonomous driving and live surveillance. For Paper 1's
  *offline* trajectory analysis use case, post-processing matters at least
  as much as raw association quality.
- Reporting only raw metrics would obscure the linker contribution
  (which is a substantive Stage 2 methodological output) and would make
  the paper's tracker recommendations misleading for traffic-engineering
  practitioners who *will* post-process.
- Reporting only post-linker metrics would lose comparability with the
  MOT literature and would force reviewers to take the linker on faith.
- The dual evaluation is roughly twice the compute, which is negligible
  in absolute terms (well under one wall-clock hour for all 4 trackers ×
  4 clips on the 5090).

**Safety net — mini-experiment before full commit.** Before the full
4-tracker × 4-clip benchmark is run, a single-clip mini-experiment
(120_m_DJI_0006_1min, the moderate-density roundabout) will measure raw
and post-linker IDF1 for all four trackers. Decision rule:

- **If the post-linker spread stays wide** (e.g., raw 0.55–0.75 → linked
  0.65–0.85), the dual evaluation is informative and Stage 3 proceeds as
  planned (Option B).
- **If the post-linker spread compresses heavily** (e.g., raw 0.55–0.75 →
  linked 0.85–0.90), the tracker choice barely matters post-processing.
  Paper 1 then drops the comparative tracker benchmark entirely and uses
  a single end-to-end evaluation (one tracker — likely ByteTrack — plus
  linker, evaluated once against GT). The Paper 1 narrative shifts to
  position the linker and the Stage 4 trajectory framework as the
  headline contributions (Option A as fallback).

The fallback path was prepared because the post-linker quality may be
dominated by the linker rather than by the input tracker, in which case
the comparative benchmark adds little value.

**Inter-annotator validation explicitly dropped.** Per the same
discussion that produced this decision: §8 of `docs/stage2_annotation_protocol.md`
(original draft) called for a second annotator on one clip to establish
an IDF1 ceiling. That section was removed on the grounds that the
top-down UAV view at 100-120 m minimizes the perspective-induced
annotation ambiguity that typically motivates such checks in MOT17 /
MOT20-style ground-level benchmarks. A short Limitations paragraph in
the paper will note this choice for reviewers.

**Trade-offs.**
- Doubles Stage 3 evaluation compute (negligible in absolute terms).
- Forces the paper to explain two metrics with two slightly different
  narratives ("real-time online tracking" vs "offline trajectory
  pipeline"). Cleanly framable but adds a paragraph of methodological
  exposition.
- The fallback path means the paper's structure is not fully locked
  until the mini-experiment runs. Acceptable because the mini-experiment
  is cheap (~30 minutes) and yields a clear go/no-go signal.

**Action items.**
- Build TrackEval-based evaluation harness (HOTA / MOTA / IDF1) that
  reads MOT-format predictions + GT.
- Run mini-experiment on 120_m_DJI_0006_1min before the full Stage 3
  benchmark commits.
- After mini-experiment, lock Paper 1 framing (Option A or B) and update
  CLAUDE.md §3 accordingly.

**Resolution (2026-06-03) — Option B locked.** The full 4-tracker ×
4-clip dual evaluation was run (rather than only the single-clip
mini-experiment) and scored with TrackEval; results in
`results/stage3/tracker_metrics.csv`. The decision rule keys on the
post-linker IDF1 spread on the designated clip `120_m_DJI_0006_1min`:

| scope | raw IDF1 spread | linked IDF1 spread |
|---|---|---|
| `120_m_DJI_0006_1min` (designated clip) | 0.585–0.654 (w0.068) | **0.691–0.802 (w0.111)** |
| `100_m_DJI_0001_1min` | w0.104 | 0.560–0.638 (w0.078) |
| `120_m_DJI_0047_1min` | w0.093 | 0.819–0.867 (w0.048) |
| `100_m_DJI_0054_1min` (dense) | w0.079 | 0.880–0.913 (w0.033) |
| ALL (pooled) | w0.083 | 0.835–0.873 (w0.038) |

On the designated clip the post-linker spread stays **wide** (0.111,
landing at 0.69–0.80 — not the compressed 0.85–0.90 that would trigger
the Option A fallback), so the dual evaluation is informative and the
comparative benchmark is retained (**Option B**).

The pooled "ALL" row looks compressed (w0.038) only because the dense
clip (`0054`, ~259k GT rows) dominates the pool. This density dependence
is itself a finding: tracker choice **compresses on dense free-flow**
(all trackers reach IDF1 ≈ 0.88–0.91 post-link on `0054`) but **stays
wide on sparser / complex geometry** (the `0006` roundabout, where
BoT-SORT pulls clear at 0.802 vs ~0.69). It is reported as such rather
than averaged away.

Headline aggregate (clip=ALL, overall), raw → linked:

| tracker | HOTA | IDF1 | IDSW |
|---|---|---|---|
| ByteTrack | 0.823 → 0.891 | 0.801 → 0.871 | 634 → 495 |
| BoT-SORT | 0.820 → 0.881 | 0.808 → 0.873 | 680 → 539 |
| Deep OC-SORT | 0.776 → 0.855 | 0.745 → 0.846 | 2665 → 850 |
| OC-SORT | 0.763 → 0.845 | 0.725 → 0.835 | 3433 → 1062 |

Because all four trackers consume identical RT-DETR detections
(`DetA` ≈ 0.82 for all), the spread is pure association quality; the
linker's gain is overwhelmingly ID-switch repair (largest for the
weakest raw trackers). Caveat: BoT-SORT / Deep OC-SORT still ran with
the off-domain person-ReID encoder (`osnet_x0_25_msmt17.pt`); vehicle
ReID is a pending refinement that can only help the appearance trackers.

Engineering note: Deep OC-SORT required a runtime numerical-stability
patch (`src/trackers/boxmot_patches.py`) — BoxMOT's Kalman filter threw
a non-positive-definite Cholesky on dense scenes during its
freeze/unfreeze re-update; the patch projects the covariance onto the
nearest PSD matrix only when a factorization would otherwise crash.
Without it the benchmark would have been 3 trackers.

**Vehicle-ReID follow-up (2026-06-03) — encoder choice is immaterial
for top-down traffic.** The pending refinement noted above (replace the
person-ReID encoder for the two appearance trackers) was carried out:
the Stage 3 default ReID is now `clip_veri.pt` (a CLIP-ReID model
trained on VeRi-776, a multi-camera urban *vehicle* surveillance
dataset), run in FP16 (`--half`), set as the runner default in
`03_run_tracker.py`. BoT-SORT and Deep OC-SORT were re-run + re-linked +
re-scored; ByteTrack / OC-SORT are motion-only and unchanged. The
person-ReID baseline outputs are archived at
`data/video_annotations/tracker_outputs/_personreid_archive/` and the
person-ReID metrics are preserved in git at commit `6078260`.

Result — swapping the off-domain person encoder (`osnet_x0_25_msmt17`)
for the domain-matched vehicle encoder moved every aggregate metric by
≤ 0.003 (within noise):

| tracker / arm | HOTA (person → vehicle) | IDF1 (person → vehicle) |
|---|---|---|
| BoT-SORT / raw | 0.820 → 0.819 | 0.808 → 0.808 |
| BoT-SORT / linked | 0.881 → 0.881 | 0.873 → 0.874 |
| Deep OC-SORT / raw | 0.776 → 0.779 | 0.745 → 0.749 |
| Deep OC-SORT / linked | 0.855 → 0.856 | 0.846 → 0.846 |

Interpretation (a reportable finding): for top-down UAV traffic at
100–120 m, data association is **motion / IoU-dominated** — vehicles are
well-separated and move predictably, so frame-to-frame IoU resolves
nearly every match and the appearance embedding is a tiebreaker that
rarely changes the outcome (BoT-SORT's `appearance_thresh=0.25`
confirms appearance is a minor term). Overhead appearance is also weakly
discriminative — same-class vehicles look near-identical from directly
above — so there is little signal for *any* ReID to exploit. Practical
takeaway: invest in detection / motion quality, not appearance ReID.
`clip_veri` is nonetheless retained as the canonical config because it
is the domain-appropriate choice and pre-empts the "why a person model
on vehicles?" question; the null delta is reported as an observation.
The earlier (`6078260`) person-ReID numbers are therefore validated
rather than compromised.
---

