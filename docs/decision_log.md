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

