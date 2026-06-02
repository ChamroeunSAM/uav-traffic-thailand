# UAV Traffic Detection Benchmark — Project Context

## 1. What This Project Is

A UAV-based vehicle detection and trajectory benchmark for Thai traffic
scenes. The work is structured as a two-paper arc: **Paper 1** delivers a
six-detector benchmark plus a trajectory-quality evaluation framework
under top-down UAV imagery at two altitudes; **Paper 2** (future) applies
the resulting trajectories to traffic state-space modeling (SSM). Primary
target venue is *Transportation Research Part C* (Q1). All compute runs
locally on an owned RTX 5090 (32 GB) under Windows 11 — no cloud rental.

## 2. Current Status

5-stage methodology (see §3). **Stage 1 is complete — the detector
benchmark deliverable is in hand.** All six detectors × five seeds (30
trained models, 30 stratified test evaluations) committed to
`results/stage1/`. RT-DETR-L wins on every metric; tight three-tier
separation across the six. Stage 2-4 remain scaffolded but unvalidated.

- [x] Stage 0: data-preparation pipeline
      (`00_split_raw_export.py` → `01_rename_with_altitude_prefix.py`
      → `00_prepare_splits.py`) committed and exercised end-to-end
      on real annotations
- [x] **Real annotations split and merged** — 1497 (100m) + 1513
      (120m) = 3010 image/label pairs from
      `data/annotations/data_raw/`, partitioned 70/15/15
      (train=2107 / val=452 / test=451; seed=42), all files
      altitude-prefixed for provenance via `altitude_index.json`.
      Class distribution is heavily imbalanced: LV ≈ 2% of objects,
      MV ≈ 57%, SV ≈ 41% — but the full-data 100-epoch training
      learned LV cleanly (AP50 ≈ 0.95); the imbalance worry was
      overblown
- [x] **Stage 1 detector benchmark complete (5 seeds × 6 detectors,
      ~58 GPU-hours wall-clock).** Final ranking by overall mAP50 /
      mAP50-95 on the stratified test set:
        1. RT-DETR-L  0.9628 / 0.8747
        2. YOLO26-m   0.9545 / 0.8635
        3. YOLOv10-m  0.9514 / 0.8563
        4. YOLO11-m   0.9319 / 0.8399
        5. YOLO12-m   0.9312 / 0.8372
        6. YOLOv9-m   0.9300 / 0.8335
      Std across seeds ≤ 0.002 everywhere — ranking is stable.
      100 m beats 120 m by ~1.5-2 pts mAP50 universally
- [x] Stage 1 pipeline + supporting infrastructure
      (`01_train_detector.py`, `01_eval_detector.py`,
      `run_benchmark.py`); chunked-prediction OOM fix landed during
      yolov9m recovery
- [x] Environment locked (`requirements.lock.txt`, torch 2.11.0+cu128)
      and cross-machine setup hardened (`docs/machine_transfer.md`)
- [x] YOLO12m batch=4 ceiling confirmed empirically on 32 GB hardware
      (43.6 GB peak at batch=8 — physically does not fit); 22.3 GB
      peak at batch=4 confirmed in the full benchmark run
- [x] Altitude relabel (90 m → 100 m) propagated to forward-looking
      scripts, configs, and docs; historical commits/decision-log
      entries preserved as written
- [in progress] Stage 2-4 scripts scaffolded
      (`02_generate_pseudo_tracks.py`, `03_run_tracker.py`,
      `03_eval_tracker.py`, `04_extract_trajectories.py`,
      `04_eval_trajectories.py`, `00_calibrate.py`) but not yet
      pilot-validated end-to-end
- [in progress] Tracker wrappers exist (`src/trackers/{botsort,
      bytetrack, ocsort, deep_ocsort}_wrapper.py`); tracker configs
      committed (`configs/trackers/*.yaml`); not yet exercised
- [in progress] Trajectory utilities scaffolded
      (`src/trajectory/{calibration, kinematics, smoothing}.py`)
- [x] Stage 1 benchmark commits pushed to `origin/master` (the
      benchmark is now backed up off the local machine)
- [ ] Per-class mean/std aggregation — `aggregate_runs()` currently
      reports overall + 100m/120m only; per-class numbers live in
      each `runs/stage1_detectors/<run>/eval_results.json` and need a
      cross-seed roll-up for the paper's class-difficulty table
- [ ] Stage 2 kickoff: pseudo-track generation from the winning
      detector (RT-DETR-L) for tracker ground-truth construction
- [ ] Stage 2-4 execution
- [ ] `docs/annotation_guide.md` — currently a one-line TODO stub
- [ ] `docs/methodology.docx` — author must manually sync to current
      decision log + add the final 6-detector results table

**Observed during Stage 1 (worth noting for future long runs on this
machine):** occasional transient GPU faults under sustained day-long
load — `CUDA error: unknown error` killed yolo11m seed 3 at epoch 58
(recovered cleanly via `--seeds 3` re-run); mid-run wall-clock
slowdowns (~37% on rtdetr_l seeds 1-3) that self-recovered. None
affected training correctness, but exposure scales with run length.

## 3. Five-Stage Methodology

**Stage 0 — Data preparation.** Per-altitude Roboflow YOLOv8 exports
(`data/annotations/100m/`, `data/annotations/120m/`) are altitude-prefixed
(`01_rename_with_altitude_prefix.py`) then merged with a stratified
train/val/test split (`00_prepare_splits.py`) into `data/splits/`
containing a flat `dataset.yaml`, `altitude_index.json`, and
`split_metadata.json`. The merged dataset is the input to all later
stages; per-altitude evaluation is supported via `altitude_index.json`.

**Stage 1 — Detector benchmarking.** Six detectors × five seeds, trained
on the combined dataset and evaluated stratified by altitude (Strategy C,
see D2 in decision log). Training driver:
`scripts/01_train_detector.py`; evaluation:
`scripts/01_eval_detector.py`; sweep orchestrator:
`scripts/run_benchmark.py`. All metrics aggregated to
`results/stage1_*.csv`.

**Stage 2 — Semi-automatic video annotation for tracker evaluation.**
The best Stage 1 detector generates pseudo-tracks over held-out video
sequences (`02_generate_pseudo_tracks.py`); these are manually corrected
to produce tracker ground truth. SAHI tiling
(`yolo11_sahi_wrapper.py`) is the planned mechanism for high-recall
small-object detection on long sequences. *Scaffolded, not validated.*

**Stage 3 — Tracker benchmarking.** Four trackers — ByteTrack, BoT-SORT,
OC-SORT, Deep OC-SORT — coupled to the Stage 1 winner detector and
scored against Stage 2 ground truth with HOTA / MOTA / IDF1
(`src/evaluation/tracking_metrics.py`, `03_run_tracker.py`,
`03_eval_tracker.py`). *Configs committed; runs pending real data.*

**Stage 4 — Trajectory quality evaluation.** Paper 1's headline
contribution: a framework that evaluates trajectories — not just boxes
— along axes such as kinematic plausibility, smoothness, and identity
persistence. Pixel-to-world conversion (`src/trajectory/calibration.py`),
kinematic feature extraction (`kinematics.py`), and smoothing
(`smoothing.py`) feed `src/evaluation/trajectory_metrics.py`. Calibration
protocol drafted in `docs/calibration_protocol.md`. *Scaffolded.*

## 4. Key Architectural Decisions

(For full rationale and trade-offs, see `docs/decision_log.md` — it is
the authoritative record. Each line below is a summary.)

- **Six detectors at the medium size:** YOLOv9-m, YOLOv10-m, YOLO11-m,
  YOLO12-m, YOLO26-m, RT-DETR-L. *(decision_log: 2026-05-08, Stage 1
  scope.)*
- **Batch sizes asymmetric and fixed (not auto):** batch=8 for all
  except YOLO12m=4. The asymmetry is intrinsic — YOLO12m's
  area-attention peaks at 43.6 GB on batch=8 at 1280×1280, exceeding 32
  GB physical VRAM. Auto-batch is rejected because it confounds
  cross-detector comparison with stochastic batch-size variance.
  *(decision_log: 2026-05-08 batch sizing, 2026-05-16 5090 confirmation.)*
- **Image resolution 1280×1280:** chosen to retain small vehicles
  (motorcycles, distant cars) at 100 m and 120 m altitudes.
- **Two altitudes — 100 m and 120 m:** the dataset originally labeled
  "90 m" was a labeling error; footage was collected at 100 m. The 120 m
  set is correct as labeled. Forward-looking artifacts use 100 m; prior
  commits and earlier decision-log entries still say "90 m" and are
  **deliberately preserved** as accurate record of prior understanding.
  *(decision_log: 2026-05-18; commit a8fd0a5.)*
- **Three vehicle classes (class-ID order matters):**
  `0 = LV` (large vehicle: trucks / buses / trailers),
  `1 = MV` (medium vehicle: cars / vans / pickups / SUVs),
  `2 = SV` (small vehicle: motorcycles).
  The mapping is hardcoded in `scripts/00_split_raw_export.py`
  (`CLASS_NAMES`) and reflected in per-altitude `data.yaml` files. Note
  the ID order is large→small, not small→large; the `.txt` label files
  already encode the class IDs and must not be remapped. Annotation
  guidelines are informal — `docs/annotation_guide.md` is a TODO stub.
- **Axis-aligned bounding boxes (HBB), not OBB:** top-down imagery
  makes orientation recoverable from trajectory direction; HBB has
  first-class support across detector and tracker stacks; standard
  tracking metrics (HOTA/MOTA/IDF1) are HBB-native; articulated
  vehicles (trailers) would break a single rigid OBB. *(decision_log:
  D1.)*
- **Image-level train/val/test splits (Roboflow defaults):**
  defensible because frames are extracted at ~10 s intervals, so
  intra-clip leakage is minimal. No tracking-aware split is needed at
  the Stage 1 detector level (Stage 2-3 use separate held-out
  sequences).
- **Strategy C — combined training, stratified evaluation:** one
  detector is trained on the merged 100 m + 120 m set and reported
  separately per altitude. Maximizes data, yields a single deployable
  model, preserves per-altitude diagnostics. *(decision_log: D2.)*
- **Hardware: RTX 5090 (32 GB), Windows 11, native (not WSL):** runs
  the full 6×5 benchmark locally in well under a day. No GPU rental
  needed. *(decision_log: 2026-05-16.)*
- **RF-DETR dropped, RT-DETR-L added:** RF-DETR's training-time weight
  loader fails at imgsz=1280 with a position-embedding size mismatch
  (upstream issue #497, December 2025). Patching would create a
  maintenance fork. RT-DETR-L has no fixed-PE-grid constraint and trains
  cleanly at 1280 through Ultralytics' YOLO() auto-detection.
  `src/detectors/rfdetr_wrapper.py` is retained but unused in Stage 1.
  *(decision_log: 2026-05-08 Stage 1 scope.)*

## 5. Project Structure

```
uav-traffic-thailand/
├── CLAUDE.md                    ← this file
├── README.md
├── pyproject.toml
├── requirements.lock.txt        ← source of truth for env
├── requirements.txt
├── .gitignore                   ← /data/* and /runs/ ignored
├── .gitattributes               ← line-ending normalization
├── configs/
│   ├── data/                    ← dataset_combined.yaml, dataset_100m.yaml, dataset_120m.yaml
│   ├── detectors/               ← per-detector training configs
│   │                              (yolov9m, yolov10m, yolo11m, yolo12m, yolo26m, rtdetr_l;
│   │                               also pilot_yolo11n, yolo11_sahi, rfdetr — retained, unused in Stage 1)
│   ├── trackers/                ← botsort.yaml, bytetrack.yaml, deep_ocsort.yaml, ocsort.yaml
│   └── experiments/             ← stage1_detector_benchmark.yaml, stage3_tracker_benchmark.yaml,
│                                  stage4_trajectory_eval.yaml
├── data/                        ← gitignored, ~10 GB locally
│   ├── annotations/
│   │   ├── 100m/                ← Roboflow YOLOv8 export (currently placeholder)
│   │   ├── 120m/                ← Roboflow YOLOv8 export (currently placeholder)
│   │   └── merged/              ← generated by 00_prepare_splits.py, regenerable
│   └── splits/                  ← generated: dataset.yaml, altitude_index.json, split_metadata.json
├── docs/
│   ├── methodology.docx         ← living methodology reference (binary, in git)
│   ├── decision_log.md          ← AUTHORITATIVE record of architectural decisions
│   ├── machine_transfer.md      ← validated cross-machine setup procedure
│   ├── calibration_protocol.md  ← pixel-to-world calibration draft (Stage 4)
│   └── annotation_guide.md      ← TODO stub; informal guidelines only
├── runs/                        ← gitignored — training run outputs (weights, logs)
├── results/                     ← committed — aggregated metrics CSVs
├── scripts/
│   ├── setup_environment.ps1                ← reproduce the venv from scratch
│   ├── 00_prepare_splits.py                 ← merge per-altitude Roboflow exports
│   ├── 00_calibrate.py                      ← Stage 4 calibration (scaffolded)
│   ├── 01_rename_with_altitude_prefix.py    ← prefix raw exports with altitude tag
│   ├── 01_train_detector.py                 ← train one detector from one config
│   ├── 01_eval_detector.py                  ← evaluate with altitude-stratified breakdown
│   ├── 02_generate_pseudo_tracks.py         ← Stage 2 pseudo-track generation (scaffolded)
│   ├── 03_run_tracker.py                    ← Stage 3 tracker inference (scaffolded)
│   ├── 03_eval_tracker.py                   ← Stage 3 HOTA/MOTA/IDF1 (scaffolded)
│   ├── 04_extract_trajectories.py           ← Stage 4 trajectory extraction (scaffolded)
│   ├── 04_eval_trajectories.py              ← Stage 4 trajectory-quality metrics (scaffolded)
│   ├── run_benchmark.py                     ← orchestrate full 6×5 Stage 1 sweep
│   └── run_all_experiments.ps1              ← top-level driver (Stage 1+)
└── src/
    ├── detectors/
    │   ├── base.py
    │   ├── ultralytics_wrapper.py           ← handles all 5 YOLO variants + RT-DETR
    │   ├── rfdetr_wrapper.py                ← RETAINED but unused in Stage 1
    │   └── yolo11_sahi_wrapper.py           ← Stage 2 placeholder
    ├── trackers/
    │   ├── base.py
    │   └── {botsort,bytetrack,ocsort,deep_ocsort}_wrapper.py
    ├── data/                                ← conversion, dataset_loader, splits
    ├── evaluation/                          ← detection_metrics, stratified_eval,
    │                                          tracking_metrics, trajectory_metrics
    ├── trajectory/                          ← calibration, kinematics, smoothing
    └── utils/                               ← io, logging, visualization
```

## 6. Critical Workflows

### Launching the Stage 1 benchmark (data is already ingested)

The current dataset (3,010 frames, classes `[LV, MV, SV]`) is already
split, prefixed, and merged. The only remaining step is the sweep:

```
python scripts/run_benchmark.py
```

~30–45 hours wall-clock on the RTX 5090 for the full 6 detectors × 5
seeds = 30 runs. Outputs go to `runs/` (gitignored) and aggregated
metrics to `results/`.

### Ingesting a fresh Roboflow export (re-annotation, additional data)

If a new Roboflow export arrives (new version, expanded annotations,
additional altitudes), replay the data pipeline end-to-end. Each
script is dry-run by default; pass `--apply` to commit.

1. Drop the new Roboflow YOLOv8 export folders into
   `data/annotations/data_raw/` (replacing or augmenting
   `UAV_Thai_100m.v1i.yolov8/` and `UAV_Thai_120m.v1i.yolov8/`).
2. Partition into per-altitude train/valid/test (70/15/15, seed=42):
   - `python scripts/00_split_raw_export.py`         (dry-run)
   - `python scripts/00_split_raw_export.py --apply`
   - This script also clears the stale `merged/` and `splits/` derived
     artifacts so step 4 starts from a clean state.
3. Add altitude prefixes to filenames for cross-altitude provenance:
   - `python scripts/01_rename_with_altitude_prefix.py`         (dry-run)
   - `python scripts/01_rename_with_altitude_prefix.py --apply`
4. Merge into the unified training dataset:
   - `python scripts/00_prepare_splits.py --force`
5. Verify the merge summary:
   - Per-altitude image counts look right (100m vs 120m roughly
     comparable).
   - Class distribution sanity (LV is expected to be ~2%; if it
     collapses to 0% in any split, investigate before training).
   - `data/splits/dataset.yaml` exists with `nc=3`, `names=[LV, MV, SV]`.
6. Launch the sweep as above.

Caveats:
- `00_split_raw_export.py` hardcodes the class-ID → name mapping
  (`CLASS_NAMES = ["LV", "MV", "SV"]`) and ignores whatever `names:`
  the Roboflow export uses. If Roboflow ever reorders or expands the
  class set, update the constant before re-running.
- The split seed (42) is fixed for reproducibility. Re-running on the
  same source gives identical splits; changing source files reshuffles
  the assignment.
- Raw exports under `data/annotations/data_raw/` are never modified
  by the pipeline — they are the immutable archive.

### Cross-machine setup (new PC, rental, collaborator)

See `docs/machine_transfer.md` — validated procedure with explicit
warning about regenerating `dataset.yaml` / splits (absolute paths
don't survive transfer).

### When making any architectural choice

Read `docs/decision_log.md` first. It contains the rationale for every
settled question (HBB vs OBB, Strategy C, batch sizing, detector roster,
hardware) — including ones that might look tempting to re-open. The log
is append-only; entries are preserved even when they reflect prior
understanding (see the 90 m → 100 m correction note).

## 7. Environment

- Windows 11, native (not WSL)
- Python 3.12, virtualenv at `.venv/`
- PyTorch 2.11.0+cu128 (CUDA 12.8, sm_120 Blackwell support)
- RTX 5090, 32 GB VRAM
- `requirements.lock.txt` is the source of truth;
  `scripts/setup_environment.ps1` recreates the venv from scratch
- Reproducible install order: torch from the cu128 index **first**, then
  `pip install --no-deps -r requirements.lock.txt`
- Known dependency quirk: pip emits a metadata conflict between
  `transformers` (needs `regex>=2025.10.22`) and `boxmot` (metadata pin
  `regex<2025.0.0`). The lock pins `regex==2026.4.4`; boxmot's pin is
  metadata-only and harmless at runtime. See decision_log entry from
  2026-05-08 (dependency conflict).
- Ultralytics MLflow callback is disabled programmatically in
  `src/detectors/ultralytics_wrapper.py` — do not re-enable; it crashes
  on Windows paths and we don't use MLflow here.

## 8. Conventions for AI Assistants Working on This Project

### DO

- Read `docs/decision_log.md` before suggesting architectural changes.
- Use the existing wrapper architecture — `UltralyticsWrapper` handles
  all five YOLO variants *and* RT-DETR via the same Ultralytics `YOLO()`
  entry point; no per-detector branching is needed for Stage 1.
- Use atomic commits with a clear subject + body (multi-line via
  `-m -m`); follow the style in `git log`.
- Verify before assuming — `git status` before a commit, `--dry-run`
  before `--apply`, `Read` before `Edit`.
- Follow `docs/machine_transfer.md` exactly for fresh setups; regenerate
  splits with `--force` after any source change — never hand-edit
  generated artifacts in `data/annotations/merged/` or `data/splits/`.

### DO NOT

- Add `Co-Authored-By` trailers to commits — explicit project
  convention; AI disclosure lives in paper acknowledgments only.
- Rewrite git history. Existing commits and decision-log entries
  preserve historical understanding (notably the pre-correction "90 m"
  label) and are deliberately not edited.
- Modify `docs/methodology.docx` programmatically — author handles
  Word docs manually.
- Commit anything under `data/` or `runs/` — both are gitignored on
  purpose; respect that.
- Auto-install packages without asking, especially anything that might
  downgrade torch off the cu128 wheel (Blackwell support breaks).
- Treat "task complete" as a result without verification — surface
  evidence (output, file diff, command exit) rather than assertions.
- Use `batch=-1` (auto-batch) in benchmark configs. Fixed sizes are
  intentional per the documented batch-sizing decision; auto-batch
  contaminates cross-detector comparison.

## 9. What This Document Does NOT Cover

This file is a briefing for orientation, not a substitute for the
authoritative sources:

- Architectural decisions and rationale → `docs/decision_log.md`
- Full methodology narrative → `docs/methodology.docx` *(note: this
  file predates several decisions including the 90 m → 100 m
  correction; the decision log is the current source of truth where
  they conflict)*
- Cross-machine setup → `docs/machine_transfer.md`
- Environment recreation → `scripts/setup_environment.ps1`
- Pixel-to-world calibration → `docs/calibration_protocol.md`

## 10. Recent Activity

Last 5 commits (refresh periodically):

```
c4cd8e1  Add RT-DETR-L Stage 1 benchmark results -- Stage 1 complete
ec59852  Add YOLO26-m Stage 1 benchmark results (5 seeds)
81e5ee2  Add YOLO12-m Stage 1 benchmark results (5 seeds)
9394914  Add YOLO11-m Stage 1 benchmark results (5 seeds)
47c9e79  Add YOLOv10-m Stage 1 benchmark results (5 seeds)
```

## 11. Document Maintenance

Last refreshed: **2026-06-02**

If today is more than one month past that date, sections 2 (Current
Status) and 10 (Recent Activity) may be stale. Other sections are more
stable but check whether major architectural decisions appear missing
versus `docs/decision_log.md`.
