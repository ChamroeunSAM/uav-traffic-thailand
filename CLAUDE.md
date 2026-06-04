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
separation across the six. **Stage 2 is complete (all 4 clips annotated
in CVAT and exported to ground truth); Stage 3 is complete — the full
4-tracker × 4-clip dual (raw + linked) benchmark is scored, D3 resolved
to Option B. Stage 4 is complete — all four phases (A–D) done; the
trajectory-quality framework is run and the D4 hypothesis tested.** The
Paper 1 methodology arc is finished; what remains is paper-writing.

> **Next session starts here (2026-06-04):** **All five stages COMPLETE;
> the project is in the paper-assembly phase.** Beyond Stages 1–4, this
> session added: (1) a **GT-free generalisation test** — the Axis-3
> framework + self-consistency proxies scored two *new, unlabelled* 5-min
> 120 m intersection clips (`120_m_DJI_0001_5min`, `120_m_DJI_0041_5min`;
> hard-linked into `data/video_clips/`, GCP-calibrated) with NO ground
> truth → `results/stage4/trajectory_metrics_gtfree.csv`. **Result
> replicates the labelled benchmark with zero GT:** ByteTrack/BoT-SORT
> lead on self-consistency (seg/track ~2.1 vs ~3.7, teleport splits 80–94
> vs 200–207) *and* plausibility (raw accel-ok 0.86 vs 0.77, jerk 21 vs
> 26). (2) **Paper materials generated under `paper/`** — 7 booktabs tables
> (`scripts/make_paper_tables.py` → T1 dataset, T2 detectors, T3 per-class,
> T4 trackers, T5/T5b trajectory, T6 GT-free, T7 calibration) and 8 figures
> (`scripts/make_paper_figures.py` → F1 pipeline-spec text, F2 scene
> gallery, F3 detector accuracy-vs-FPS Pareto, F4 box-vs-trajectory, F5
> smoothing, F6a/F6b plausibility ECDFs, F7 roundabout spaghetti, F8
> calibration overlay, F9 GT-free bars). (3) **Duplicate-label bug fixed
> (decision_log D5)** — deduped GT + Stage-1 re-eval (numbers above).
> ★ NEXT (paper-writing, not pipeline): author drafts the results text
> around these tables/figures and syncs `docs/methodology.docx`; verify the
> inferred **scene-type labels** in F2/T1 (e.g. is 0001 a roundabout?);
> optional `docs/annotation_guide.md` stub. Optional figures not built: F3
> already done; none pending except author choices.

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
      mAP50-95 on the stratified test set (deduped GT — see decision_log
      D5; pre-dedupe values were ~0.003 lower, ranking identical):
        1. RT-DETR-L  0.9633 / 0.8756
        2. YOLO26-m   0.9579 / 0.8661
        3. YOLOv10-m  0.9536 / 0.8582
        4. YOLO11-m   0.9353 / 0.8433
        5. YOLO12-m   0.9345 / 0.8407
        6. YOLOv9-m   0.9333 / 0.8370
      Std across seeds ≤ 0.002 everywhere — ranking is stable.
      100 m beats 120 m by ~1.5-2 pts mAP50 universally. Per-class:
      RT-DETR-L wins every class; SV (motorcycles) is hardest at
      mAP50-95 (~0.75-0.79 vs ~0.88-0.95 LV/MV); YOLO11/12/v9-m tie on
      MV (the easy dominant class) and separate only on LV/SV (real,
      reproduced by native val — D5)
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
- [x] **Stage 2 pseudo-tracks generated** — RT-DETR-L + ByteTrack
      (`02_generate_pseudo_tracks.py`) + 3-pass offline linker
      (`02b_link_tracks.py`; "v3" = script defaults) over the 4 video
      clips; raw→linked track counts cut ~80% (intersection 958→218,
      etc.). Seeds at `data/video_annotations/pseudo_tracks/`. Note:
      the linker is a fragment-stitcher, NOT a tracker — it depends on
      ByteTrack's frame-to-frame association (verified empirically)
- [x] **CVAT annotation round-trip solved** —
      `02d_pack_cvat_mot.py` wraps each seed into a CVAT-importable
      MOT-1.1 zip (`gt/gt.txt` + `gt/labels.txt`, class shifted to
      1-based for Datumaro's `class_id−1` lookup);
      `02e_unpack_cvat_mot.py` converts the export back to
      repo-convention 0-based GT. Protocol fully rewritten
      (`docs/stage2_annotation_protocol.md` §6/§8/§10)
- [x] **Stage 2 human annotation complete** — all 4 clips corrected in
      CVAT and exported; the 4 `.final.cvat_mot.zip` exports are
      archived at `data/video_annotations/cvat_exports/` and unpacked to
      ground truth at `data/video_annotations/ground_truth/<clip>.gt.txt`
      via `02e_unpack_cvat_mot.py`. GT track counts: 0001=40, 0054=218,
      0006=61, 0047=115; all cover the full 1799-frame clip
- [x] **Stage 3 benchmark RUN on real data** — full 4-tracker × 4-clip
      dual (raw + linked) eval scored with TrackEval; results committed
      to `results/stage3/tracker_metrics.csv`. Aggregate post-linker:
      ByteTrack HOTA 0.891 / IDF1 0.871, BoT-SORT 0.881 / 0.873,
      Deep OC-SORT 0.855 / 0.846, OC-SORT 0.845 / 0.835. `DetA` ≈ 0.82
      for all (shared RT-DETR detections), so the spread is pure
      association; linking lifts every tracker (+0.06–0.11 IDF1) and
      collapses IDSW (OC-SORT 3433→1062). The evaluator was previously
      validated against synthetic GT (perfect→1.0; IDSW / FN /
      cross-sequence combine all correct)
- [x] **D3 resolved → Option B** (keep the comparative 4-tracker
      benchmark + dual eval). The designated clip `120_m_DJI_0006` keeps
      a wide post-linker IDF1 spread (0.69–0.80); see decision_log D3
      Resolution. Density-dependent finding: tracker choice compresses on
      dense free-flow, stays wide on sparse / complex geometry
- [x] **Deep OC-SORT crash fixed** — BoxMOT's KF threw a non-PD Cholesky
      on dense scenes (freeze/unfreeze re-update); reproducible runtime
      patch `src/trackers/boxmot_patches.py` projects the covariance onto
      the nearest PSD matrix only when a factorization would crash (no
      hyperparameter change, locked env untouched). Without it the
      benchmark would have been 3 trackers
- [x] **Stage 4 complete (Phases A–D).** Per-clip homographies + world
      trajectories (`src/trajectory/{calibration,kinematics,smoothing}.py`)
      feed the 3-axis trajectory-quality framework
      (`src/evaluation/trajectory_metrics.py`,
      `scripts/04_eval_trajectories.py`) →
      `results/stage4/trajectory_metrics.csv` (50 rows). D4 hypothesis
      tested: rankings agree (no KF-smoothing inversion); ByteTrack/
      BoT-SORT lead identity *and* plausibility. See decision_log D4
      Resolution.
- [x] Stage 1 benchmark commits pushed to `origin/master` (the
      benchmark is now backed up off the local machine)
- [x] **Stage 2/3 pipeline committed and pushed** (`cbfea1a`) — CVAT
      round-trip + full Stage 3 tracker pipeline now backed up on
      `origin`
- [x] boxmot version drift resolved — **not a real drift.** Installed
      `dist-info` is `18.0.0`, matching the lock pin; only boxmot's
      hardcoded `__init__.__version__` string lies ("17.0.0", upstream
      forgot to bump it). All 4 wrappers construct + `update()` cleanly
      on the installed package via `create_tracker` (low-level API
      intact in 18.0.0; signature accepts every kwarg we pass)
- [x] **Vehicle ReID adopted** — canonical Stage 3 ReID is now
      `clip_veri.pt` (VeRi-776 CLIP-ReID, FP16), the runner default in
      `03_run_tracker.py`. BoT-SORT / Deep OC-SORT re-run + re-linked +
      re-scored. **Null result:** vs the person-ReID baseline every
      aggregate metric moved ≤ 0.003 — association is motion/IoU-
      dominated from overhead and top-down appearance is weakly
      discriminative (decision_log D3 follow-up). Person-ReID outputs
      archived at `tracker_outputs/_personreid_archive/`; person-ReID
      metrics preserved at commit `6078260`. `--half` (FP16) also fixed
      a near-OOM / ~9× slowdown the FP32 CLIP encoder hit on dense clips
- [x] Per-class detection table built (`paper/tables/T3_per_class.tex`,
      via `make_paper_tables.py` from per-seed `eval_results.json`); the
      Stage 1 class-difficulty result is now reported
- [x] **Duplicate-label bug found and fixed (decision_log D5)** — source
      export had ~0.8% exact-duplicate boxes (~98% MV); `00_prepare_splits.py`
      now dedupes; 30 Stage-1 evals re-run on cleaned GT (+~0.003 mAP,
      ranking unchanged); weights untouched
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
RT-DETR-L + ByteTrack (`02_generate_pseudo_tracks.py`) then a 3-pass
offline linker (`02b_link_tracks.py`) produce pseudo-tracks over the 4
held-out clips. Seeds are packed for CVAT (`02d_pack_cvat_mot.py`), a
human corrects them per `docs/stage2_annotation_protocol.md` (5 passes),
and the export is converted to ground truth (`02e_unpack_cvat_mot.py`).
*Complete — all 4 clips annotated and exported; GT at
`data/video_annotations/ground_truth/`.* (SAHI tiling via
`yolo11_sahi_wrapper.py` remains an unused option for long sequences.)

**Stage 3 — Tracker benchmarking.** Four trackers — ByteTrack, BoT-SORT,
OC-SORT, Deep OC-SORT — run through one BoxMOT harness
(`03_run_tracker.py`, `src/trackers/*_wrapper.py`,
`configs/trackers/boxmot/*.yaml`) coupled to RT-DETR-L via "detect once,
track four times", and scored against Stage 2 GT with HOTA / MOTA / IDF1
via TrackEval (`src/evaluation/tracking_metrics.py`,
`03_eval_tracker.py`). Each tracker is evaluated twice — raw and
post-linker (decision_log D3, resolved to **Option B**). *Run on real
GT — results in `results/stage3/tracker_metrics.csv`. ByteTrack/BoT-SORT
lead post-linker (HOTA ≈ 0.89). A runtime KF patch
(`src/trackers/boxmot_patches.py`) keeps Deep OC-SORT from crashing on
dense scenes. ReID is `clip_veri.pt` (VeRi-776 vehicle encoder, FP16),
which changed metrics ≤ 0.003 vs the person-ReID baseline — a null
result (association is motion/IoU-dominated from overhead).*

**Stage 4 — Trajectory quality evaluation.** Paper 1's headline
contribution: a framework that evaluates trajectories — not just boxes
— along three axes (identity persistence, kinematic accuracy vs GT, and
intrinsic physical plausibility), averaged across clips. Pixel-to-world
conversion (`src/trajectory/calibration.py`), kinematic feature
extraction (`kinematics.py`), and smoothing (`smoothing.py`) feed
`src/evaluation/trajectory_metrics.py`. *Framing locked (decision_log
**D4**) — including why box metrics don't subsume trajectory quality and
a pre-registered Stage 3 hypothesis. **COMPLETE (Phases A–D).** Phase A:
per-clip ground-plane homography from road-marking GCPs (`00_calibrate.py`,
`calibration.py`, `docs/calibration_protocol.md`,
`configs/calibration/<clip>.gcp.yaml`) — all 4 clips fitted, scale
cross-validates (100 m ≈0.030, 120 m ≈0.040 m/px). Phase B: world
trajectories with smoothed kinematics (`smoothing.py` Savitzky–Golay,
`kinematics.py`, `04_extract_trajectories.py`) — splits at long gaps and
physically-impossible jumps. Phase C: the 3-axis metrics
(`trajectory_metrics.py`, `04_eval_trajectories.py`) with per-frame
greedy world-space GT↔pred matching, both raw and smoothed variants →
`results/stage4/trajectory_metrics.csv`. Phase D: hypothesis tested
(decision_log **D4 Resolution**) — rankings agree (the anticipated
KF-smoothing inversion did not occur); ByteTrack/BoT-SORT lead identity
and plausibility, so box metrics are a validated cheap proxy here.
Positional ADE ≈ 3 mm (GT shares detector provenance) — the Axis-2 signal
is in the differentiated speed/heading channels.*

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
│   ├── trackers/                ← Stage 2 (Ultralytics-format): botsort.yaml, bytetrack.yaml,
│   │                              deep_ocsort.yaml, ocsort.yaml
│   │       └── boxmot/          ← Stage 3 (BoxMOT-format): bytetrack, botsort, ocsort, deepocsort
│   ├── calibration/             ← Stage 4 GCP specs (committed): <clip>.gcp.yaml
│   └── experiments/             ← stage1_detector_benchmark.yaml, stage3_tracker_benchmark.yaml,
│                                  stage4_trajectory_eval.yaml
├── data/                        ← gitignored, ~10 GB locally
│   ├── annotations/
│   │   ├── 100m/                ← Roboflow YOLOv8 export (currently placeholder)
│   │   ├── 120m/                ← Roboflow YOLOv8 export (currently placeholder)
│   │   └── merged/              ← generated by 00_prepare_splits.py, regenerable
│   ├── splits/                  ← generated: dataset.yaml, altitude_index.json, split_metadata.json
│   ├── video_clips/             ← 4 source clips (~750 MB each, 4K @ 29.97 fps)
│   └── video_annotations/       ← Stage 2/3 working data (gitignored)
│       ├── pseudo_tracks/       ← linker seeds (.linked_v3.txt) + CVAT .cvat_mot.zip
│       ├── cvat_exports/        ← human-corrected CVAT exports (.final.cvat_mot.zip), archive
│       ├── ground_truth/        ← ★ corrected GT (<clip>.gt.txt) — Stage 3 input (all 4 clips)
│       ├── tracker_outputs/     ← Stage 3 tracker results (raw + .linked.txt) + detection cache
│       ├── calibration/         ← Stage 4: fitted homographies <clip>.json + frames/ (regenerable)
│       └── trajectories/        ← Stage 4: world trajectories + kinematics <name>.npz (regenerable)
├── docs/
│   ├── methodology.docx         ← living methodology reference (binary, in git)
│   ├── decision_log.md          ← AUTHORITATIVE record of architectural decisions
│   ├── machine_transfer.md      ← validated cross-machine setup procedure
│   ├── calibration_protocol.md  ← pixel-to-world calibration draft (Stage 4)
│   └── annotation_guide.md      ← TODO stub; informal guidelines only
├── runs/                        ← gitignored — training run outputs (weights, logs)
├── results/                     ← committed — aggregated metrics CSVs
│   ├── stage1/                  ← detector benchmark CSVs (+ inference_speed.csv)
│   ├── stage3/                  ← tracker_metrics.csv (raw + linked, per clip × tracker × scope)
│   └── stage4/                  ← trajectory_metrics.csv + trajectory_metrics_gtfree.csv
├── paper/                       ← manuscript assets (committed)
│   ├── tables/                  ← T1–T7 booktabs .tex (make_paper_tables.py)
│   ├── figures/                 ← F1 text spec + F2–F9 PNG/PDF (make_paper_figures.py)
│   └── references.bib
├── scripts/
│   ├── setup_environment.ps1                ← reproduce the venv from scratch
│   ├── 00_prepare_splits.py                 ← merge per-altitude Roboflow exports
│   ├── 00_calibrate.py                      ← Stage 4 calibration (scaffolded)
│   ├── 01_rename_with_altitude_prefix.py    ← prefix raw exports with altitude tag
│   ├── 01_train_detector.py                 ← train one detector from one config
│   ├── 01_eval_detector.py                  ← evaluate with altitude-stratified breakdown
│   ├── 02_generate_pseudo_tracks.py         ← Stage 2 pseudo-tracks (RT-DETR-L + ByteTrack)
│   ├── 02b_link_tracks.py                    ← Stage 2 3-pass offline track linker ("v3" seed)
│   ├── 02c_render_mot.py                     ← Stage 2 MOT-over-video visual QC renderer
│   ├── 02d_pack_cvat_mot.py                  ← Stage 2 seed → CVAT MOT-1.1 import zip
│   ├── 02e_unpack_cvat_mot.py               ← Stage 2 CVAT export → 0-based ground truth
│   ├── 03_run_tracker.py                    ← Stage 3 BoxMOT runner (detect-once, 4 trackers)
│   ├── 03_eval_tracker.py                   ← Stage 3 HOTA/MOTA/IDF1 (TrackEval CLI)
│   ├── 04_extract_trajectories.py           ← Stage 4 trajectory extraction (scaffolded)
│   ├── 04_eval_trajectories.py              ← Stage 4 trajectory-quality metrics (scaffolded)
│   ├── run_benchmark.py                     ← orchestrate full 6×5 Stage 1 sweep
│   ├── make_paper_tables.py                 ← emit paper/tables/*.tex from result CSVs
│   ├── make_paper_figures.py                ← emit paper/figures/* (per-figure CLI)
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

Last commits (refresh periodically):

```
33a9a62  Dedupe duplicate GT labels; re-eval Stage 1; add per-class table (T3)
6194520  Add F3 detector accuracy-vs-speed (Pareto) + FPS timing harness
36f394d  Add F2 scene gallery (6 clips, box overlays coloured by class)
4b38f3a  Add F8 calibration overlay (GCP quad + reprojected 10 m grid)
a9aacc6  Polish F4 (legend) and F7 (aspect)
a225077  Add F7 trajectory spaghetti (roundabout GT, speed-coloured)
5695f69  Add paper figure generator + F1 framework text, F5/F6a/F6b
fbd308e  Add paper table generator + emit T1/T2/T4/T5/T5b/T7 (LaTeX booktabs)
a95be39  GT-free generalisation on two 120 m clips: benchmark ranking holds
e4015cf  Add GT-free trajectory scoring (Axis-3 + self-consistency proxies)
671a2ba  Stage 4 Phase C: trajectory-quality metrics (3-axis D4 eval)
```

**Stage-1 dedupe + per-class table (`33a9a62`, 2026-06-04).** Found a
duplicate-label annotation bug (~0.8% exact-duplicate boxes, ~98% MV) via a
T3 cross-check; deduped in the pipeline + re-ran the 30 Stage-1 evals on
cleaned GT (decision_log **D5**). All detectors +~0.003 mAP, ranking
unchanged. Confirmed the MV per-class tie among YOLO11/12/v9-m is *real*
(native val reproduces it), not an artifact. Also recorded the supervision
vs native-val ~0.011 offset (implementation difference; supervision kept as
canonical, native used only as cross-check).

**Paper materials + GT-free generalisation (`e4015cf`…`6194520`,
2026-06-04).** Added `scripts/make_paper_tables.py` (7 booktabs tables →
`paper/tables/`) and `scripts/make_paper_figures.py` (8 figures →
`paper/figures/`), all reproducible from the committed CSVs/JSONs. Ran a
**GT-free** evaluation on two new unlabelled 5-min 120 m intersection clips
(Axis-3 plausibility + self-consistency proxies, no ground truth) →
`results/stage4/trajectory_metrics_gtfree.csv`; the labelled-benchmark
ranking (ByteTrack/BoT-SORT ahead of the OC-SORT family) **replicates with
zero GT** — the framework's reusability claim, demonstrated. New-clip
inputs (clips, GCP specs, fitted homographies) live under gitignored
`data/`; GCP specs are committed in `configs/calibration/`.

**Stage 4 Phases C+D — trajectory-quality framework (`671a2ba`,
2026-06-04). Stage 4 (and the Paper 1 methodology arc) is COMPLETE.**
Implemented the 3-axis D4 metrics (`src/evaluation/trajectory_metrics.py`,
`scripts/04_eval_trajectories.py`): Axis 1 identity persistence
(completeness, frags/GT, ID switches), Axis 2 kinematic accuracy vs GT
(ADE/FDE/RMSE m, speed MAE, heading deg), Axis 3 GT-free plausibility
(accel-in-bounds, jerk, curvature, speed). GT↔pred correspondence is
per-frame greedy nearest matching in world metres (gate 5 m); all metrics
reported for both raw and smoothed series. Ran over all 4 clips × 4 linked
trackers + the GT ceiling, auto-extracting the 15 missing trajectory
`.npz` on demand → `results/stage4/trajectory_metrics.csv` (50 rows).
Evaluator validated GT-vs-self (completeness/frag = 1.0, switches = 0,
ADE/speed/heading = 0). **D4 hypothesis outcome: the anticipated
KF-smoothing inversion did NOT occur — rankings agree.** ByteTrack/BoT-SORT
lead identity (completeness 0.95, switches 149/132) *and* plausibility
(raw `frac_accel_ok` 0.85 vs OC-SORT family 0.72; smoothed mean jerk 22 vs
28); OC-SORT's weaker association injects more kinematic noise than its KF
removes — so box metrics are a validated cheap proxy here. Positional ADE
≈ 3 mm across all trackers (GT shares RT-DETR detection provenance from
semi-automatic Stage 2 seeding) — the Axis-2 signal is in the
differentiated speed/heading channels, exactly D4's point. Full write-up:
decision_log **D4 Resolution**. NEXT: paper-writing only (final
trajectory-quality table + author sync of `methodology.docx`).

**Stage 4 Phases A+B (`7a763ea`, `c6cb22d`, `a180988`, 2026-06-03).**
Calibration: 4 per-clip homographies fitted from human GCPs (specs in
`configs/calibration/`, fitted JSON gitignored); `00_calibrate.py` flags
inconsistent points via a scale-consistency check (caught a transposed
coord on 0054); scale cross-validates by altitude. Extraction:
`smoothing.py` (Savitzky–Golay) + `kinematics.py` + `04_extract_trajectories.py`
turn MOT tracks into world-coordinate trajectories with speed / accel /
heading / jerk, splitting at long gaps and impossible jumps. Validated on
GT + a tracker output (plausible speeds; tracker matches GT). NEXT: Phase
C — `trajectory_metrics.py` + `04_eval_trajectories.py`.

**Stage 4 Phase A — calibration (`7a763ea`, 2026-06-03).** Stage 4 framing
locked as decision_log **D4** (3-axis trajectory quality; why box metrics
don't subsume it; pre-registered Stage 3 hypothesis + anticipated
KF-smoothing inversion). Implemented per-clip ground-plane homography
calibration: `src/trajectory/calibration.py`, `scripts/00_calibrate.py`
(`--dump-frame` + GCP fit with reprojection-RMSE + held-out scale check),
`docs/calibration_protocol.md`, and a GCP template
`configs/calibration/120_m_DJI_0006_1min.gcp.yaml`. Validated on a
synthetic homography (~1e-6 m) and the CLI fit path end-to-end; 4 reference
frames dumped (gitignored). **Blocked on human GCP entry** — fill the
`.gcp.yaml`, fit, then Phase B (extract/smooth/kinematics) + Phase C
(metrics). `smoothing.py`, `kinematics.py`, `trajectory_metrics.py`, and
the `04_*` scripts remain stubs.

**Vehicle ReID swap (`d5e4296`, 2026-06-03).** Canonical Stage 3 ReID is
now `clip_veri.pt` (VeRi-776 CLIP-ReID, FP16) — runner default in
`03_run_tracker.py`. BoT-SORT / Deep OC-SORT re-run + re-linked +
re-scored (`results/stage3/tracker_metrics.csv` regenerated; ByteTrack /
OC-SORT unchanged). **Null result:** vs the person-ReID baseline every
aggregate metric moved ≤ 0.003 — for top-down traffic, association is
motion/IoU-dominated and overhead appearance is weakly discriminative
(decision_log D3 follow-up has the delta table). `clip_veri` kept as the
principled config; the earlier (`6078260`) person-ReID numbers are
validated, not compromised. FP16 also fixed a near-OOM / ~9× slowdown on
dense clips. Person-ReID outputs archived under gitignored
`tracker_outputs/_personreid_archive/`.

**Stage 3 first real results (`6078260`, 2026-06-03).** Stage 2
annotation finished for all 4 clips and the full Stage 3 benchmark ran
end-to-end:

- **Stage 2 GT in hand** — 4 CVAT exports archived at
  `data/video_annotations/cvat_exports/`, unpacked to
  `data/video_annotations/ground_truth/` (track counts 40 / 218 / 61 /
  115; all full 1799 frames). *(GT + tracker outputs live under
  gitignored `data/`; not in the commit.)*
- **Benchmark results** — `results/stage3/tracker_metrics.csv` (160
  rows: clip × tracker × arm × scope). Post-linker leaders ByteTrack /
  BoT-SORT at HOTA ≈ 0.89; linking lifts all four (+0.06–0.11 IDF1).
- **Deep OC-SORT KF fix** — `src/trackers/boxmot_patches.py` (+ wired
  into `src/trackers/base.py`): PSD-projection guard for BoxMOT's
  non-PD Cholesky crash on dense scenes.
- **D3 resolved → Option B** — `docs/decision_log.md` D3 Resolution
  block; CLAUDE.md §2/§3 updated.

Pending before the paper table is final: per-class aggregation polish
(LV is N/A on `120_m_DJI_0006` — no GT large vehicles). (Vehicle ReID
swap is done — see below; the Stage 4 trajectory-quality build is now
done too — see the Phase C+D entry above.)

## 11. Document Maintenance

Last refreshed: **2026-06-04**

If today is more than one month past that date, sections 2 (Current
Status) and 10 (Recent Activity) may be stale. Other sections are more
stable but check whether major architectural decisions appear missing
versus `docs/decision_log.md`.
