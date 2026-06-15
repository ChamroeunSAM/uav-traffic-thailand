# Design Doc — Custom Detector Paper ("Paper 1.5", working name Nadir-YOLO)

Drafted 2026-06-12. Companion to `docs/litscan_custom_yolo.md` (citation
base). **This is a living document, not a contract** — everything below is
a default with named alternatives; revise as pilots teach us things. When
we commit to building, add a decision-log entry recording the scope
(separate paper; Stage 1 roster and Paper 1 untouched).

---

## 1. One-paragraph thesis

Our benchmark (Paper 1) showed that on top-down UAV traffic imagery the
accuracy leader, RT-DETR-L, buys its win — concentrated almost entirely in
small-vehicle (motorcycle) AP50-95 (0.787 vs 0.747–0.771 for YOLO-m
models) — at ~2× the parameters (32.8M) and ~2.5× lower FPS (55 vs
138–161). We propose a YOLO-scale detector that closes the small-vehicle
gap at a fraction of that cost, built around a prior unique to our
setting: in nadir imagery at known altitude, ground-sample distance is
exact, so object pixel size is nearly deterministic per class. The model
conditions scale handling on the *known* GSD (the unclaimed sliver per the
lit scan: metric regression/assignment, not classification reweighting),
and is validated on VisDrone (genre headroom), our two-altitude Thai
benchmark (application + efficiency), and end-to-end in the Paper 1
tracking + trajectory-quality pipeline (system differentiation).

## 2. Relationship to existing work (fixed points)

- Paper 1 is untouched: no 7th detector, no re-runs, submit first.
- The new paper cites Paper 1 for: the benchmark gap (motivation), the
  dataset, the trajectory-quality framework (system section).
- Reused assets: dataset/splits, 6×5 benchmark results (comparison
  table), FPS/params/GFLOPs harness, Grad-CAM tooling, VisDrone pipeline
  from the domain-gradient ablation, Stage 3/4 tracking + trajectory
  pipeline, 5-seed protocol and `run_benchmark.py` orchestration.
- New repo for release: the paper needs a public GitHub repo (only ~2/15
  genre competitors release code). Internal development can live in this
  repo; the public repo is curated at the end.

## 3. Name

Working name **Nadir-YOLO** (free per lit scan; names the geometry that
makes the prior valid). Alternates if taste or a late collision says
otherwise: AGL-YOLO, Alti-YOLO, Ortho-YOLO, GSDA-YOLO. Re-check the
chosen name immediately before submission.

## 4. Base model

Default: **YOLO11-s**. Rationale: the genre publishes on n/s scales (the
lightweight story); YOLO11's Ultralytics implementation is the most
mature for custom-YAML surgery; our Stage 1 already characterizes the
YOLO11 family. Alternates: **YOLOv10-s first** (peer-reviewed NeurIPS
2024 base — a citability edge — and fastest/lightest in our benchmark;
disqualified as default only because its NMS-free dual-assignment head
is exactly where MSP operates), then YOLO26-s (same E2E-head objection,
newer code path), or -n scale if the efficiency story wants to be more
extreme. **Trigger conditions recorded 2026-06-12:** (1) if Phase 1
settles on the metric-REGRESSION mechanism (5.1b, no assignment
surgery), the dual-assignment objection mostly evaporates — run a
YOLOv10-s pilot, it costs one training run; (2) if Phase 2 FPS is
NMS-bound (P2 inflates candidates; the Phase 0 smoke already showed
15.7 ms postprocess vs 1.2 ms inference at 1280), that is a measured
reason to revisit NMS-free bases. Decide after Phase 1 pilots —
whichever base gets the modules working cleanly wins; do not burn time
forcing a stubborn codebase.

## 5. Module menu (centerpiece + supporting; final roster decided by ablation)

Genre convention wants 2–4 named modules, each ablatable. Plan: one
centerpiece carrying the novelty claim + 1–2 supporting modules from the
non-saturated bin. Anything that fails to earn its ablation row gets cut
— the roster is an output of the experiments, not an input.

### 5.1 Centerpiece — Metric Scale Prior (working acronym MSP)

The claim that survived the novelty check: use **known, exact GSD** to
condition **regression and/or assignment** (prior art — GSDet/GSDDet/
P2M-PF — estimates GSD and conditions classification; cite and
differentiate). Three candidate mechanisms, in order of preference:

- **(a) Metric assignment prior.** Modify the task-aligned assignment so
  scale matching uses expected pixel size = class physical-size prior ÷
  GSD (per-class size bounds: motorcycle ~0.7×2 m, car ~1.8×4.5 m, truck/
  bus ~2.5×8–12 m). Concretely: per-pyramid-level candidate gating (which
  level may claim which class) derived analytically instead of learned.
- **(b) Metric regression targets.** Predict box dimensions in metres
  (pixel dims × GSD as the regression space), normalizing targets across
  altitudes — a 100 m and a 120 m motorcycle become the same regression
  target.
- **(c) GSD-conditioned feature modulation** (FiLM-style scalar input).
  Closest to Xue et al. 2021 → weakest novelty; fallback only, or an
  ablation row showing (a)/(b) beat it.

GSD plumbing: per-image scalar from altitude metadata (our data: 0.030 /
0.040 m/px from Stage 4 calibration; the altitude prefix in every
filename gives it for free). On datasets without altitude metadata
(VisDrone) the module degrades gracefully: learned default scalar or
disabled — be explicit about this in the paper rather than hiding it.
NOTE (2026-06-12): `src/data/gsd.py` returns *native-4K* GSD; the module
must consume **effective GSD** = native GSD × (native size ÷ input size)
— at imgsz=1280 that is 0.090 (100 m) / 0.120 m/px (120 m). The resize
factor is known exactly per image, so the prior stays exact.

### 5.2 Supporting — GSD-Derived Pyramid (working acronym GDP)

Add P2, drop P5 — but *derive* the level selection analytically from GSD
× class-size bounds (at 0.03–0.04 m/px, no vehicle exceeds ~400 px;
motorcycles are 15–25 px → P2 needed, P5 provably idle). The
architecture matches the saturated P2/P5 recipe (cite SOD-YOLO etc.; do
not claim the recipe), but the analytic derivation is our framing and it
funds the params/FLOPs reduction that the lightweight claim needs.

### 5.3 Supporting — SV-targeted small-box loss

NWD-blended regression loss (common bin, not yet saturated) for tiny
boxes, possibly unified with MSP (a physically-normalized Wasserstein —
distance in metres rather than pixels — would tie the loss to the
centerpiece instead of reading as a checklist item). Optional extra:
DySample upsampler in the neck (common bin) if the P2 path needs it.

### 5.4 Explicitly NOT claimed as contributions

Bolt-on attention blocks, BiFPN variants, SPD-Conv, plain IoU-loss swaps
— saturated per the lit scan. Use only if an ablation shows real gain,
and present as adopted engineering with citations.

## 6. Evidence package (three legs)

1. **VisDrone-DET** — the genre-typical numbers leg. Headroom exists
   (baselines ~35–45 mAP50). Modules 5.2/5.3 apply directly; 5.1 runs in
   degraded mode (stated openly). Comparison vs base YOLO11-s + the
   published variants from the lit scan.
2. **Thai two-altitude benchmark** — the application + efficiency leg.
   Target: match/beat RT-DETR-L on SV AP50-95 at YOLO cost; full
   comparison table inherited from Paper 1 (note RT-DETR has no medium
   variant — state it, as Paper 1 does).
3. **Cross-altitude transfer (the experiment nobody else can run).**
   Train at 100 m only, test at 120 m (and reverse), with vs without
   MSP. If the metric prior is real, it absorbs the altitude shift
   analytically where the baseline degrades. This is the cleanest
   isolated test of the centerpiece and uses `altitude_index.json`
   stratification we already have. (It also answers the obvious reviewer
   objection: "your scale variance is near zero, why condition at all?")
4. *(System section, not a claims leg)* — plug the final model into the
   Stage 3/4 pipeline (detect → 4 trackers → trajectory quality) to show
   downstream viability; cross-cite Paper 1's framework.

## 7. Success criteria (pre-registered targets, with fallbacks — not pass/fail gates)

- **Primary:** SV AP50-95 ≥ 0.787 (RT-DETR-L) on the Thai test set, at
  ≤ ~20M params and ≥ 2× RT-DETR-L's FPS.
- **Secondary:** ≥ +2 mAP50 over base YOLO11-s on VisDrone without
  increasing params.
- **Tertiary:** measurable cross-altitude transfer gain attributable to
  MSP in the §6.3 experiment.
- **Hygiene:** every kept module contributes positively in the cumulative
  ablation (5 seeds, std reported), else it is cut.
- **Fallback framing** if primary falls short: accuracy-efficiency Pareto
  ("90% of the transformer's SV gain at 35% of the FLOPs") + the
  cross-altitude result. JPD-YOLO (ESWA genre) shows +0.29 mAP at half
  the params is publishable; we should not ship below that bar.

## 8. Phases (flexible; each ends with a keep/cut/pivot decision)

- **Phase 0 — scaffolding (~days).** Custom-YAML model path through
  `UltralyticsWrapper`; VisDrone-DET train/val loop (extend the
  domain-gradient ablation setup); confirm GSD scalar plumbing from
  filename prefix / metadata. No new architecture yet.
- **Phase 1 — pilots (~1–2 weeks, the learning phase).** Prototype MSP
  mechanisms (a)/(b) and GDP on the base model; 1-seed short-epoch runs
  on Thai + VisDrone subsets. Output: chosen base, chosen MSP mechanism,
  provisional module roster. Expect surprises here; this is where the
  doc gets revised.
- **Phase 2 — ablation matrix (~1 week incl. GPU time).** Cumulative
  ablation (base → +GDP → +MSP → +loss), 5 seeds each, Thai (stratified)
  + VisDrone; cross-altitude transfer experiment; FPS/params/GFLOPs via
  the existing harness. ~20–30 training runs ≈ 1.5–2 days GPU on the
  5090 per dataset — same order as Stage 1, well within local capacity.
  (Mind the known transient-GPU-fault exposure on long sweeps; the
  benchmark orchestration's per-seed recovery already handles this.)
- **Phase 3 — system + figures (~days).** Final model through Stage 3/4
  pipeline; Grad-CAM (existing tooling); Pareto figure (F3 pattern).
- **Phase 4 — manuscript + public repo.** Venue default ESWA;
  alternates: Measurement, Remote Sensing, IEEE T-ITS, J-STARS.

Sequencing constraint: none of this blocks Paper 1 — paper-writing for
Paper 1 continues in parallel and submits first.

## 8b. Phase 1 pilot log (running record; newest last)

- **2026-06-12 — pilot pair 1, Thai val, 30 ep, seed 0, imgsz 1280.**
  Baseline scaffold (= YOLO11s, batch 8): mAP50-95 all 0.816 / LV 0.893 /
  MV 0.874 / SV 0.681. **GDP-A (P2-P4, batch 3): all 0.767 / LV 0.816 /
  MV 0.847 / SV 0.637 — CUT.** The LV canary fired (-0.077): the size
  analysis used LV <= 12 m, but LV includes trailers (16-18 m ~= 200 px
  at 100 m → P5 is required). SV -0.044 is confounded (head + SPPF/C2PSA
  from scratch, 128/409 transfer). Positives: -70% params real; trained-
  model P2 NMS cost negligible (1.0 ms postprocess — the Phase 0 15.7 ms
  reading was an untrained-model artifact).
  *Memory finding:* P2 at 1280 spikes with per-frame instance count
  (mosaic >1200 objects): batch 8 → 45 GB, batch 4 → 40+ GB mid-epoch
  (silent WDDM spillover, erratic ~6x slowdowns). Fix: `cuda_mem_fraction:
  0.9` wrapper guard (fail fast) + batch 3 (15.4 GB steady, stable
  throughput). All P2-bearing variants train at batch 3.
- **2026-06-12 — GDP-B (stock P3-P5 + added P2; 9.58M params, 115.8
  GFLOPs @1280; 297/593 transfer), 30 ep:** all 0.788 / LV 0.841 /
  MV 0.859 / SV 0.663 — below baseline on every class, but *uniformly*
  (~0.02-0.05), the signature of the from-scratch descend path rather
  than a P2-specific failure. Restoring the pyramid recovered most of
  GDP-A's losses. NOT cut yet — confounded by initialization at 30 ep.
  Alternative hypothesis on record: SV AP50 is already 0.958 (detection
  fine; strict-IoU localization is the gap) and P2 may add no usable
  information at 17-22 px object scale — if the 100-ep pair confirms,
  GDP is cut entirely and MSP becomes the sole architectural centerpiece.
- **2026-06-12 — deconfounding pair (baseline vs GDP-B, full 100-ep
  protocol, seed 0, patience 20) launched then TERMINATED mid-run at the
  author's request (session break). ★ RELAUNCH IS THE NEXT ACTION:**
  `python scripts/01_train_detector.py --config configs/detectors/nadir_yolo_s.yaml --name nadir_pilot_base_100ep`
  then the same with `nadir_yolo_s_gdp_b.yaml` / `nadir_pilot_gdp_b_100ep`
  (~3 h total; a partial `runs/nadir_yolo/nadir_pilot_base_100ep` dir may
  exist — safe to overwrite, exist_ok is set). This is the GDP keep/cut
  gate. Speed note for the eventual efficiency table: trained GDP-B
  inference 2.1 ms / postprocess 1.3 ms per image at 1280 (batch-1
  val) — P2's NMS cost is negligible after training.
- **2026-06-15 — deconfounding pair COMPLETE → GDP CUT (gate resolved).**
  Both ran the full 100 ep (seed 0, patience 20 did not trigger; baseline
  exit 17:01, GDP-B exit 18:49, ~110 min @ batch 3 + ~65 s/ep). Uniform
  re-val of both `best.pt` (imgsz 1280, batch 4, val split, workers 0 —
  Windows spawn guard), so the deltas are apples-to-apples:

  | run | all 50-95 | mAP50 | LV 50-95 | MV 50-95 | **SV 50-95** |
  |-----|-----------|-------|----------|----------|--------------|
  | baseline | 0.8405 | 0.9442 | 0.8962 | 0.8940 | 0.7312 |
  | GDP-B | 0.8372 | 0.9472 | 0.8858 | 0.8900 | **0.7358** |
  | Δ | **−0.0033** | +0.0029 | −0.0104 | −0.0040 | **+0.0046** |

  **The 30-ep story REVERSED once deconfounded.** At 30 ep GDP-B trailed on
  SV (0.663 vs 0.681, −0.018); at 100 ep GDP-B **beats** baseline on SV
  (+0.0046) — so the §8b alternative hypothesis ("P2 adds no usable info at
  17–22 px") is **refuted**: P2 *does* help small vehicles, the 30-ep deficit
  was pure from-scratch initialization. BUT the SV gain is small (+0.005) and
  paid for by LV (−0.0104) and MV (−0.0040), netting −0.0033 overall (GDP-B is
  ahead only on loose-IoU mAP50). Decisive against keeping: GDP-B costs
  **~115.8 GFLOPs vs ~21.6 baseline (≈5×)** for **no params saving** (9.58M vs
  9.41M) — a poor trade for a net-flat overall mAP. **DECISION: cut GDP from
  the architecture; record as an ablation finding (P2 helps SV but not at 5×
  FLOPs). MSP becomes the sole architectural centerpiece** (decision_log D6
  resolution; §8b contingency executed). Note absolute "all" here (0.8405)
  exceeds the in-training-logged 0.8264 by a uniform re-val offset; only the
  Δ is load-bearing. ★ NEXT ACTION: build the MSP metric-prior prototype
  (§5.1, mechanism (b) metric-regression first; if (b) is chosen, trigger the
  YOLOv10-s base pilot per §4).

## 8c. MSP prototype plan (Phase 1 centerpiece; started 2026-06-15)

GDP is cut (§8b), so MSP carries the novelty. Mechanism **(b) metric
regression** is the chosen starting point (§5.1) → per §4 trigger (1) the
dual-assignment objection evaporates, so a **YOLOv10-s base pilot is now
unlocked** (one training run, peer-reviewed base, citability edge).

**Done so far.** `src/data/gsd.py` gains `effective_gsd_m_per_px(name, imgsz)`
= native × (3840 / imgsz). Verified on all 2107 train files: 100 m → 0.090,
120 m → 0.120 m/px @1280 (matches §5.1). This is the exact per-image scalar
every MSP mechanism consumes; degrades to `default`/None on names without an
altitude prefix (VisDrone).

**The core idea restated.** A motorcycle is ~2 m of road regardless of
altitude, but at imgsz 1280 it is ~22 px at 100 m vs ~17 px at 120 m — so the
*pixel* regression target for the same physical object differs by altitude.
MSP-(b) removes that altitude dependence by making the regression space
metric (or, equivalently, GSD-normalized).

**Three implementation routes (decreasing simplicity, increasing fidelity) —
DECISION FORK, not yet chosen:**
- **(i) GSD-equalizing resize (preprocessing; NO internals surgery).** Resize
  each image so effective GSD is a fixed constant (e.g. 0.090 m/px for all),
  i.e. downscale 120 m images by 0.090/0.120 = 0.75 before the 1280 letterbox.
  Then a physical object is the same pixel size at both altitudes and *stock
  pixel regression already is metric regression*. Cleanest prototype, no DFL
  surgery, locked env untouched (a dataloader/transform hook). Risk: changes
  effective input scale per altitude; interacts with mosaic/letterbox; must
  confirm it doesn't just throw away 120 m resolution.
- **(ii) Metric DFL target rescaling (loss surgery; faithful "predict in
  metres").** Transform anchor→box ltrb targets to metres via per-image
  eff-GSD before DFL binning, decode back to px at inference. Snag: DFL has a
  fixed bin range (reg_max=16 in stride units) — metric targets must be
  renormalized into that range, so this reduces to a per-image scale on the
  regression coordinates. Runtime-patch `v8DetectionLoss` (boxmot_patches.py
  pattern); locked env untouched.
- **(iii) Metric loss-weighting (lightest touch).** Keep pixel regression but
  weight the box/DFL loss per-image by eff-GSD so metric error (m) is the
  optimized quantity. Weakest novelty; likely an ablation row, not the lead.

**Recommended prototype path:** start with **(i)** — it tests the central
hypothesis (altitude-normalized scale helps) with the least machinery and no
internals patch, and it cleanly answers "does removing altitude scale variance
help SV?" If (i) shows signal, escalate to (ii) for the paper's headline
mechanism and to claim metric regression proper; (iii) is the fallback/ablation.
**Open question for the author before coding (ii):** is the claim "regress in
metres" (needs ii) or "GSD-invariant detection" (i suffices)? The novelty
sliver in D6 is *known-exact-GSD conditioning of regression* — (ii) is the
literal realization; (i) is a strong, honest first probe.

**Smoke + gate.** Prototype on Thai val, 30 ep, seed 0, vs the just-finished
`nadir_pilot_base_100ep` baseline (re-val numbers in §8b: all 0.8405 / LV
0.8962 / MV 0.8940 / SV 0.7312). MSP earns its keep only if it lifts SV
mAP50-95 without an LV/MV regression that nets negative — the same bar GDP
failed. Cross-altitude transfer (100 m→120 m) is the designed showcase (§6
leg); a flat fixed-GSD result still feeds the Pareto/benchmark framing (§9).

- **2026-06-15 — route (i) A/B RESULT → probe SUCCEEDED, escalate to (ii).**
  Built a GSD-equalized dataset (`scripts/05_make_gsd_normalized_dataset.py`:
  120 m centre-cropped to the 100 m scale, 74.8% of 120 m boxes kept) + a
  matched plain-1280×720 control (no crop). Two 30-ep smokes (seed 0, stock
  YOLO11s scaffold), each scored on its own matched val. Per-class mAP50-95:

  | scope | all | LV | MV | **SV** |
  |-------|-----|----|----|--------|
  | msp_i  all  | 0.8335 | 0.8780 | 0.8904 | 0.7320 |
  | ctrl_i all  | 0.8266 | 0.8941 | 0.8889 | 0.6967 |
  | **Δ all**   | **+0.0069** | −0.0161 | +0.0015 | **+0.0353** |
  | Δ 100 m | −0.0015 | −0.0036 | +0.0007 | −0.0015 |
  | Δ 120 m | **+0.0147** | −0.0322 | −0.0003 | **+0.0765** |

  **Read:** 100 m Δ≈0 is the built-in control (100 m frames are identical in
  both pipelines → must match, and do — harness validated). The action is all
  at 120 m, where GSD-equalization acts: **SV +0.0765** (scale-normalizing the
  coarser-altitude motorcycles up to the finer reference helps exactly the
  smallest objects — the MSP hypothesis confirmed); LV −0.0322 (large vehicles
  don't need it, and the crop costs some). **Caveat capping the magnitude, not
  the direction:** msp_i's 120 m val is cropped (~25% edge boxes gone), so its
  val is genuinely easier → the +0.0765 is optimistic, and route (i) has an
  inherent cost (edge objects cropped out at inference too). **Decision: route
  (i) is a successful PROBE, not the final mechanism. Escalate to route (ii)
  (metric DFL regression) — same scale invariance, no cropping, no val
  asymmetry, and the literal D6 novelty claim.** Datasets/configs:
  `configs/data/dataset_{gsdnorm,plain1280}.yaml`,
  `configs/detectors/nadir_{msp,ctrl}_i_30ep.yaml`.

## 8d. MSP route (ii) — metric DFL regression (started 2026-06-15)

Goal: make the box-size regression target metric (GSD-normalized) so a 100 m
and a 120 m motorcycle present the SAME regression target, WITHOUT cropping.
Realized as a runtime patch of Ultralytics' `v8DetectionLoss` (boxmot_patches
pattern; locked env untouched), keyed off the per-image effective GSD from
`src/data/gsd.effective_gsd_m_per_px`. (Running record; newest last.)

- **2026-06-15 — internals analysis: pure metric-target DFL is NOT loss-only
  learnable (corrects the §4 assumption).** Read `v8DetectionLoss.bbox_loss`
  (`utils/loss.py`): CIoU is scale-invariant (global GSD scaling is a no-op
  there), so GSD can only act on the DFL target (`bbox2dist`, distances in
  stride-unit bins [0,15]). But making the DFL target altitude-invariant
  (a 2 m bike → same target at 22 px @100 m and 17 px @120 m) asks the network
  to emit the SAME distribution from DIFFERENT pixels — only possible if it
  KNOWS the GSD. Altitude isn't reliably inferable from a local nadir patch,
  so a metric-target loss patch ALONE would train toward a blurry average.
  **Conclusion: faithful metric regression (b) requires conditioning the
  forward pass on the known GSD (b necessarily entails a c-like input).** Three
  viable realizations, decision pending:
  - **(ii-A) GSD input channel + metric DFL target.** Append a per-image
    GSD-broadcast plane as a 4th input channel (first conv goes 3→4 ch;
    pretrained conv1 partially transfers), and GSD-normalize the DFL target.
    Most faithful to "known-exact-GSD conditions regression" (the D6 sliver),
    biggest change.
  - **(ii-B) FiLM scalar conditioning + metric DFL target.** Inject GSD as a
    FiLM (scale/shift) on an early feature map via a tiny MLP; metric DFL
    target as above. Closest to Xue et al. 2021 (cite/differentiate); medium
    change, no first-conv edit.
  - **(ii-C) Metric loss-weighting (loss-only, IS learnable).** Keep pixel
    targets; weight box/DFL loss per-image by GSD so physical (metre) error is
    optimized consistently across altitudes. Pure runtime patch, quickest
    smoke, weakest novelty — a legitimate ablation row, not the headline.
  Recommendation: smoke **(ii-C)** first (one runtime patch, immediate read on
  whether GSD-weighted regression moves SV at 120 m), then build **(ii-A)** as
  the headline if (C) or route (i) justify it.
- **2026-06-15 — (ii-A) chosen; GSD-input-channel plumbing BUILT + smoke-passed.**
  `src/detectors/msp_patches.py`: `MSPTrainer` (builds the model with ch=4 and
  seeds the first conv's RGB sub-weights from the 3-ch pretrained checkpoint,
  4th/GSD channel zero-init → transfer preserved) + `MSPValidator`; both append
  a per-image GSD plane (`effective_gsd_m_per_px(im_file, imgsz)`) in
  preprocess. Wired via `ultralytics_wrapper` `msp_mode: ii_a` (runtime
  subclass only — stock runs untouched). Two Ultralytics-internals snags found
  and fixed: (1) import path is top-level (`data.gsd`, `.msp_patches`), not
  `src.*`; (2) `final_eval` reloads best.pt through AutoBackend, whose warmup
  uses a `self.data["channels"]`-channel dummy → needs `channels: 4` in the
  data yaml (`configs/data/dataset_msp4ch.yaml`); the loader still reads 3-ch
  (`IMREAD_COLOR` branches only on grayscale), so the GSD channel is added in
  preprocess as intended. 1-epoch/10%-data smoke ran train + during-train val +
  final best.pt val with no error (exit 0). **STATUS: GSD-conditioning path
  validated; metric-DFL-target half (gated `msp_metric_target`) NOT yet built.
  ★ NEXT: (a) 30-ep A/B — `nadir_msp_iia` (channel only) vs the baseline/control
  — to read whether GSD-conditioning alone moves SV@120 m; (b) add the metric
  DFL target on top.** Smoke config: `configs/detectors/nadir_msp_iia_smoke.yaml`.

## 9. Risks and honest unknowns

- **MSP may show ~nothing on fixed-GSD data** (conditioning helps when
  scale varies). Mitigation: the cross-altitude experiment is designed
  to surface its value; if even that is flat, the paper pivots to the
  efficiency/Pareto + benchmark-setting story and MSP becomes an
  ablation finding, not the centerpiece.
- **Assignment surgery is deeper than YAML.** Mechanism (a) touches
  Ultralytics' TAL/loss internals — version-pin and isolate patches
  (the `boxmot_patches.py` pattern: runtime patch, locked env
  untouched). Mechanism (b) is shallower; start there if (a) fights us.
- **Near-ceiling Thai data** caps visible gains there by construction —
  that is why VisDrone carries the headline numbers leg.
- **Reviewer "incremental" risk** — the genre's chronic rejection mode.
  Mitigations already designed in: one genuinely differentiated
  centerpiece, the unmatched benchmark setting, the system section, open
  code, 5-seed rigor with std (almost nobody in the genre reports seeds).
- **Name/novelty drift** — the genre publishes weekly; re-run the name
  collision + closest-prior-art check before submission.

## 10. Immediate next actions (when green-lit)

1. Decision-log entry (new paper scoped; Stage 1/Paper 1 untouched).
2. Phase 0 scaffolding.
3. Re-verify VisDrone-DET licensing/version for training + publication.
