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
- **2026-06-12 — deconfounding pair launched: baseline vs GDP-B at the
  full 100-ep protocol (seed 0, patience 20).** This is the GDP keep/cut
  gate. Speed note for the eventual efficiency table: trained GDP-B
  inference 2.1 ms / postprocess 1.3 ms per image at 1280 (batch-1
  val) — P2's NMS cost is negligible after training.

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
