# Paper 1 — Related Work (draft)

**Venue strategy:** see `abstract.md` (not locked; TR-C primary, T&F/Springer
portable). Citations in (Author, Year) form; full entries + verification
flags live in the appendix of `introduction.md` — not repeated here.
**Status:** first full draft — author to revise.

---

## 2. Related Work

We organise prior work into the four streams our contribution touches: (2.1)
UAV-based traffic trajectory datasets and extraction pipelines, (2.2) object
detection on aerial imagery, (2.3) multi-object tracking and its evaluation
metrics, and (2.4) trajectory quality and plausibility in traffic
engineering. Each subsection closes on the specific opening our work fills.

### 2.1 UAV-based traffic trajectory datasets and extraction pipelines

The drone's occlusion-free vantage has produced a now-dominant line of
naturalistic trajectory datasets. The Aachen family — highD on highways
(Krajewski et al., 2018), inD at urban intersections (Bock et al., 2020),
rounD at roundabouts (Krajewski et al., 2020), and exiD on highway merges
(Moers et al., 2022) — established drones as a precise, occlusion-free
trajectory sensor and remains the reference for German, lane-disciplined
traffic. At urban scale, the pNEUMA experiment flew a swarm of ten drones
over congested downtown Athens to collect roughly half a million vehicle
trajectories across more than a hundred intersections (Barmpounakis and
Geroliminis, 2020). More recent pipelines emphasise georeferencing and
scale: CitySim couples a five-step stabilise–filter–stitch–track–filter
pipeline to safety surrogates over US interchanges (Zheng et al., 2024),
while geo-trax registers detections to orthophotos to extract ~689k
trajectories from a ten-drone study over Songdo, South Korea, at the same
29.97 fps 4K regime as our footage (Fonod et al., 2025).

Two limitations recur. First, these are *dataset or single-pipeline*
contributions: each fixes one detector and one tracker and validates its
output ad hoc (typically positional georeferencing accuracy), rather than
comparing perception components or offering a transferable quality protocol.
Second, the traffic is Western and lane-disciplined (the Aachen family,
CitySim) or car-dominated planned grids (Songdo). The closest mixed-traffic
comparator, SIND, captures a single signalised intersection in China with
two-wheelers included (Xu et al., 2022), but is again a single-site dataset
rather than a perception benchmark. Dense, motorcycle-dominated Southeast
Asian traffic across multiple flight altitudes is absent from this
literature — the setting our benchmark targets.

### 2.2 Object detection on aerial and UAV imagery

Aerial detection research centres on two benchmarks. VisDrone provides 288
clips and over ten thousand images from fourteen Chinese cities across ten
categories (Zhu et al., 2022), and UAVDT contributes ~80k vehicle-centric
frames attributed by altitude, weather, and view (Du et al., 2018). Both are
foundational, but neither matches trajectory-grade traffic conditions:
VisDrone is captured at oblique and mixed viewing angles and folds
motorcycles into a crowded ten-class taxonomy, while UAVDT — though
vehicle-focused — annotates only cars, trucks, and buses, omitting the
motorcycle class entirely. The dominant vehicle type in Southeast Asian
traffic, and the hardest to resolve from altitude, is therefore
under-represented in exactly the benchmarks the field relies on.

On the model side, the detector landscape has shifted rapidly. The YOLO line
has progressed through programmable gradient information and GELAN in YOLOv9
(Wang et al., 2024a), NMS-free dual-assignment training in YOLOv10 (Wang et
al., 2024b), the Ultralytics YOLO11 and YOLO26 releases (Jocher and Qiu,
2024; Sapkota et al., 2025) — the latter's small-target-aware label
assignment (STAL) being directly pertinent to motorcycle detection — and the
attention-centric YOLO12 (Tian et al., 2025). In parallel, RT-DETR brought
the first real-time, end-to-end transformer detector, reporting that DETRs
can beat YOLOs at comparable speed (Zhao et al., 2024). A recurring mitigation
for small-object difficulty is slicing-aided inference such as SAHI (Akyon et
al., 2022), which we treat as the comparator to our native high-resolution
(1280×1280) choice rather than adopt. The bulk of applied UAV-detection
papers, however, improve a single architecture and report a single training
run on VisDrone or UAVDT — a fragile practice, since detection metrics carry
substantial run-to-run variance and single-run rankings overestimate
separation (Wang et al., 2020). Variance-aware, multi-seed comparison across
this modern detector generation, on controlled top-down traffic, has not been
reported; it is our first contribution.

### 2.3 Multi-object tracking and its evaluation metrics

Tracking-by-detection is the dominant paradigm for the trajectory stage, and
its evaluation rests on a mature metric family: the detection-oriented
MOTA/MOTP of CLEAR-MOT (Bernardin and Stiefelhagen, 2008), the
identity-oriented IDF1 (Ristani et al., 2016), and the now-standard HOTA,
which unifies detection, association, and localisation into a single
interpretable score (Luiten et al., 2021). The trackers we benchmark span the
current motion- and appearance-based families: ByteTrack associates every
detection box, including low-confidence ones, to recover occluded targets
(Zhang et al., 2022); BoT-SORT adds camera-motion compensation and an
improved Kalman state, relevant to moving-platform UAV video (Aharon et al.,
2022); OC-SORT introduces an observation-centric re-update to curb Kalman
drift under non-linear motion (Cao et al., 2023); and Deep OC-SORT augments
it with adaptive re-identification (Maggiolino et al., 2023).

The decisive limitation for our purposes is that *all* of these metrics
operate in image and box space. They reward detection recall, bounding-box
overlap, and identity bookkeeping, but none inspects the kinematics of the
trajectory a tracker emits. A tracker can lead on HOTA or IDF1 while
producing speed, acceleration, and jerk profiles that are physically
impossible, because association correctness and IoU are insensitive to
differentiated motion. Whether a box-level lead survives translation into
trajectory quality is, to our knowledge, untested — and is precisely the
question our framework poses.

### 2.4 Trajectory quality, smoothing, and plausibility in traffic engineering

The transportation community has long recognised that extracted trajectories
carry kinematic artefacts. Critical re-analyses of the NGSIM dataset showed
widespread physically implausible accelerations and oscillations (Punzo et
al., 2011; Coifman and Li, 2017), motivating accuracy-assessment procedures
based on speed–acceleration–jerk consistency and spectral checks (Punzo et
al., 2011) and multi-step reconstruction methods that remove infeasible
behaviour (Montanino and Punzo, 2015). The standard tool for estimating
velocity and acceleration from noisy position is the Savitzky–Golay
smoothing-differentiation filter (Savitzky and Golay, 1964), which we apply
and report alongside raw series. The choice of bounding-box representation
also bears on downstream trajectories: oriented boxes have been used to
improve trajectory consistency from drone imagery (Riehl et al., 2025),
whereas we deliberately adopt axis-aligned boxes (HBB) for top-down imagery,
where orientation is recoverable from trajectory direction and standard
tracking metrics are HBB-native (see decision log D1).

This literature, however, applies plausibility checks and reconstruction to
*validate or repair a single dataset*, disconnected from the box-level
metrics used to *select* a tracker. There is no reusable, tracker-agnostic
protocol that bridges identity-level tracking quality and kinematic realism.
Our trajectory-quality framework fills this gap: it decomposes quality into
identity persistence, kinematic accuracy against ground truth, and intrinsic
physical plausibility; it is applied comparatively across trackers; and, as
we show, it reproduces the box-level ranking even without ground truth —
making it a transferable evaluation tool rather than a one-off dataset audit.

### 2.5 Summary of gaps

Across these four streams, three gaps converge and define our contribution:
(1) no prior work pairs a variance-aware, multi-detector × multi-tracker
perception benchmark with a dedicated trajectory-quality framework; (2) dense,
motorcycle-dominated Southeast Asian traffic, observed top-down across two
altitudes, is absent from the nadir drone-trajectory literature; and (3) the
MOT-metric and traffic-engineering evaluation traditions remain disconnected,
with no reusable protocol testing whether box-level accuracy yields
trajectory-level quality.

---

## Drafting notes (not part of the section)

- **Citation hygiene:** all keys here are the same as the `introduction.md`
  appendix — verify the ⚠️-flagged ones (RT-DETR=CVPR2024; YOLO11/26 = no
  peer-reviewed paper, cite Ultralytics; BoT-SORT = preprint; YOLOv10/12 and
  inD/rounD/exiD/SIND venues) before camera-ready.
- **Overlap with intro:** §2 deliberately restates the three gaps more fully
  than §1.5; if a venue wants a shorter combined "Introduction and Related
  Work," merge §2.1–2.4 into §1 and keep only §2.5 as the gap summary.
- **Possible additions if length allows:** AU-AIR / CityFlow as
  sensor-modality contrasts (low-altitude oblique; fixed CCTV) — currently
  omitted as weak fits; VRUD (2026 preprint) only if its venue verifies.
- **HBB vs OBB (Riehl et al., 2025):** placed here as the natural spot to
  defend decision-log D1; expand to a sentence or two if a reviewer is likely
  to push on orientation.
