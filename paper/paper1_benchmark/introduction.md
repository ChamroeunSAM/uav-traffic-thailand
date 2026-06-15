# Paper 1 — Introduction (draft)

**Venue strategy (not locked):** primary target *Transportation Research
Part C* (Elsevier, Q1), but kept portable to Taylor & Francis (*J. of
Intelligent Transportation Systems*, *Transportmetrica*, *Int. J. of Remote
Sensing*) and Springer (*Machine Vision and Applications*, *Neural Computing
and Applications*, *Int. J. of ITS Research*). The intro below is written for
a transportation/ITS-general audience so it ports across publishers; per
venue we re-weight emphasis (benchmark-forward vs. trajectory-quality-forward)
rather than rewriting. See `abstract.md` for the full venue list.

**Status:** first full draft — author to revise. Citations in (Author, Year)
form; full entries + verification flags in the appendix at the bottom.

---

## 1. Introduction

### 1.1 Motivation

Unmanned aerial vehicles (UAVs) have become a transformative sensor for
traffic monitoring. Hovering above a scene, a drone observes vehicles from
a near-vertical viewpoint that is largely free of the inter-vehicle
occlusion and perspective foreshortening that limit roadside cameras, and
it can cover a full intersection or motorway segment with a single device.
This capability has produced a new generation of naturalistic vehicle
trajectory datasets — highD and its intersection, roundabout, and merging
siblings for German roads (Krajewski et al., 2018; Bock et al., 2020;
Krajewski et al., 2020; Moers et al., 2022), the pNEUMA drone-swarm
experiment over downtown Athens (Barmpounakis and Geroliminis, 2020), and
more recent georeferenced pipelines such as Songdo/geo-trax (Fonod et al.,
2025) and CitySim (Zheng et al., 2024). These trajectories are the raw
material for traffic-flow theory, safety surrogate analysis, and
data-driven state-space modelling — the second paper of the present arc
applies them to exactly this end.

The value of a UAV trajectory dataset, however, rests entirely on a
perception pipeline that must (i) **detect** every vehicle in each frame,
(ii) **track** each vehicle's identity across frames, and (iii) project the
result into **world coordinates** to yield a usable trajectory. Each stage
is an active research area with its own benchmarks and metrics, yet they
are almost never evaluated together, and three gaps recur.

### 1.2 Gap 1 — detector evaluation is narrow and variance-blind

Aerial object detection is dominated by two benchmarks, VisDrone (Zhu et
al., 2022) and UAVDT (Du et al., 2018). Both are valuable, but neither
matches the conditions of trajectory-grade traffic monitoring: VisDrone is
captured at oblique and mixed viewing angles and folds motorcycles into a
crowded ten-class taxonomy, while UAVDT, though vehicle-centric and
altitude-attributed, contains only cars, trucks, and buses — it has **no
motorcycle class at all**. The downstream consequence is that the vehicle
type that dominates Southeast Asian traffic, and that is hardest to detect
from altitude, is effectively untested. Furthermore, the typical UAV
detection paper improves a single architecture and reports a single
training run on one of these two datasets. This is methodologically fragile:
detection metrics carry substantial run-to-run variance (Wang et al.,
2020), and the inter-detector gaps among modern architectures are often
smaller than that variance. A credible ranking of today's detectors —
which now span both YOLO-line models (Wang et al., 2024a; Wang et al.,
2024b; Jocher and Qiu, 2024; Tian et al., 2025; Sapkota et al., 2025) and
real-time transformers (Zhao et al., 2024) — requires variance-aware,
multi-seed evaluation that the literature rarely provides.

### 1.3 Gap 2 — tracking metrics measure boxes, not trajectories

Multi-object tracking is evaluated with a mature family of metrics —
MOTA/MOTP (Bernardin and Stiefelhagen, 2008), IDF1 (Ristani et al., 2016),
and the now-standard HOTA (Luiten et al., 2021) — and strong trackers such
as ByteTrack (Zhang et al., 2022), BoT-SORT (Aharon et al., 2022), OC-SORT
(Cao et al., 2023), and Deep OC-SORT (Maggiolino et al., 2023) optimise
against them. Critically, **all of these metrics operate in image/box
space**: they reward detection recall, bounding-box overlap, and identity
bookkeeping. None inspects the *kinematics* of the trajectory a tracker
ultimately produces. A tracker can therefore lead on HOTA or IDF1 while
emitting speed, acceleration, and jerk profiles that are physically
impossible, because association correctness and box IoU are insensitive to
differentiated motion. The traffic-engineering community has long known
that extracted trajectories carry such artefacts — critical re-analyses of
NGSIM revealed widespread implausible accelerations (Punzo et al., 2011;
Coifman and Li, 2017), and similar anomalies are documented in drone data —
and has developed plausibility checks and reconstruction methods (Punzo et
al., 2011; Montanino and Punzo, 2015; Savitzky and Golay, 1964 for the
smoothing-differentiation step). But these are applied to *validate or
repair a single dataset*, disconnected from the box-level metrics used to
*choose* a tracker in the first place. The assumption that bridges the two
communities — that better detection and tracking yield better
trajectories — is widely held and almost never tested.

### 1.4 Gap 3 — the setting: top-down, two-altitude, motorcycle-dense traffic

The dominant trajectory datasets are Western and lane-disciplined (the
Aachen highD family; CitySim) or car-dominated planned grids (Songdo),
while the closest mixed-traffic comparator, SIND (Xu et al., 2022), is a
single signalised intersection in China. Dense, motorcycle-dominated
Southeast Asian traffic — where two-wheelers weave between lanes and
constitute the plurality of road users — is essentially absent from the
nadir drone-trajectory literature, as is a controlled comparison across
flight altitudes, even though altitude directly trades spatial coverage
against the pixel footprint of small vehicles.

### 1.5 Contributions

This paper addresses these gaps with an end-to-end UAV traffic benchmark on
top-down video of Thai roads at two altitudes (100 m and 120 m), recorded
over scenes ranging from free-flow carriageways to a complex roundabout.
Our contributions are:

1. **A variance-aware six-detector benchmark.** We train six modern
   detectors — YOLOv9-m, YOLOv10-m, YOLO11-m, YOLO12-m, YOLO26-m, and
   RT-DETR-L — each over five random seeds (30 trained models), and report
   per-seed variance. To our knowledge this is the first variance-aware
   head-to-head of this detector generation on UAV traffic. RT-DETR-L leads
   on every metric within a tight, reproducible three-tier separation
   (seed std ≤ 0.002).

2. **A two-altitude, per-class difficulty analysis.** We evaluate stratified
   by altitude, quantifying a consistent ~1.5–2 mAP@50-point advantage for
   100 m over 120 m, and isolate small vehicles (motorcycles) as the
   hardest class (mAP@50-95 ≈ 0.75–0.79 vs. ≈ 0.88–0.95 for larger
   vehicles) in a setting where they dominate the object count.

3. **A four-tracker benchmark on human-corrected ground truth.** Coupling
   the best detector to ByteTrack, BoT-SORT, OC-SORT, and Deep OC-SORT over
   four held-out, manually corrected sequences, scored with HOTA/MOTA/IDF1.

4. **A trajectory-quality evaluation framework — the paper's central
   contribution.** After homography-based pixel-to-world calibration, we
   score world-space trajectories along three axes: (i) identity
   persistence, (ii) kinematic accuracy against ground truth, and (iii)
   intrinsic physical plausibility (acceleration/jerk bounds). The framework
   *tests*, rather than assumes, that box-level accuracy implies good
   trajectories, and bridges the MOT-metric and traffic-engineering
   evaluation traditions. We further demonstrate that it reproduces the
   box-level ranking with **zero ground truth** on two additional unlabelled
   clips, establishing a reusable, annotation-light protocol.

The remainder of the paper is organised as follows. Section 2 reviews
related work; Section 3 describes the dataset and methodology across the
detection, tracking, and trajectory-quality stages; Section 4 reports
results; Section 5 discusses implications and limitations; and Section 6
concludes.

---

## Appendix — references to add to `references.bib` (with verification flags)

> **SUPERSEDED:** `references.bib` now exists with verified BibTeX entries and
> is the source of truth. This appendix is kept only as a human-readable
> relevance/gap map; where it disagrees with the `.bib`, the `.bib` wins.
> Compiled from a web literature scan (2026-06-15). **Flagged items must be
> verified against the primary source before camera-ready** — venues/years
> for fast-moving detector releases and a few preprints are uncertain.

### Datasets & trajectory pipelines
- **Barmpounakis & Geroliminis (2020)** — pNEUMA. *Transportation Research
  Part C*, 111:50–71. ✅ verified (same target venue — strong anchor).
- **Krajewski et al. (2018)** — highD. *IEEE ITSC 2018*. ✅ verified.
- **Bock et al. (2020)** — inD. *IEEE IV 2020*. ⚠️ confirm venue/year.
- **Krajewski et al. (2020)** — rounD. *IEEE ITSC 2020*. ⚠️ confirm.
- **Moers et al. (2022)** — exiD. *IEEE IV 2022*. ⚠️ confirm venue.
- **Xu et al. (2022)** — SIND. *IEEE ITSC 2022*. ⚠️ confirm first-author given name (IEEE Xplore 9921959).
- **Zheng et al. (2024)** — CitySim. *Transportation Research Record*. ✅.
- **Fonod et al. (2025)** — geo-trax / Songdo, "Advanced computer vision for
  extracting georeferenced vehicle trajectories from drone imagery."
  *Transportation Research Part C*, vol. 178:105205. ✅ (same venue; same 29.97
  fps 4K regime as our clips). NOTE: an earlier draft listed a separate "Chen
  et al. (2025)" — that was a **mis-identification of this same paper** (no
  author "Chen"; correct volume 178, not 175). Now a single entry.

### Aerial detection benchmarks & detectors
- **Zhu et al. (2022)** — VisDrone. *IEEE TPAMI* 44(11):7380–7399. ✅.
- **Du et al. (2018)** — UAVDT. *ECCV 2018*. ✅.
- **Zhao et al. (2024)** — RT-DETR. *CVPR 2024* (cite as CVPR 2024, not the
  2023 arXiv date). ✅ verified in CVPR proceedings.
- **Wang et al. (2024a)** — YOLOv9 (PGI/GELAN). *ECCV 2024*. ✅.
- **Wang et al. (2024b)** — YOLOv10. *NeurIPS 2024*. ⚠️ confirm acceptance.
- **Jocher & Qiu (2024)** — YOLO11 (Ultralytics). ⚠️ **no peer-reviewed
  paper** — cite Ultralytics GitHub/docs (standard convention).
- **Tian et al. (2025)** — YOLO12 (attention-centric). arXiv 2502.12524;
  possibly NeurIPS 2025 poster. ⚠️ confirm venue.
- **Sapkota et al. (2025)** — YOLO26 architectural writeup. arXiv 2509.25164.
  (Corrected from "Khanam et al." — that was a different YOLOv11 overview.)
  YOLO26 is a real Sept-2025 Ultralytics release; the arXiv paper is a
  third-party writeup. STAL (small-target-aware) feature is citable for the
  motorcycle-difficulty point.
- **Akyon et al. (2022)** — SAHI (slicing for small objects). *IEEE ICIP
  2022*. ✅ (frame as the tiling comparator we chose not to use).
- **Wang et al. (2020)** — "Frustratingly Simple Few-Shot Object Detection."
  *ICML 2020*. ✅ (anchor for variance-aware / multi-seed evaluation).

### Tracking & trajectory-quality
- **Bernardin & Stiefelhagen (2008)** — CLEAR-MOT (MOTA/MOTP). *EURASIP
  J. Image Video Process.* ✅.
- **Ristani et al. (2016)** — IDF1. *ECCV 2016 Workshops (BMTT)*. ⚠️ confirm workshop designation.
- **Luiten et al. (2021)** — HOTA. *IJCV* 129:548–578. ✅.
- **Zhang et al. (2022)** — ByteTrack. *ECCV 2022*. ✅.
- **Aharon et al. (2022)** — BoT-SORT. arXiv 2206.14651. ⚠️ **preprint, not
  peer-reviewed** — flag in bib.
- **Cao et al. (2023)** — OC-SORT. *CVPR 2023*. ✅.
- **Maggiolino et al. (2023)** — Deep OC-SORT. *IEEE ICIP 2023*. ✅.
- **Punzo et al. (2011)** — NGSIM trajectory-accuracy assessment.
  *Transportation Research Part C* 19(6):1243–1262. ✅ (Axis-3 precursor).
- **Coifman & Li (2017)** — critical evaluation of NGSIM. *Transportation
  Research Part B* 105:362–377. ✅.
- **Montanino & Punzo (2015)** — trajectory reconstruction. *Transportation
  Research Part B* 80:82–106. ⚠️ confirm exact title/page range.
- **Savitzky & Golay (1964)** — smoothing/differentiation filter.
  *Analytical Chemistry* 36(8):1627–1639. ✅.

### Drafting notes
- Decide whether SIND/VRUD and the OBB-pipeline (Riehl et al., 2025, *Sci.
  Reports*) belong in §2 Related Work rather than the intro — the OBB paper
  is the natural place to defend our HBB choice (decision_log D1).
- Sub-section split (§1.2/§1.3/§1.4 as three gaps) is a suggestion; could
  collapse to two if the editor prefers a shorter intro.
- The Paper-2 SSM hook in §1.1 ties the arc together — keep or trim
  depending on how much to foreshadow.
