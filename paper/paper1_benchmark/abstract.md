# Paper 1 — Abstract (draft)

**Venue strategy (not locked):** aim high, stay adaptable. Primary target
*Transportation Research Part C: Emerging Technologies* (Elsevier, Q1), but
written to also fit:
- **Taylor & Francis** — e.g. *Journal of Intelligent Transportation
  Systems*, *Transportmetrica A/B*, *International Journal of Remote Sensing*.
- **Springer** — e.g. *Machine Vision and Applications*, *Neural Computing
  and Applications*, *International Journal of ITS Research*.

The framing is transportation/ITS-general so it ports across all three
publishers; per-venue we only re-weight emphasis (benchmark-forward for
CV/remote-sensing journals vs. trajectory-quality-forward for transportation
journals) and reformat. We still aim for the strongest possible paper.

**Status:** skeleton / first draft — author to revise

---

## Title (locked)

**Toward Reliable Trajectory Extraction from UAV Traffic Video: A
Two-Altitude Detector–Tracker Benchmark with a Trajectory-Quality
Framework**

## Keywords

UAV traffic monitoring; vehicle detection benchmark; multi-object
tracking; trajectory-quality evaluation; aerial imagery; intelligent
transportation systems

---

## Abstract (draft, ~250 words)

Unmanned aerial vehicles (UAVs) are an increasingly attractive sensor for
extracting vehicle trajectories from traffic scenes, yet most studies stop
at bounding-box detection accuracy and assume good detection implies good
trajectories — an assumption that has never been tested systematically for
top-down imagery, and rarely on the dense, motorcycle-heavy traffic typical
of Southeast Asian cities. We present a UAV traffic benchmark recorded over
Thai roads at two altitudes (100 m and 120 m), comprising 3,010 annotated
top-down frames across three vehicle classes (large, medium, and small
vehicles, the last dominated by motorcycles). On this dataset we benchmark
six modern detectors (YOLOv9-m, YOLOv10-m, YOLO11-m, YOLO12-m, YOLO26-m, and
RT-DETR-L), each over five random seeds for variance-aware ranking. RT-DETR-L
leads on every metric (mAP@50 0.963, mAP@50–95 0.876) within a tight,
reproducible three-tier separation (seed std ≤ 0.002); small vehicles remain
the hardest class at strict IoU, and 100 m consistently outperforms 120 m.
We then couple the best detector to four trackers (ByteTrack, BoT-SORT,
OC-SORT, Deep OC-SORT) on four held-out, human-corrected sequences, scored
with HOTA/MOTA/IDF1 (ByteTrack and BoT-SORT lead, HOTA ≈ 0.89). Finally, we
contribute a trajectory-quality evaluation framework that scores world-space
trajectories along three axes — identity persistence, kinematic accuracy
against ground truth, and intrinsic physical plausibility — after
homography-based pixel-to-world calibration. The framework reproduces the
box-level ranking with zero ground truth on two additional unlabeled clips,
demonstrating a reusable, annotation-light protocol for evaluating UAV
traffic-sensing pipelines end to end.

---

## Notes for revision (not part of the abstract)

- **Numbers grounded in committed results** (`results/stage1/`, `stage3/`,
  `stage4/`; `data/splits/split_metadata.json`): 3,010 frames = 2,107 train /
  452 val / 451 test; RT-DETR-L 0.9633 / 0.8756; ByteTrack/BoT-SORT HOTA ≈ 0.89.
- **Decide framing emphasis:** the trajectory-quality framework is the stated
  headline contribution (CLAUDE.md §3, Stage 4) — consider leading harder on
  it vs. the benchmark, depending on reviewer-novelty positioning.
- **Open question to settle before submission:** report whether trajectory
  rankings *agree* with box rankings (D4 result: they do — box metrics are a
  validated cheap proxy here). Currently implied by "reproduces the box-level
  ranking"; could be stated explicitly as a finding.
- **Possible additions:** dataset/code availability statement; mention the
  GCP-calibrated cross-altitude scale check; one line on real-world speed/FPS
  if a deployment angle is wanted.
