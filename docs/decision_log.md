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
