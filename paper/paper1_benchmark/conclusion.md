# Paper 1 — Conclusion (draft)

**Venue strategy:** see `abstract.md` (not locked).
**Status:** first full draft — author to revise.

---

## 6. Conclusion

We presented an end-to-end UAV traffic benchmark for top-down vehicle
detection, tracking, and trajectory extraction on dense, motorcycle-dominated
Thai traffic recorded at two flight altitudes. Three contributions follow.
First, a variance-aware comparison of six modern detectors (five seeds each,
30 models) establishes a stable three-tier ranking in which the real-time
transformer RT-DETR-L leads the entire YOLO line by a margin exceeding
run-to-run variance, with difficulty concentrated in the strict-IoU
localisation of small vehicles and a consistent 100 m-over-120 m advantage.
Second, a four-tracker benchmark on human-corrected sequences shows that
motion/IoU-based trackers (ByteTrack, BoT-SORT) lead from the overhead
viewpoint, that appearance re-identification adds little, and that an offline
linking pass lifts every tracker while collapsing identity switches. Third —
and centrally — a trajectory-quality framework evaluates world-space
trajectories along identity, kinematic, and physical-plausibility axes,
moving evaluation beyond boxes in image space.

The framework's key result is that trajectory quality and box-level quality
*agree* in this setting: the trackers that win on HOTA and IDF1 also produce
the most persistent and most physically plausible trajectories, so box
metrics serve as a validated cheap proxy here. This conclusion is earned
rather than assumed — we pre-registered and rejected the opposing
Kalman-smoothing-inversion hypothesis — and the framework's value is its
ability to test the proxy rather than presume it, demonstrated by reproducing
the ranking with no ground truth on additional clips. Together these results
provide both a reproducible benchmark and a reusable, annotation-light
protocol for evaluating UAV traffic-sensing pipelines.

The benchmark also serves as the quality-controlled measurement layer for
subsequent traffic state-space modelling, where the extracted trajectories —
gated by the quality framework introduced here — become the modelling
substrate. Future work includes extending the dataset across geographies,
weather, and times of day; sourcing an independent ground truth to
characterise absolute positional accuracy beyond the differentiated kinematic
channels; and a sensitivity analysis of the plausibility bounds. All code,
configurations, and aggregated results are released to support reproduction
and reuse.

---

## Drafting notes (not part of the section)

- **No new claims or numbers** introduced here by design — conclusions
  restate §4–§5 only. Verify nothing crept in beyond what the tables support.
- **Length:** two body paragraphs + future work; trim the future-work list to
  match the target venue's conclusion norms (transportation journals tolerate
  a fuller outlook; some CV venues prefer one tight paragraph).
- **Reproducibility sentence** assumes the code/data release is finalised —
  confirm the repository/DOI availability statement before submission.
- With this, the first-draft skeleton is complete: `abstract`,
  `introduction`, `related_work`, `methodology`, `results`, `discussion`,
  `conclusion`. Suggested next steps: (1) verify the ⚠️-flagged citations and
  build `references.bib`; (2) stitch the sections into one manuscript file;
  (3) author sync with `docs/methodology.docx`.
