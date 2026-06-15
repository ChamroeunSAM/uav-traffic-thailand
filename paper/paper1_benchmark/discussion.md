# Paper 1 — Discussion (draft)

**Venue strategy:** see `abstract.md` (not locked). Citations in (Author,
Year) form; entries in `introduction.md` appendix.
**Status:** first full draft — author to revise.

---

## 5. Discussion

### 5.1 What the benchmark establishes

Three results carry beyond this dataset. First, on top-down traffic imagery a
real-time transformer (RT-DETR-L) outperforms the entire modern YOLO line,
and it does so by a margin that exceeds run-to-run variance — a claim only a
multi-seed protocol can support. The wider lesson is methodological: because
the strongest YOLO competitors sit within ~0.5 mAP@50 of one another (and of
RT-DETR-L on the easy classes), single-seed comparisons of this detector
generation are not trustworthy, and reported rankings that omit variance
should be read with caution (Wang et al., 2020). Second, the difficulty is
concentrated in small vehicles at strict localisation IoU: motorcycles are
*detected* well (AP@.5 ≈ 0.96) but *localised* poorly (AP@.5:.95 ≈ 0.79),
which both explains the inter-detector separation and identifies the precise
failure mode that downstream trajectory work must tolerate. Third, lower
flight altitude helps uniformly — 100 m beats 120 m by ~1.5–2 mAP@50 points
across every architecture — quantifying the coverage-versus-resolution trade
that any UAV deployment must make.

### 5.2 Why trajectory quality needed its own framework

The headline contribution is not the leaderboard but the demonstration that
trajectory quality can and should be evaluated directly. Box-level metrics
(HOTA, IDF1) live in image space; they cannot see whether a trajectory's
speed, acceleration, and jerk are physically realistic. Our framework closes
that gap by scoring world-space trajectories along identity, kinematic, and
plausibility axes. The empirical outcome is, in a sense, reassuring: the
trajectory ranking *agrees* with the box ranking — ByteTrack and BoT-SORT
lead on identity *and* plausibility — so in this setting box metrics are a
validated cheap proxy. But that conclusion is earned, not assumed. We
pre-registered the opposite hypothesis (a Kalman-smoothing inversion in which
OC-SORT's filter cleans its trajectories enough to overtake) and tested it;
it did not occur because OC-SORT's weaker association injects more kinematic
noise than its filter removes (Tables T5, T5b). A framework that can *confirm*
the proxy is also one that can *detect its failure* on other data — which is
the reusable value, underscored by the framework reproducing the ranking with
zero ground truth on two unlabelled clips (Table T6).

### 5.3 Practical implications

For practitioners building UAV trajectory datasets, the results suggest a
concrete recipe: detect with the most accurate available model when
throughput is not binding (here RT-DETR-L for offline extraction); track with
a strong motion/IoU tracker (ByteTrack or BoT-SORT), since appearance
re-identification adds little from overhead (decision log D3); and follow with
an offline linking pass, which lifts every tracker and sharply reduces
identity switches. Crucially, the trajectory-quality framework gives such a
pipeline a *self-check*: the Axis-3 plausibility and self-consistency proxies
require no annotation, so a team can monitor trajectory quality on new
footage where ground truth is unavailable — the common case in deployment.

### 5.4 Limitations

Several limitations bound the claims and should be stated plainly.

- **Near-ceiling accuracy.** Detection and box-tracking metrics are high
  across the board, partly because the top-down view is favourable and the
  scenes, while dense, are well-resolved at 1280². Margins between detectors
  are correspondingly tight; the benchmark discriminates, but the headroom
  for further gains on *this* data is limited. This is also why the
  differentiated trajectory channels, not the box metrics, carry the
  discriminating signal.
- **Axis-2 positional error is provenance-confounded.** Because the Stage 2
  ground truth was seeded semi-automatically from RT-DETR-L detections, GT
  and predictions share detection provenance, so positional ADE is near-zero
  (≈3 mm) and does not measure absolute localisation accuracy. The Axis-2
  signal lives in the differentiated speed/heading channels; an
  independently sourced GT (e.g. manual re-annotation or an external sensor)
  would be needed to characterise absolute positional error.
- **Single region and scene set.** The data is Thai traffic at one set of
  sites; the motorcycle-dominated composition is the point, but
  generalisation to other geographies, weather, and times of day is untested
  here. The GT-free clips extend coverage but remain same-region.
- **Calibration is planar and GCP-based.** Pixel-to-world mapping assumes a
  flat ground plane and uses four road-marking GCPs per clip, giving an exact
  (RMSE ≈ 0) but unredundant fit; accuracy is guarded only by the
  scale-consistency check and altitude-GSD agreement (Table T7). Terrain
  slope or lens distortion residuals are not separately modelled.
- **Plausibility bounds are fixed reference values.** The Axis-3 thresholds
  (|a| ≤ 8 m/s², 5 m match gate, Savitzky–Golay window 15 / order 2) are
  reasoned defaults; a sensitivity analysis would strengthen the framework's
  external validity.

### 5.5 Toward traffic state-space modelling (Paper 2)

This benchmark is the measurement layer for a second study that applies the
extracted trajectories to traffic state-space modelling. Two findings here
shape that work directly: the concentration of error in small-vehicle
localisation flags where trajectory uncertainty will be largest precisely for
the most numerous road users (motorcycles), and the validated trajectory-
quality framework provides the input-quality gate that any downstream
modelling should pass its trajectories through before fitting. In this sense
Paper 1 delivers not only a benchmark but the quality-controlled trajectory
substrate the modelling arc depends on.

---

## Drafting notes (not part of the section)

- **Limitations are deliberately frank** — for a Q1 venue this is a strength,
  not a weakness; reviewers reward pre-registration (the D4 hypothesis) and
  honest provenance disclosure (Axis-2 ADE). Keep them.
- **Sensitivity analysis (§5.4 last bullet)** is the most likely reviewer
  request; if time permits before submission, a small sweep over the Axis-3
  thresholds / SG window would pre-empt it. Currently framed as a limitation,
  not a result.
- **Paper 2 bridge (§5.5)** foreshadows without over-promising; trim if a
  venue prefers a self-contained paper.
- For a CV/remote-sensing venue, consider moving some of §5.1's methodological
  argument (variance-aware evaluation) up into the contributions, where that
  audience weights it more heavily.
