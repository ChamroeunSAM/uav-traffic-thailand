# Literature Scan — Custom YOLO Variant (Paper 1.5 candidate)

Scanned 2026-06-12 (three parallel web-research passes). Purpose: scope a
"Name-YOLO" paper (ESWA-genre) built on this project's dataset + pipeline.
This file is the citation base for the design doc; verify quotes/numbers
against the linked sources before using them in a manuscript.

---

## 1. Genre survey — named YOLO variants for UAV vehicle/traffic detection (2023–2026)

### Closest competitors (UAV vehicle/traffic-specific)

| Paper | Venue/Year | Base | Modules | Result | Code |
|---|---|---|---|---|---|
| RSW-YOLO | Sensors 2025 | YOLOv8n | Restormer block, small-target head, WIoUv3 | +5.4 mAP50 on DJCAR (custom UAV vehicle set), but params 3.95× | no |
| DSCW-YOLO | Sensors 2025 | YOLOv11s | C3K2DS, CoordConv, P2 head (motivated by altitude-driven scale change) | +4.2 mAP50 on DJCAR (70–110 m) | no |
| OSD-YOLOv10 | Sci Rep 2025 | YOLOv10n | OCRConv, SPCC, DFMA, DySample, P1/P2 heads | +1.6 mAP50 VisDrone, −40.7% params | promised |
| LMAD-YOLO | PLOS One 2025 | YOLO11n | MultiEdgeEnhancer, MSPF/LSKA, ADown, MDFPN | +2.2 mAP50 DroneVehicle | no |
| EBR-YOLO | Sensors 2025 | YOLOv8n | AZML, Tri-CAT, MLCA, PConv, ADown (non-motorized vehicles ≈ our SV) | −0.2 mAP50 at −70% params | no |
| PVswin-YOLOv8s | Drones 2024 | YOLOv8s | Swin block, CBAM, Soft-NMS | +4.8 mAP50 VisDrone | no |
| Improved YOLOv8 traffic monitoring | PLOS One 2025 | YOLOv8s | MFF, INMS | +4.8 mAP50 VisDrone | public |
| MHA-YOLOv8 + DeepSORT | IEEE conf 2025 | YOLOv8 | MHA + CA; intersection trajectory tool | mAP 93.6, MOTA 98.6 | no |

### Recipe-setters (general UAV small-object)

UAV-YOLOv8 (Sensors 2023: BiFormer, FFNB, WIoUv3, 5-scale; +7.7 VisDrone),
Drone-YOLO (Drones 2023: RepVGG, sandwich fusion, P2; +13.4),
SOD-YOLO (Remote Sensing 2024: RFCBAM, BSSI-FPN, DySample; name already
collides 3 ways), RLRD-YOLO (Drones 2025; +12.2, Jetson), EL-YOLO (ESWA
2024: SCAFPN, CSL-MHSA, Jetson FP16), MFEL-YOLO (ESWA 2025: HE-C2F, MFFM).

### Survey conclusions

- **Saturated template:** P2/P1 head + bolt-on attention + redesigned FPN
  (often DySample) + IoU-loss swap (WIoUv3 dominant) + lightweighting
  garnish; bases are v8n/s shifting to v10/11; claims +2–12 mAP50 on
  VisDrone with flat-or-lower params.
- **Code public in only ~2 of 15** — an open repo is itself a differentiator.
- **No paper handles true-nadir multi-altitude traffic with stratified
  per-altitude evaluation.** VisDrone/UAVDT are mostly oblique low-altitude;
  DJCAR (70–110 m, 4K) is the closest but unreleased and not stratified.
- **No journal paper couples a custom detector to tracker benchmarking +
  trajectory-quality evaluation** (only a thin IEEE-conference
  MHA-YOLOv8+DeepSORT tool). Our Stage 1–4 pipeline has no equivalent.

Links: RSW https://pmc.ncbi.nlm.nih.gov/articles/PMC12298911/ · DSCW
https://pmc.ncbi.nlm.nih.gov/articles/PMC12157958/ · OSD
https://pmc.ncbi.nlm.nih.gov/articles/PMC12254417/ · LMAD
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12262879/ · EBR
https://pmc.ncbi.nlm.nih.gov/articles/PMC11723268/ · PVswin
https://www.mdpi.com/2504-446X/8/3/84 · traffic-monitoring
https://pmc.ncbi.nlm.nih.gov/articles/PMC11918428/ · UAV-YOLOv8
https://pmc.ncbi.nlm.nih.gov/articles/PMC10458807/ · Drone-YOLO
https://www.mdpi.com/2504-446x/7/8/526 · SOD-YOLO
https://www.mdpi.com/2072-4292/16/16/3057 · RLRD
https://www.mdpi.com/2504-446X/9/4/293 · EL-YOLO
https://dl.acm.org/doi/10.1016/j.eswa.2024.124848 · MFEL-YOLO
https://dl.acm.org/doi/10.1016/j.eswa.2025.128459

---

## 2. Novelty check — GSD/altitude prior: MOSTLY ALREADY CLAIMED

**The broad idea "condition a detector on GSD/altitude" is published.**
Required citations + differentiation targets:

- **GSDet** (IEEE TIP 2021) — claims to be first to put GSD into detection;
  estimates GSD, uses physical size to reweight the *classification* branch.
  https://ieeexplore.ieee.org/document/9411691/
- **GSDDet** (IEEE 2023) — GSD-guided features against cross-scale category
  confusion. https://ieeexplore.ieee.org/document/10285472/
- **P2M-PF "From Pixels to Meters"** (IEEE 2025) — lightweight physical-size
  prior injected into the classification branch; confirms an active
  sub-genre. https://ieeexplore.ieee.org/document/11316490/
- **Xue et al. 2021** — known flight altitude as a scalar input to a UAV
  detection head (aerial animal surveillance).
  https://arxiv.org/abs/2111.06830
- **Scale-MAE** (ICCV 2023) — GSD-conditioned positional encoding.
- **NDFT** (ICCV 2019) — altitude metadata used adversarially (invariance,
  the opposite sign). https://arxiv.org/pdf/1908.03856
- **Altitude-aware dynamic tiling** (arXiv 2511.19728, 2025) — live altitude
  drives inference-time tile scaling (+38% small-object mAP, SeaDronesSee).
- **Canonical-GSD resampling is standard practice** (YOLT 2018; Scale Match
  WACV 2020; GSD-matched domain adaptation arXiv 2108.07779).
- **P2-add/P5-drop is commoditized** with the same FLOPs-savings argument we
  would use (SOD-YOLO arXiv 2507.12727, LAF-YOLOv10, PC-YOLO11s, YOLO-Tiny,
  many more). Not claimable as a contribution; fine as cited engineering.

**What remains open (narrow but real):**
1. Using **known, exact GSD** (not estimated) to condition the **regression
   targets / anchor-scale priors in metric units** — e.g. box sizes
   predicted in metres, or assignment priors set analytically from class
   physical-size bounds × GSD. The GSDet family conditions classification,
   not regression/assignment.
2. **Analytic pyramid-level selection derived from GSD × class size bounds**
   (vs the empirical dataset-statistics argument) — defensible as framing,
   but the resulting architecture matches the existing P2/P5 recipe.
3. The fixed-GSD nadir traffic regime itself (scale variance ≈ 0) is
   unstudied — but that also weakens the case for conditioning (conditioning
   helps most when scale varies → our two-altitude setup is what makes it
   testable at all).

**Verdict:** "GSD-aware detection" cannot be claimed as new. A
**known-GSD metric-prior module acting on regression/assignment**, carefully
differentiated from GSDet/GSDDet/P2M-PF/Xue/tiling, is the surviving sliver.

---

## 3. Names

Taken (avoid): GSD-YOLO (×2 + reversed forms; acronym polluted — reads
"gear surface defect"), UTD-YOLO (×2), UAV-YOLO (badly collided),
AVD-YOLO, TSD-YOLO (×3), TrafficYOLO.

Apparently free as of 2026-06-12: **Nadir-YOLO**, **Alti-YOLO**,
**UVD-YOLO**, **Ortho-YOLO**, **AGL-YOLO** (above-ground-level; keep
distance from AG-YOLO, Remote Sensing 2025), **GSDA-YOLO**. The phrase
"GSD-aware" as a module brand also has no detector hits.

---

## 4. Module saturation tally (what not to lead with)

| Ingredient | Status |
|---|---|
| P2/high-res head | SATURATED (genre default; several papers also drop P5) |
| SPD-Conv | SATURATED (papers named after it) |
| BiFPN-style necks | SATURATED (the letter-prefix-BiFPN cottage industry) |
| CBAM/CA/EMA/SimAM | SATURATED (most saturated; menu-ablated) |
| WIoU/SIoU/EIoU/Focaler | SATURATED (cheapest checklist line) |
| Ghost/lightweight backbones | COMMON |
| NWD tiny-object loss | COMMON → saturating (plain NWD no longer fresh) |
| DySample/CARAFE | COMMON |
| Slim-neck/GSConv | COMMON, saturating |

Guidance: saturated items are fine as engineering, fatal as claimed
contributions. Pair the metric-prior centerpiece with COMMON-bin support
(NWD-style assignment for SV, DySample) to keep the ablation table from
reading as a checklist.

---

## 5. Synthesis — positioning that survives the scan

The novelty cannot be "GSD conditioning" per se. The package that holds up:

1. **Centerpiece module:** known-AGL/GSD **metric-scale prior on
   regression/assignment** (open sliver §2.1), with analytic pyramid
   selection as supporting design rationale (§2.2).
2. **Benchmark setting nobody has:** true-nadir, two-altitude (100/120 m),
   per-altitude-stratified traffic benchmark with SV (motorcycles) as the
   hard class — no precedent in the genre survey.
3. **System differentiation:** detector feeds the Stage 3/4
   tracking + trajectory-quality pipeline (cross-cite Paper 1) — no
   journal-level precedent.
4. **Open code** — only ~2/15 competitors release it.
5. **Evidence package:** VisDrone (headroom for genre-typical +mAP claims)
   + our Thai benchmark ("matches RT-DETR-L SV accuracy at a fraction of
   the FLOPs") + ablation per module × 5 seeds.

Name shortlist: Nadir-YOLO (top pick — names the geometry that makes the
prior valid), AGL-YOLO, Alti-YOLO.
