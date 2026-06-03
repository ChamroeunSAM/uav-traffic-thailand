# Calibration Protocol (Stage 4)

Image-to-world calibration for trajectory extraction. Converts pixel
trajectories into a local **metric ground frame (metres)** so kinematics
(speed, acceleration, heading) are physical. Decision: `decision_log.md`
**D4**.

## Method: per-clip ground-plane homography

UAV footage here is near-nadir over locally flat road, so the image→world
map on the ground plane is a single **planar homography** `H` (3×3), fit
per clip:

```
[X', Y', w]ᵀ = H · [u, v, 1]ᵀ        (X, Y) = (X'/w, Y'/w)
```

`(u, v)` = pixel, `(X, Y)` = metres in a local road-aligned frame. One `H`
per clip (each clip is a different site / altitude). No global
georeferencing — kinematics need only **relative** metric distance.

There is no drone telemetry (the clips carry no SRT/EXIF flight metadata),
so `H` is anchored to **known road-marking geometry**, not flight
parameters.

## Procedure

1. **Dump a reference frame.**
   ```
   python scripts/00_calibrate.py --video data/video_clips/<clip>.mp4 --dump-frame
   ```
   Writes `data/video_annotations/calibration/frames/<clip>.png` (full 4K).
   If markings are occluded by vehicles, pick a clearer frame with
   `--frame N`.

2. **Select ground control points (GCPs).** Open the PNG in a viewer that
   shows pixel coords on hover. Choose **6–8** points (≥4 required) that are:
   - on the **flat road plane** (not vehicle roofs or raised kerbs),
   - **well spread** across the frame (clustered points give an unstable
     fit), and
   - identifiable to known geometry — lane-edge / centre-line marking
     ends, dashed-segment corners, stop-line and crosswalk corners.

3. **Assign world coordinates.** Define a local metric frame: one GCP is
   the origin `(0, 0)`; give every other point its `(X, Y)` in **metres**
   along road-aligned axes, from known marking spacing.

4. **Fill** `configs/calibration/<clip>.gcp.yaml` (template provided): each
   point's `pixel: [u, v]` and `world: [X, Y]`, the `reference:` note
   (exact distances used and their source), and the optional `validation:`
   segment.

5. **Fit.**
   ```
   python scripts/00_calibrate.py --gcp configs/calibration/<clip>.gcp.yaml
   ```
   Writes `data/video_annotations/calibration/<clip>.json` (the 3×3 `H`,
   reprojection RMSE, GCP count, scale-check result).

## Error budget

- **Reprojection RMSE (metres)** — mean distance between each surveyed GCP
  and the reprojection of its pixel. Reported and stored per clip. As a
  rule of thumb keep it well under the vehicle scale (≲ a few × 0.1 m at
  these altitudes); the script warns above 1 m. Large RMSE ⇒ a mistyped
  point or an off-plane / poorly-spread GCP set — fix before trusting
  kinematics.
- **Held-out scale check** — RMSE is computed over the *fit* points, so it
  is blind to a **globally-wrong reference distance** (a wrong lane width
  scales every speed/acceleration uniformly and leaves RMSE untouched). The
  optional `validation:` segment (two pixels + an independently-known
  distance, *not* used in the fit) exposes exactly this: the fitter reports
  measured-vs-known and the % error.

## Assumptions & limitations

- **Flat ground plane.** A single homography assumes the road is planar.
  Valid for these flat urban sites; would break on graded / banked roads.
- **Near-nadir.** Slight gimbal tilt is absorbed by the homography (this is
  why a homography is preferred over a single GSD scalar). Severe obliquity
  would degrade off-centre accuracy.
- **Reference fidelity.** Accuracy is bounded by how well the assumed
  marking dimensions match reality — measured/satellite distances beat
  standards. Record the source in `reference:`.
