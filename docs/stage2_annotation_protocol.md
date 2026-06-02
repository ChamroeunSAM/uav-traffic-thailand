# Stage 2 Annotation Protocol

> Video-tracker ground truth from linker pseudo-tracks via CVAT.

## 1. Purpose

Stage 2 produces the **ground-truth tracker annotations** used to evaluate
ByteTrack / BoT-SORT / OC-SORT / Deep OC-SORT in Stage 3. The detector +
ByteTrack + linker pipeline (`scripts/02_generate_pseudo_tracks.py` and
`scripts/02b_link_tracks.py`) emits *pseudo-tracks* — automatically-generated
but imperfect annotations. A human annotator opens each clip in CVAT, imports
the pseudo-tracks as a starting seed, corrects the errors, and exports a
clean MOT-format ground-truth file.

This document is the operational protocol for that correction workflow.

**Downstream use (per decision-log D3):** each Stage 2 GT file is the
reference against which Stage 3 will evaluate every tracker *twice* — once
on raw tracker output (standard MOT-literature comparability) and once on
tracker-plus-linker output (relevance to the offline trajectory work in
Stage 4). So the GT only needs to be produced once; it serves both
evaluations.

## 2. Inputs and outputs

**Inputs** (one set per clip; four clips total):

```
data/video_clips/<clip>.mp4                                    # source video
data/video_annotations/pseudo_tracks/<clip>.linked_v3.txt      # seed annotations (MOT 1.1)
data/video_annotations/pseudo_tracks/<clip>.linked_v3.vis.mp4  # visual reference
```

The four `<clip>` stems are:

| Stem | Altitude | Scenario | Seed track count |
|---|---|---|---:|
| `100_m_DJI_0001_1min` | 100 m | Free-flow | 40 |
| `100_m_DJI_0054_1min` | 100 m | Intersection (~200 vehicles) | 218 |
| `120_m_DJI_0006_1min` | 120 m | Moderate / roundabout | 61 |
| `120_m_DJI_0047_1min` | 120 m | Congested | 115 |

**Output** (committed under `data/video_annotations/ground_truth/`):

```
data/video_annotations/ground_truth/<clip>.gt.txt   # corrected MOT 1.1
```

These four files become the Stage 3 evaluation ground truth.

## 3. Tool choice: CVAT self-hosted

**Tool: CVAT, self-hosted on the Windows workstation via Docker Desktop.**
Not cvat.ai cloud — the traffic footage is local data and the videos
(~750 MB each at 4K) are slow to upload.

Why CVAT over alternatives:

- Native MOT 1.1 import + export (matches our pseudo-track format).
- First-class video tracking mode with keyframe interpolation.
- `Merge tracks` and `Split track` operations are first-class — exactly the
  edits needed when correcting the linker's residual mistakes.
- Open source, runs locally, no per-annotator licensing.

Avoid Roboflow for this stage — its tracking annotation support is
substantially weaker than CVAT's; the static-frame export workflow we used
for Stage 0 is not the right tool here.

## 4. CVAT setup (one-time)

Prerequisite: Docker Desktop installed on Windows 11 with WSL2 backend
enabled. (Allowed exception to the project's "native, not WSL" rule —
CVAT only runs in containers.)

```powershell
git clone https://github.com/cvat-ai/cvat
cd cvat
docker compose up -d
```

CVAT is then accessible at `http://localhost:8080`. Create an admin user
on first visit. Default resource budget (8 GB RAM, 4 vCPUs allocated to
Docker) is sufficient for 4K @ 29.97 fps annotation; bump to 16 GB if
multiple annotators work in parallel.

Allocate one CVAT **Project** for this work (e.g., "UAV Thai Stage 2");
each video becomes a **Task** within that project.

## 5. Project label setup

Create the project with exactly three labels, in this order, so CVAT's
internal class IDs match the dataset's class IDs (0=LV, 1=MV, 2=SV):

| Order | Label name | Color (suggested) | Class ID |
|---:|---|---|---:|
| 1 | `LV` | red | 0 |
| 2 | `MV` | blue | 1 |
| 3 | `SV` | green | 2 |

**This order matters.** CVAT assigns class IDs by label creation order, and
the MOT import maps the class column to the corresponding label index. If
the order is wrong, every box gets the wrong class.

For each label, attribute type = `none` (no extra attributes needed for
Stage 2). Shape type allowed: rectangle only.

## 6. Task creation per clip

Repeat for each of the four clips:

1. **Create Task** → assign to the Stage 2 project.
2. **Upload source video** → `data/video_clips/<clip>.mp4`.
3. **Image quality**: 70 (default). Higher slows down the UI for 4K with
   minimal annotation benefit.
4. **Use cache**: enabled (lets CVAT pre-decode the video for smooth scrubbing).
5. **Chunk size**: 36 frames (default — works for 4K).
6. **Segment size**: leave empty (let CVAT pick one segment per task).
7. **Start frame**: 0. **Stop frame**: leave empty (process all 1799 frames).
8. **Frame filter / step**: leave defaults (no subsampling — Stage 3 metrics
   are computed at native frame rate).
9. Submit. CVAT will spend ~1-2 minutes extracting frames; wait for status
   to flip to `Completed`.

Once the task is created, open the task page and use **Actions → Upload
annotations**:

- **Annotation format**: `MOT 1.1`
- **File**: the corresponding `<clip>.linked_v3.txt`
- Submit. The seed pseudo-tracks should appear when you open the task.

## 7. Annotation workflow (five passes per clip)

Work through each clip in a fixed sequence of correction passes. Do
**not** try to fix everything in a single scrub — the passes are ordered
from cheapest-to-fix to most-expensive-to-fix, and earlier passes reduce
the work for later ones.

### Pass 1 — Visual scan, no edits (~5 minutes per clip)

Play the clip start-to-finish at 1x speed without making any corrections.
Goal: build a mental map of the scene, identify trouble spots, see how
the pseudo-tracks behave. Note (on paper or a separate doc) any time
ranges where many errors cluster — those are the regions for Pass 2-4.

### Pass 2 — Delete junk tracks (~10-30 minutes per clip)

Open the **Object Sidebar**. CVAT lists every track with its lifetime
and label. Scan for clearly-spurious tracks — typically:

- Tracks lasting only 1-3 frames at the edge of frame (false positive on
  motion blur, leaves, shadows).
- Tracks fully overlapping a longer "real" track that the linker missed
  consolidating.
- Tracks on stationary objects misidentified as vehicles (signs, painted
  road markings, lamp posts).

For each junk track: select it in the sidebar → press `T` (or right-click
→ Remove track). Track-delete is one click in CVAT — this pass is fast.

### Pass 3 — Merge fragmented tracks (~30-90 minutes per clip, scenario-dependent)

Linker residual fragmentation: a single vehicle is split into two or more
ID sequences. Use CVAT's **Merge tracks** operation:

1. Select track A (the first half).
2. Hold `Ctrl` and click track B (the second half).
3. Press `M` (or right-click → Merge tracks).

The merged track inherits track A's ID and the union of both
annotations. The class label takes track A's value — if A and B
disagree (e.g., the same vehicle was MV then LV), set the correct class
before or after the merge by right-clicking the merged track →
Properties → Label.

**Where to look for these:** any time a vehicle exits frame and the
linker missed a brief occlusion that lasted >6 seconds (the linker's
default `max-gap`), or any time the linker rejected a valid merge due
to the size tolerance. The intersection clip (`DJI_0054`) is where this
pass is most expensive — budget ~60-90 minutes for it alone.

### Pass 4 — Split incorrectly-merged tracks (~10-30 minutes per clip)

The linker is more aggressive about merging than splitting. Occasionally
it will merge two distinct vehicles whose trajectories happened to be
similar (same class, similar size, gap within tolerance, position
roughly consistent). The visual signature is a track that "teleports"
between two physically reasonable positions across a multi-second gap.

For each teleport:

1. Scrub to the last frame where the track is in its first position.
2. Right-click that keyframe → **Split track**.
3. The second half gets a new ID. Verify the split point is correct.

This pass is least common in the free-flow / moderate clips and most
common in the intersection clip (where many similar motorcycles converge).

### Pass 5 — Fix class labels and fill in missed vehicles (~5-20 minutes per clip)

The detector occasionally misclassifies LV ↔ MV at the boundary
(pickup trucks, large SUVs, vans). The linker's overlap-merge already
fixes the persistent cross-class duplicates, but the *class label* on
the merged track is taken from the longer fragment, which may be wrong.

Scrub the clip once more, watching the label colors (red/blue/green).
For each obvious class error: right-click track → Properties → change
Label. Class-edit affects all frames of the track simultaneously.

If you see a vehicle clearly on screen with no bounding box at all
(detector missed it entirely): draw the box and assign a class. Use
keyframes — set a box at frame N, scrub to the last frame where the
vehicle is visible (say frame N+50), set a box there; CVAT interpolates
the in-between frames. Only do this for vehicles fully missed
end-to-end. Detector recall is high enough that this is rare.

## 8. Export

After Pass 5 on a clip:

1. Open the task in CVAT.
2. **Actions → Export task dataset**.
3. **Format**: `MOT 1.1`.
4. Save the resulting zip; extract `gt/gt.txt`.
5. Rename to `<clip>.gt.txt` and place under
   `data/video_annotations/ground_truth/`.

Verify the output is parseable by running a quick sanity check:

```
python -c "
from collections import defaultdict
t = defaultdict(list)
with open('data/video_annotations/ground_truth/<clip>.gt.txt') as f:
    for line in f:
        p = line.strip().split(',')
        t[int(p[1])].append(int(p[0]))
print(f'tracks: {len(t)}')
print(f'rows:   {sum(len(v) for v in t.values())}')
print(f'frames: {max(max(v) for v in t.values())}')
"
```

Track count should be in the same ballpark as the linker output (within
±30%), row count should be roughly tracks × average lifetime, frame
range should hit 1799.

## 9. Annotation order across the four clips

**Annotate in difficulty order: easiest first, hardest last.** This lets
your correction protocol stabilize on the cheap clips before the
expensive intersection clip:

1. **`120_m_DJI_0006_1min`** (moderate roundabout, 61 seed tracks) — use
   this to calibrate the protocol. ~3-5 hours.
2. **`100_m_DJI_0001_1min`** (freeflow, 40 seed tracks) — ~2-3 hours.
3. **`120_m_DJI_0047_1min`** (congested, 115 seed tracks) — ~6-10 hours.
4. **`100_m_DJI_0054_1min`** (intersection, 218 seed tracks) — ~10-15
   hours. Save this for last.

**Total budget: ~25-35 hours of careful annotator labor.** Single
annotator; inter-annotator validation was dropped per decision-log D3,
on the basis that the top-down UAV view at 100-120 m altitude minimizes
the perspective-induced ambiguity that motivates such checks in
ground-level MOT benchmarks. A short paragraph in the paper's
Limitations section will note this choice for reviewers.

## 10. Common pitfalls and recovery

- **Lost CVAT session / browser crash:** CVAT auto-saves after every
  edit. On reopen, your work is preserved. There is no manual save
  needed — but do confirm via `Actions → Export task dataset` at the
  end of each work session as a backup.
- **Accidentally deleted a real track:** `Ctrl+Z` reverts the last
  operation. CVAT keeps a per-session undo history; use it before
  navigating away from the task.
- **Class IDs in exported gt.txt don't match expectations:** verify the
  project's label order is `LV, MV, SV` exactly (§5). If labels were
  created in a different order, the class column in the export will be
  shuffled — fix the label order in CVAT settings and re-export.
- **MOT import shows "no tracks loaded":** verify the file is plain
  `<clip>.linked_v3.txt`, not the visualization MP4. CVAT's MOT
  importer reads the first column as frame number; if the file has a
  header row, delete it.
- **Frame numbers don't line up after import:** the linker emits
  1-indexed frames per MOT spec. CVAT uses 0-indexed internally but
  handles the conversion automatically on MOT import. If you see a
  consistent 1-frame offset in the displayed boxes, file a bug — this
  shouldn't happen in current CVAT versions.

## 11. Definition of done

A clip is "done" when:

- All five passes have been worked through.
- The track count in the exported `.gt.txt` is stable (you re-scrubbed
  and made no further edits).
- The exported file is at `data/video_annotations/ground_truth/<clip>.gt.txt`.
- The corresponding row in `docs/decision_log.md` (Stage 2 section, when
  written) lists annotator name + annotation date + final track count.

After all four clips are done, Stage 2 is complete and Stage 3 can begin.
