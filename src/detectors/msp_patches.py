"""Metric Scale Prior, route (ii-A): known-GSD input channel (Nadir-YOLO, D6).

Design doc §8d. We feed the *known, exact* ground-sample-distance (GSD) into
the detector as a 4th input channel so the regression head can condition on
physical scale — a 100 m and a 120 m motorcycle look different in pixels but
carry their GSD, letting the network resolve metric size. Prior art conditions
the *classification* branch with an *estimated* GSD; we condition with a known
exact GSD and let it inform *regression*.

Implementation follows the `boxmot_patches.py` philosophy: NO edits to the
locked Ultralytics install. Everything here is a `DetectionTrainer` /
`DetectionValidator` subclass, activated only when a training config sets
``msp_mode: ii_a``; stock runs are byte-for-byte unaffected.

Pieces:
  1. 4-channel first conv  — `MSPTrainer.get_model` builds the model with
     ch=4 and seeds the RGB sub-weights from the 3-channel pretrained conv
     (4th channel zero-init → initially a no-op, so transfer is preserved).
  2. GSD-plane injection   — both train (`preprocess_batch`) and val
     (`MSPValidator.preprocess`) append a per-image constant plane = effective
     GSD(im_file, imgsz). Images without an altitude prefix get GSD_DEFAULT.

Metric DFL target (the §8d "+ metric target" half) is staged separately and
gated by ``msp_metric_target`` — added once the channel path is validated.
"""

from __future__ import annotations

from copy import copy

import torch

from ultralytics.models import yolo
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK

from data.gsd import effective_gsd_m_per_px

# Images with no altitude prefix (e.g. VisDrone) get a neutral 0 plane: the
# zero-seeded 4th conv channel makes that a no-op, i.e. "no prior available".
GSD_DEFAULT = 0.0


def _gsd_plane(batch: dict) -> torch.Tensor:
    """A (B,1,H,W) plane filled per-image with effective GSD(im_file, imgsz)."""
    img = batch["img"]
    b, _, h, w = img.shape
    imgsz = max(h, w)  # network input long-side; effective GSD scales to it
    files = batch.get("im_file") or [None] * b
    vals = []
    for f in files:
        g = effective_gsd_m_per_px(f, imgsz, default=GSD_DEFAULT) if f else GSD_DEFAULT
        vals.append(GSD_DEFAULT if g is None else float(g))
    g = torch.tensor(vals, dtype=img.dtype, device=img.device).view(b, 1, 1, 1)
    return g.expand(b, 1, h, w)


def _append_gsd_channel(batch: dict) -> dict:
    if batch["img"].shape[1] == 4:  # idempotent guard
        return batch
    batch["img"] = torch.cat([batch["img"], _gsd_plane(batch)], dim=1)
    return batch


def _seed_first_conv_from_rgb(model: DetectionModel, weights) -> bool:
    """Copy a 3-channel pretrained first-conv into the 4-channel one.

    RGB sub-weights -> channels 0..2; channel 3 (GSD) zero-init. Best-effort:
    returns False (leaving the 4ch conv at its random init) if a [*,3,k,k]
    source conv can't be located, rather than corrupting the build.
    """
    # locate the new model's first conv weight ([C_out, 4, k, k])
    try:
        dst = model.model[0].conv.weight
    except Exception:
        return False
    if dst.shape[1] != 4:
        return False
    # find the source 3-channel conv weight on the pretrained object
    src = None
    src_module = getattr(weights, "model", weights)
    try:
        cand = src_module[0].conv.weight  # type: ignore[index]
        if cand.ndim == 4 and cand.shape[1] == 3:
            src = cand
    except Exception:
        src = None
    if src is None:
        return False
    with torch.no_grad():
        dst.zero_()
        dst[:, :3].copy_(src.to(dst.dtype).to(dst.device))
    return True


class MSPValidator(DetectionValidator):
    """Validator that appends the GSD channel before the 4-channel forward."""

    def preprocess(self, batch: dict) -> dict:
        return _append_gsd_channel(super().preprocess(batch))


class MSPTrainer(DetectionTrainer):
    """Trainer with a 4-channel (RGB+GSD) first conv and GSD-plane injection."""

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        model = DetectionModel(cfg, nc=self.data["nc"], ch=4, verbose=verbose and RANK == -1)
        if weights is not None:
            model.load(weights)  # transfers all matching layers; 4ch conv1 skipped
            seeded = _seed_first_conv_from_rgb(model, weights)
            if RANK in (-1, 0):
                msg = "seeded RGB" if seeded else "RANDOM init (no 3ch source found)"
                print(f"[MSP ii-A] 4-channel first conv: {msg}; 4th (GSD) channel zero-init")
        return model

    def preprocess_batch(self, batch: dict) -> dict:
        return _append_gsd_channel(super().preprocess_batch(batch))

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return MSPValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
