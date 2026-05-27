"""Train ONE detector with ONE config and ONE seed.

Usage
-----
    python scripts/01_train_detector.py --config configs/detectors/pilot_yolo11n.yaml
    python scripts/01_train_detector.py --config configs/detectors/yolo11m.yaml --seed 0
    python scripts/01_train_detector.py --config configs/detectors/yolo11m.yaml --seed 0 --resume

Behaviour
---------
1. Loads YAML config; --seed overrides config.seed.
2. Verifies CUDA availability before instantiating the model (fail-fast).
3. Picks the right wrapper (ultralytics vs rfdetr) from `detector_class`.
4. Substitutes {seed} in `name` so each run gets its own output dir.
5. Writes results.json in the run dir capturing config, metrics, duration,
   git SHA, and timestamp. Evaluation is a separate script.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.io import get_git_commit, load_config, save_json  # noqa: E402
from utils.logging import configure_logging  # noqa: E402

logger = logging.getLogger("train_detector")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train one detector for Stage 1.")
    p.add_argument("--config", required=True, type=Path, help="Path to detector YAML config.")
    p.add_argument("--seed", type=int, default=None, help="Override config.seed for this run.")
    p.add_argument("--resume", action="store_true", help="Resume from existing partial run if found.")
    # Smoke-test / ad-hoc overrides. None = keep config value.
    p.add_argument("--epochs", type=int, default=None, help="Override config.epochs (e.g. 1 for smoke tests).")
    p.add_argument("--fraction", type=float, default=None, help="Fraction of dataset (0 < f <= 1) for smoke tests.")
    p.add_argument("--project", type=str, default=None, help="Override config.project (output root).")
    p.add_argument("--name", type=str, default=None, help="Override config.name (may omit the {seed} placeholder).")
    return p.parse_args()


def verify_gpu() -> str:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Stage 1 training requires a CUDA GPU. "
            "Check `nvidia-smi` and the torch install (see README)."
        )
    name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    logger.info("GPU OK: %s (compute capability %d.%d), torch=%s", name, cc[0], cc[1], torch.__version__)
    return name


def build_wrapper(detector_class: str, config: dict):
    detector_class = detector_class.lower()
    if detector_class == "ultralytics":
        from detectors.ultralytics_wrapper import UltralyticsWrapper

        return UltralyticsWrapper(config)
    if detector_class == "rfdetr":
        from detectors.rfdetr_wrapper import RFDETRWrapper

        return RFDETRWrapper(config)
    raise ValueError(f"Unknown detector_class: {detector_class!r}")


def main() -> int:
    configure_logging()
    args = parse_args()

    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    if args.resume:
        config["resume"] = True
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.fraction is not None:
        config["fraction"] = args.fraction
    if args.project is not None:
        config["project"] = args.project
    if args.name is not None:
        config["name"] = args.name

    raw_name: str = config["name"]
    config["name"] = raw_name.format(seed=config["seed"])

    project_dir = PROJECT_ROOT / config["project"]
    output_dir = project_dir / config["name"]

    if not args.resume and output_dir.exists() and any(output_dir.iterdir()):
        logger.warning("Output dir %s is non-empty (no --resume). Files may be overwritten.", output_dir)

    gpu_name = verify_gpu()

    print("=" * 78)
    print(f"Stage 1 training run")
    print(f"  config       : {args.config}")
    print(f"  detector     : {config['detector_class']} / {config['model_name']}")
    print(f"  seed         : {config['seed']}")
    print(f"  epochs       : {config['epochs']} (patience {config.get('patience', 0)})")
    print(f"  imgsz        : {config['imgsz']}")
    print(f"  batch        : {config['batch']}")
    print(f"  data_yaml    : {config['data_yaml']}")
    print(f"  output_dir   : {output_dir}")
    print(f"  gpu          : {gpu_name}")
    print("=" * 78)

    wrapper = build_wrapper(config["detector_class"], config)
    train_result = wrapper.train(output_dir)

    record = {
        "config_path": str(args.config),
        "config": config,
        "final_metrics": train_result["final_metrics"],
        "training_time_seconds": train_result["training_time_seconds"],
        "best_weights_path": str(
            Path(train_result["best_weights_path"]).resolve().relative_to(PROJECT_ROOT)
        )
        if Path(train_result["best_weights_path"]).is_absolute()
        and Path(train_result["best_weights_path"]).resolve().is_relative_to(PROJECT_ROOT)
        else str(train_result["best_weights_path"]),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(PROJECT_ROOT),
        "gpu": gpu_name,
    }
    save_json(record, output_dir / "results.json")

    print()
    print("Training complete.")
    print(f"  best weights : {train_result['best_weights_path']}")
    print(f"  duration     : {train_result['training_time_seconds']:.1f} s")
    print(f"  metrics      : {train_result['final_metrics']}")
    print(f"  results.json : {output_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
