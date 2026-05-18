# Machine Transfer Procedure

Validated checklist for setting up the project on a new Windows + NVIDIA
CUDA-12-class machine (e.g. RTX 50-series rentals, reinstalls, collaborator
boxes). Order matters.

1. **Clone the repo**

   ```
   git clone <repo-url> uav-traffic-thailand
   cd uav-traffic-thailand
   ```

2. **Manually copy source annotations**

   Copy the Roboflow exports under `data/annotations/100m/` and
   `data/annotations/120m/` from the source machine. These are the *only*
   data assets that must travel by hand — everything else under `data/`
   is regenerable.

3. **Create venv and install dependencies**

   ```
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   scripts\setup_environment.ps1
   ```

   The script installs PyTorch `+cu128` from the PyTorch index, then the
   rest of `requirements.lock.txt` with `--no-deps` (the lock is
   fully-resolved; the flag avoids re-litigating the boxmot/regex pin
   conflict).

4. **Regenerate path-dependent artifacts — REQUIRED**

   ```
   python scripts\00_prepare_splits.py --force
   ```

   `data/splits/dataset.yaml` and the merged dataset directory contain
   absolute paths baked in at generation time. They do not survive a
   transfer (different usernames, drives, project roots). Running the
   prep script regenerates `data/annotations/merged/` and
   `data/splits/{dataset.yaml,altitude_index.json,split_metadata.json}`
   with paths correct for the current machine.

5. **Validate with the pilot**

   ```
   python scripts\01_train_detector.py --config configs\detectors\pilot_yolo11n.yaml
   python scripts\01_eval_detector.py ^
     --weights runs\stage1_detectors\pilot_yolo11n_seed42\weights\best.pt ^
     --config configs\detectors\pilot_yolo11n.yaml
   ```

   On healthy hardware, training completes in well under two minutes
   and `eval_results.json` is written with stratified `overall / 100m /
   120m` blocks. Any failure here means the environment is wrong, not
   the methodology.

## Notes

- **MLflow**: Ultralytics' MLflow callback crashes on Windows paths
  (it feeds a bare `C:\...` path to mlflow, which parses `C` as an
  unknown URI scheme). The wrapper at `src/detectors/ultralytics_wrapper.py`
  disables it programmatically on first use, so no machine-local
  `~/.config/Ultralytics/settings.json` edit is needed.
- **No additional manual fixes**: as of 2026-05-16, this checklist is
  complete — no extra environment variables, registry tweaks, or
  package overrides are required beyond what the setup script and
  prep script perform.
