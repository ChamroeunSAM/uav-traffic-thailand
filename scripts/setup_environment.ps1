# Reproduce the project environment from scratch on Windows + RTX 50-series.
# Run from the project root with a Python 3.12 venv activated.

# Step 1: PyTorch with CUDA 12.8 (must come first — these +cu128 wheels are
# only on PyTorch's custom index, not PyPI).
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128

# Step 2: Everything else from the lock file, with --no-deps.
#
# Why --no-deps:
#   The lock is a fully-resolved snapshot (every transitive pin is already
#   listed), so pip's resolver has no real work to do. Skipping it also
#   sidesteps the known boxmot==18.0.0 (requires regex<2025) vs
#   regex==2026.4.4 conflict — transformers needs the newer regex, and the
#   project deliberately overrides boxmot's stale upper bound. With the
#   resolver active this conflict is unresolvable; without it, the explicit
#   lock pins win as intended.
pip install --no-deps -r requirements.lock.txt

# Step 3: Verify
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
