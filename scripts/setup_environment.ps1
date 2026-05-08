# Reproduce the project environment from scratch on Windows + RTX 50-series
# Run this from the project root with Python 3.12 venv activated

# Step 1: PyTorch with CUDA 12.8 (must come first)
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128

# Step 2: Everything else from the lock file
pip install -r requirements.lock.txt

# Step 3: Apply the regex fix for transformers/boxmot conflict
pip install regex==2026.4.4 --upgrade

# Step 4: Verify
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"