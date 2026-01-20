# Installation Guide

This guide covers different installation methods for siirl-agentic.

## Requirements

- Python >= 3.10
- CUDA >= 12.4 (for GPU support)
- PyTorch >= 2.8.0
- Ray >= 2.53.0

### Tested Environment

The following versions have been tested and verified:

| Package | Version |
|---------|---------|
| Python | 3.10+ |
| PyTorch | 2.8.0 |
| CUDA | 12.8 |
| Flash Attention | 2.8.2 |
| Transformers | 4.57.0 |
| Ray | 2.53.0 |
| SGLang | 0.5.5.post3 |
| Triton | 3.4.0 |

## Installation Methods

### 1. Basic Installation

Install the core package with basic dependencies:

```bash
pip install siirl-agentic
```

### 2. Development Installation

For development, clone the repository and install in editable mode:

```bash
git clone https://github.com/sii-research/siirl-agentic.git
cd siirl-agentic
pip install -e ".[dev]"
```

This installs the package with all development dependencies including:
- Testing tools (pytest, pytest-cov, pytest-asyncio)
- Code quality tools (ruff, black, isort)
- Pre-commit hooks
- IPython

### 3. GPU Installation

For GPU acceleration with Flash Attention and Triton:

```bash
pip install "siirl-agentic[gpu]"
```

This includes:
- flash-attn >= 2.8.2
- triton >= 3.4.0

### 4. SGLang Extra Components

For additional SGLang features:

```bash
pip install "siirl-agentic[sglang-extra]"
```

This includes:
- sglang[all] >= 0.5.5
- torch-memory-saver >= 0.0.9

### 5. Full Installation

Install with all optional dependencies:

```bash
pip install "siirl-agentic[all]"
```

Or for development with all features:

```bash
pip install -e ".[all,dev]"
```

## From Source with Specific Versions

If you need to install with specific tested versions:

```bash
# Clone repository
git clone https://github.com/sii-research/siirl-agentic.git
cd siirl-agentic

# Install PyTorch first (CUDA 12.4+)
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

# Install Flash Attention
pip install flash-attn==2.8.2 --no-build-isolation

# Install the package
pip install -e ".[dev]"
```

## Post-Installation Setup

### 1. Set up Pre-commit Hooks (Development)

```bash
pre-commit install
```

### 2. Verify Installation

```bash
python -c "import siirl; print(siirl.__version__)"
```

### 3. Verify GPU Support

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

### 4. Run Tests

```bash
pytest tests/
```

## Environment Configuration

Create a `.env` file or set environment variables:

```bash
# Logging
export LOGURU_LEVEL=INFO
export SIIRL_LOG_DIRECTORY=siirl_logs
export SIIRL_LOGGING_FILENAME=siirl

# Ray
export RAY_memory_monitor_refresh_ms=0

# Distributed Training
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# CUDA (if needed)
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

## Docker Installation

A Dockerfile is available for containerized deployment:

```bash
docker build -t siirl-agentic .
docker run --gpus all -it siirl-agentic
```

## Troubleshooting

### Common Issues

**1. CUDA/cuDNN version mismatch**
```bash
# Check versions
nvidia-smi
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"
```

**2. Ray connection issues**
```bash
# Stop all Ray processes
ray stop
# Restart Ray
ray start --head
```

**3. Flash Attention installation fails**
```bash
# Ensure CUDA toolkit is installed
nvcc --version

# Install with no build isolation
pip install flash-attn>=2.8.2 --no-build-isolation

# Or install from source
pip install git+https://github.com/Dao-AILab/flash-attention.git
```

**4. Import errors**
```bash
# Verify package is correctly installed
pip show siirl-agentic
# Reinstall if needed
pip install --force-reinstall siirl-agentic
```

**5. SGLang installation issues**
```bash
# Install SGLang with all components
pip install "sglang[all]>=0.5.5"

# Or install specific version
pip install sglang==0.5.5.post3
```

**6. Triton compilation errors**
```bash
# Clear Triton cache
rm -rf ~/.triton/cache

# Reinstall Triton
pip install --force-reinstall triton>=3.4.0
```

## Version Compatibility Matrix

| siirl-agentic | PyTorch | CUDA | Python | Transformers |
|---------------|---------|------|--------|--------------|
| 0.1.x | 2.8.x | 12.4+ | 3.10+ | 4.57+ |

## Upgrading

To upgrade to the latest version:

```bash
pip install --upgrade siirl-agentic
```

For development installation:

```bash
cd siirl-agentic
git pull
pip install -e ".[dev]"
```

## Uninstallation

```bash
pip uninstall siirl-agentic
```

## Platform-Specific Notes

### macOS

Flash Attention is not available on macOS. Install without GPU extras:

```bash
pip install siirl-agentic
```

### Windows

Some dependencies may require Microsoft C++ Build Tools. Install from:
https://visualstudio.microsoft.com/visual-cpp-build-tools/

Note: Full GPU support may be limited on Windows.

### Linux (Recommended)

Recommended for production use. Ensure CUDA drivers are properly installed:

```bash
# Ubuntu/Debian (CUDA 12.4)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-12-4

# Check installation
nvcc --version
nvidia-smi
```

## Getting Help

- Documentation: https://siirl-agentic.readthedocs.io
- Issues: https://github.com/sii-research/siirl-agentic/issues
- Discussions: https://github.com/sii-research/siirl-agentic/discussions
