# Installation Guide

This guide covers different installation methods for siirl-agentic.

## Requirements

- Python >= 3.10
- CUDA >= 11.8 (for GPU support)
- Ray >= 2.47.1

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
- Testing tools (pytest, pytest-cov)
- Code quality tools (ruff, black, isort)
- Pre-commit hooks

### 3. GPU Installation

For GPU acceleration with Flash Attention and other optimizations:

```bash
pip install "siirl-agentic[gpu]"
```

This includes:
- flash-attn
- liger-kernel
- triton

### 4. SGLang Backend

For SGLang-based asynchronous inference:

```bash
pip install "siirl-agentic[sglang]"
```

This includes:
- sglang[all]
- torch-memory-saver

### 5. Full Installation

Install with all optional dependencies:

```bash
pip install "siirl-agentic[all]"
```

Or for development with all features:

```bash
pip install -e ".[all,dev]"
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

### 3. Run Tests

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
python -c "import torch; print(torch.cuda.is_available())"
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
# Install from source with specific CUDA version
pip install flash-attn --no-build-isolation
```

**4. Import errors**
```bash
# Verify package is correctly installed
pip show siirl-agentic
# Reinstall if needed
pip install --force-reinstall siirl-agentic
```

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

### Linux

Recommended for production use. Ensure CUDA drivers are properly installed:

```bash
# Ubuntu/Debian
apt-get install cuda-toolkit-11-8

# Check installation
nvcc --version
```

## Getting Help

- Documentation: https://siirl-agentic.readthedocs.io
- Issues: https://github.com/sii-research/siirl-agentic/issues
- Discussions: https://github.com/sii-research/siirl-agentic/discussions
