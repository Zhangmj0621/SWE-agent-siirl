# Installation

*Get siirl-agentic installed and verified on your system.*

## Requirements

-   Python >= 3.10
-   CUDA >= 12.4 (for GPU support)
-   PyTorch >= 2.8.0
-   Ray >= 2.53.0

### Tested Environment

```text
Package              Version
─────────────────────────────
Python               3.10+
PyTorch              2.8.0
CUDA                 12.8
Flash Attention      2.8.2
Transformers         4.57.0
Ray                  2.53.0
SGLang               0.5.5.post3
Triton               3.4.0
```

!!! note
    The minimum required CUDA version is **12.4**. The tested version above (12.8) is the version validated by the development team. Other CUDA 12.x versions should work but are not officially tested.

!!! warning "Python Version Requirement"
    siirl-agentic requires **Python >= 3.10**. Check your version before installing:
    ```bash
    python --version
    ```
    If you see `Python 3.9.x` or earlier, upgrade your environment before proceeding.

!!! tip "CUDA 12.x Cluster Users"
    If you're on a cluster with CUDA 12.x, install with `pip install -e '.[gpu]'` to get flash-attn and triton. For CPU-only exploration, the base install (`pip install -e '.[dev]'`) works without GPU extras.

## Install from Source

```bash
git clone https://github.com/sii-research/siirl-agentic.git
cd siirl-agentic
```

=== "Dev (recommended)"

    ```bash
    pip install -e ".[dev]"
    ```

=== "GPU (Flash Attention + Triton)"

    ```bash
    pip install -e ".[gpu]"
    ```

=== "All dependencies"

    ```bash
    pip install -e ".[all]"
    ```

=== "Pinned versions"

    ```bash
    pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
    pip install flash-attn==2.8.2 --no-build-isolation
    pip install -e ".[dev]"
    ```

## Verify Installation

Run all three checks to confirm the installation is complete and your environment is ready for training:

``` bash
# 1. Check siirl package version
python -c "import siirl; print(siirl.__version__)"

# 2. Check CUDA availability and GPU count
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# 3. Verify Ray cluster management
ray start --head && ray status && ray stop
```

**Expected output for step 1:**

```
0.1.0
```

!!! note
    The version is managed by `setuptools_scm` and derived from git tags. If you installed from a git clone, you may see a version like `0.1.0.dev123+gabcdef` instead of `0.1.0`. This is normal.

**Expected output for step 2:**

```
CUDA: True, GPUs: 8
```

The GPU count should match the number of physical GPUs on your machine. If `CUDA: False`, check that your CUDA drivers are installed and that you ran `pip install -e '.[gpu]'` rather than the base dev install.

**Expected output for step 3:**

```
Local node IP: 127.0.0.1
...
Node status
---------------------------------------------------------------
Active:
 1 node(s) with resources: ...
Stopped the local Ray instance.
```

``` bash
# Additional check: GPU support with CUDA version
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, version: {torch.version.cuda}')"
```

**Expected output:**

```
CUDA: True, version: 12.8
```

!!! tip "Python Version Matters"
    Python 3.10 is the minimum, but Python 3.11 or 3.12 is recommended for better performance. If you see `SyntaxError` or import issues, check your Python version with `python --version` before troubleshooting further.

!!! tip "CPU-Only Install"
    If you don't have a GPU, install the base package without GPU extras: `pip install -e ".[dev]"`. This is useful for config validation and unit tests. Skip the Flash Attention step — it requires a CUDA-capable GPU at build time.

## Environment Variables

``` bash
# Logging
export LOGURU_LEVEL=INFO
export SIIRL_LOG_DIRECTORY=siirl_logs

# Ray
export RAY_memory_monitor_refresh_ms=0

# Distributed Training
export MASTER_ADDR=localhost
export MASTER_PORT=29500
```

## What success looks like

After completing this guide, you should see:

- `python -c "import siirl; print(siirl.__version__)"` prints `0.1.0` (or a dev variant like `0.1.0.dev123+gabcdef`)
- `python -c "import torch; print(torch.cuda.is_available())"` prints `True`
- No import errors or missing dependency warnings
- Flash Attention loads without error: `python -c "import flash_attn; print(flash_attn.__version__)"`

If something went wrong, see [Troubleshooting](../reference/troubleshooting.md).

## Next steps

- [Quickstart](quickstart.md) — Run your first GRPO training job to verify the installation end-to-end
- [How It Works](../concepts/how_it_works.md) — Understand the async training loop and key components before diving into configuration
