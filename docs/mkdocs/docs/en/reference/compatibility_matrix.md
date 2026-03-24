# Compatibility Matrix

*Supported versions of Python, PyTorch, CUDA, and key dependencies for siirl-agentic.*

## Python

| Python Version | Status        |
| -------------- | ------------- |
| 3.10           | Supported     |
| 3.11           | Supported     |
| 3.12           | Supported     |
| < 3.10         | Not supported |

## PyTorch

| PyTorch Version | Status        | Notes                       |
| --------------- | ------------- | --------------------------- |
| >= 2.8.0        | Supported     | Required minimum            |
| 2.7.x           | Untested      | May work but not guaranteed |
| < 2.7           | Not supported |                             |

## CUDA

| CUDA Version | Status      | Notes                             |
| ------------ | ----------- | --------------------------------- |
| 12.x         | Supported   | Recommended                       |
| 11.8         | Best-effort | Requires compatible PyTorch build |

## Key Dependencies

```yaml title="Core Dependencies"
ray[default]         >= 2.53.0     # Distributed actor orchestration
transformers         >= 4.57.0     # Model loading, tokenization
sglang               >= 0.5.5      # Inference engine for rollout
sglang-router        >= 0.3.0      # Multi-engine load balancing
accelerate           >= 1.12.0     # Model parallelism utilities
datasets             >= 4.4.0      # Dataset loading
wandb                >= 0.23.0     # Experiment tracking
tensorboard          >= 2.20.0     # Training visualization
loguru               >= 0.7.3      # Structured logging
fastapi              >= 0.127.0    # AIO tool server API
aiohttp              >= 3.13.0     # Async HTTP client
hydra-core           >= 1.3.2      # Configuration management
omegaconf            (via hydra)   # CLI dot-notation config parsing
```

## Optional GPU Dependencies

```yaml title="GPU Dependencies (pip install siirl-agentic[gpu])"
flash-attn           >= 2.8.2      # FlashAttention-2 acceleration
triton               >= 3.4.0      # Custom kernel compilation
torch-memory-saver   >= 0.0.9      # GPU memory optimization (colocated mode)
```

## Hardware Requirements

### Minimum (Development / Testing)

-   1× NVIDIA GPU (A100 40GB or equivalent)
-   64 GB system RAM
-   100 GB disk space

### Recommended (Production Training)

-   8× NVIDIA A100 80GB or H100
-   512 GB system RAM per node
-   High-speed interconnect (NVLink / InfiniBand)
-   Shared filesystem (NFS or parallel FS)

### Multi-Node

-   All nodes must have identical GPU configurations
-   Ray cluster must be pre-initialized or auto-started
-   Network latency < 1ms between nodes (same datacenter)

## Model Support

| Model Family             | Status      | Notes                                          |
| ------------------------ | ----------- | ---------------------------------------------- |
| Qwen2 / Qwen2.5 / Qwen3  | Supported   | Primary development target                     |
| LLaMA 3 / 3.1            | Supported   |                                                |
| DeepSeek-V2/V3           | Supported   | MoE architecture                               |
| Other HuggingFace models | Best-effort | Requires compatible tokenizer and architecture |

## Deployment Modes

| Mode      | GPU Requirement           | Description                                     |
| --------- | ------------------------- | ----------------------------------------------- |
| Separated | actor_gpus + rollout_gpus | Dedicated GPUs for training and rollout         |
| Colocated | Shared GPU pool           | Actor and rollout share GPUs with time-division |

See [Deployment Modes](../guides/deployment_modes.md) for detailed configuration.

## Related

-   [Installation](../get_started/installation.md) — Step-by-step setup instructions
-   [Config Reference](config_reference.md) — Full parameter documentation
