# 教程：训练 SWE Agent

> 训练一个通过多轮工具交互解决软件工程任务的智能体。

本教程涵盖使用 siirl-agentic 训练软件工程（SWE）智能体的端到端流程。该智能体通过在沙箱化的代码环境中执行 bash 命令来学习修复真实的 GitHub issue，RL 奖励来源于自动化测试评估。


## 前置条件

!!! tip "核心要点"
    SWE 智能体训练结合了多轮 rollout（智能体发出命令、观察输出、迭代）和容器化沙箱（Docker 或 Kubernetes）以确保代码执行安全。这与单轮数学/代码 RL 有本质区别。

| 需求     | 最低配置                              | 推荐配置              |
| ------ | --------------------------------- | ----------------- |
| Python | 3.10+                             | 3.12              |
| GPU    | 8x A100 80GB                      | 8x A100 80GB      |
| 容器运行时  | Docker 24+ **或** Kubernetes 1.28+ | Kubernetes（便于规模化） |
| 内存     | 256 GB                            | 512 GB            |
| 磁盘     | 200 GB 可用（容器镜像）                   | 500 GB 可用         |

安装 siirl-agentic 及所有依赖：

```bash
pip install -e "siirl-agentic/[all]"
pip install swebench datasets pyarrow ray[default] sglang jinja2 pydantic
```


## 步骤 1：准备环境

!!! tip "核心要点"
    SWE 模块将三个关注点解耦：**environment**（容器生命周期）、**agent**（rollout 逻辑）和 **runtime**（数据集相关的设置/评估）。这种分离设计在 `swe/README.md` 中有说明。

### 架构概览

SWE 智能体流程由三个可插拔组件组成，通过 `config.yaml` 配置：

| 组件              | 职责                        | 实现                                                                                   |
| --------------- | ------------------------- | ------------------------------------------------------------------------------------ |
| **Environment** | 容器生命周期（启动、执行、清理）          | `DockerEnvBuilder`（`environment/docker.py`）、`K8sEnvBuilder`（`environment/k8s.py`）    |
| **Agent**       | 多轮 rollout：查询模型、解析动作、观察结果 | `MiniSWEAgentBuilder`（`agent/minisweagent.py`）                                       |
| **Runtime**     | 数据集相关：初始化仓库、生成 diff、运行评测  | `SWEBenchBuilder`（`runtime/swebench.py`）、`SWEFactoryBuiler`（`runtime/swefactory.py`） |

### Docker 设置

单节点开发环境中，Docker 是最简单的选择：

```bash
# 验证 Docker 可用
docker info

# 预拉取 SWE-bench 环境镜像（镜像较大）
# 镜像名由 SWE-bench test spec 决定，例如：
docker pull swebench/sweb.env.x86_64:latest
```

`DockerEnvBuilder`（`docker.py:226`）通过 `docker run -d` 创建容器，通过 `docker exec` 执行命令，通过 `docker rm -f` 清理。默认命令超时时间为每个 `ContainerEnv.execute`（`environment/base.py:155`）180 秒。

### Kubernetes 设置

用于大规模训练、需要大量并行沙箱实例的生产场景：

```bash
# 确保 kubectl 已配置
kubectl cluster-info

# 创建专用命名空间
kubectl create namespace swebench
```

`K8sEnvBuilder`（`k8s.py:319`）通过 `kubectl apply` 创建 Pod，通过 `kubectl exec` 执行命令，清理时删除 Pod。`K8sEnvConfig`（`k8s.py:310`）的配置选项：

| 字段                | 默认值           | 说明                       |
| ----------------- | ------------- | ------------------------ |
| `namespace`       | `"default"`   | 沙箱 Pod 的 Kubernetes 命名空间 |
| `kubeconfig`      | `None`        | kubeconfig 文件路径          |
| `context`         | `None`        | 使用的 Kubernetes 上下文       |
| `pod_name_prefix` | `"agentflow"` | 生成的 Pod 名称前缀             |


## 步骤 2：准备数据集

!!! tip "核心要点"
    SWE-bench 实例必须转换为 Parquet 格式。每行必须包含 runtime 初始化容器和运行评测所需的完整实例元数据。

### SWE-bench 数据格式

下载 SWE-bench 并转换为 Parquet：

```python title="prepare_swebench.py"
import json
import pandas as pd
from datasets import load_dataset

# 加载 SWE-bench verified 分割
ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

records = []
for row in ds:
    records.append({
        "prompt": [
            {"role": "user", "content": row["problem_statement"]}
        ],
        "data_source": "swebench",
        "reward_model": {"ground_truth": ""},  # 奖励来自评测，非字符串匹配
        "metadata": {
            "instance_id": row["instance_id"],
            "repo": row["repo"],
            "base_commit": row["base_commit"],
            "patch": row["patch"],
            "test_patch": row["test_patch"],
            "problem_statement": row["problem_statement"],
            "hints_text": row.get("hints_text", ""),
            "version": row.get("version", ""),
        },
    })

df = pd.DataFrame(records)
df.to_parquet("swebench_verified.parquet")
print(f"Prepared {len(df)} SWE-bench instances")
```

runtime 的 `parse_sampledata` 方法（例如 `SWEBenchBuiler.parse_sampledata`，`swebench.py:120`）从每行的 metadata 中提取容器参数和问题描述。`SWESampleData` 数据类（`runtime/base.py:9`）包含：

- `container_args`：`ContainerStartArgs`，指定 Docker 镜像和工作目录（`/testbed`）
- `problem_statement`：GitHub issue 文本，用作初始 prompt
- `runtime_meta`：数据集特定的元数据（SWE-bench test spec）


## 步骤 3：配置 AgentFlow

!!! tip "核心要点"
    SWE 流程配置是一个包含三个部分的 YAML 文件：`environment`、`runtime` 和 `agent`。agent 部分控制步数限制、prompt 模板和输出解析。

### 流程配置

基于 `swe/config.yaml` 参考配置创建 `swe_flow_config.yaml`：

```yaml title="swe_flow_config.yaml"
environment:
  # 选择其一：
  # name: "swe.environment.docker:DockerEnvBuilder"   # Docker 模式
  name: "swe.environment.k8s:K8sEnvBuilder"           # Kubernetes 模式
  namespace: "swebench"

runtime:
  name: "swe.runtime.swebench.SWEBenchBuilder"

agent:
  name: "swe.agent.minisweagent:MiniSWEAgentBuilder"
  step_limit: 30        # 超过此步数强制终止
  cost_limit: 3.0       # 美元成本限制（用于 API 模型）
  system_template: |
    You are a helpful assistant that can interact with a computer.

    Your response must contain exactly ONE bash code block with ONE command
    (or commands connected with && or ||).
    Include a THOUGHT section before your command where you explain your reasoning.

    To finish, issue: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
  instance_template: |
    Please solve this issue: {{task}}

    ## Recommended Workflow
    1. Analyze the codebase by finding and reading relevant files
    2. Create a script to reproduce the issue
    3. Edit the source code to resolve the issue
    4. Verify your fix works by running your script again
    5. Test edge cases to ensure your fix is robust

    ## Important Rules
    1. Every response must contain exactly one action
    2. The action must be enclosed in triple backticks
    3. Directory or environment variable changes are not persistent
    4. To finish: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
  action_observation_template: |
    <returncode>{{output.returncode}}</returncode>
    {% if output.output | length < 10000 -%}
    <output>
    {{ output.output -}}
    </output>
    {%- else -%}
    <warning>
    The output of your last command was too long.
    </warning>
    <output_head>
    {{ output.output[:5000] }}
    </output_head>
    <elided_chars>
    {{ output.output | length - 10000 }} characters elided
    </elided_chars>
    <output_tail>
    {{ output.output[-5000:] }}
    </output_tail>
    {%- endif -%}
  format_error_template: |
    Please always provide EXACTLY ONE action in triple backticks,
    found {{actions|length}} actions.
```

### 关键 Agent 参数

`MiniSWEAgent`（`minisweagent.py:43`）实现了智能体循环：

| 参数                            | 默认值                             | 说明                                                          |
| ----------------------------- | ------------------------------- | ----------------------------------------------------------- |
| `step_limit`                  | `0`（无限制）                        | 最大交互步数；达到时抛出 `LimitsExceeded`                               |
| `cost_limit`                  | `3.0`                           | 美元成本限制（用于 API 模型）                                           |
| `action_regex`                | `` r"```bash\s*\n(.*?)\n```" `` | 从模型输出提取 bash 命令的正则表达式                                       |
| `system_template`             | （见配置）                           | 系统提示的 Jinja2 模板                                             |
| `instance_template`           | （见配置）                           | 初始用户消息的 Jinja2 模板；`{{task}}` 替换为问题描述                        |
| `action_observation_template` | （见配置）                           | 格式化命令输出的 Jinja2 模板；接收 `output.returncode` 和 `output.output` |
| `format_error_template`       | （见配置）                           | 当模型输出不包含恰好一个代码块时显示                                          |

### Agent 循环流程

`MiniSWEAgent.run` 方法（`minisweagent.py:56`）实现以下循环：

1. 将系统消息和实例消息添加到对话中
2. **查询**模型（`query`，第 79 行）：对对话进行 tokenize，调用推理，追加 assistant 响应
3. **解析动作**（`parse_action`，第 106 行）：通过正则表达式提取 bash 命令
4. **在容器中执行**（`execute_action`，第 113 行）：通过 `env.execute` 运行，检查是否提交
5. **观察**（`get_observation`，第 91 行）：通过模板格式化输出，追加为 user 消息
6. 重复直到 `Submitted`（智能体输出 `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`）或 `LimitsExceeded`

智能体根据循环终止方式将 rollout 结果设置为 `SWERolloutResult.SUBMIT`、`EXCEED` 或 `FAILURE`（`base.py:12`）。


## 步骤 4：配置训练

!!! tip "核心要点"
    SWE 智能体训练使用与单轮 GRPO 相同的 `siirl.async_train` 入口，但需要在 `rollout.multiturn.*` 中设置多轮参数，并将 flow function 指向 SWE 配置。

### 多轮 Rollout 参数

这些参数定义在 `MultiturnArguments`（`model_args.py:152`）中：

| 参数路径                                           | 默认值             | SWE 推荐值                    | 说明                |
| ---------------------------------------------- | --------------- | -------------------------- | ----------------- |
| `rollout.multiturn.env_type`                   | `None`          | `"tool_env"`               | 多轮交互的环境类型         |
| `rollout.multiturn.max_env_turns`              | `1`             | `30`                       | 最大环境交互轮数          |
| `rollout.multiturn.max_assistant_turns`        | `1`             | `30`                       | 最大模型生成轮数          |
| `rollout.multiturn.max_parallel_calls`         | `1`             | `32`                       | 并发沙箱实例数           |
| `rollout.multiturn.max_env_response_length`    | `256`           | `2048`                     | 环境观察的最大 token 数   |
| `rollout.multiturn.env_response_truncate_side` | `"middle"`      | `"middle"`                 | 长输出的截断策略          |
| `rollout.flow_function`                        | `"naive"`       | `"swe"`                    | 使用的 flow function |
| `rollout.flow_config`                          | `"config.yaml"` | `swe_flow_config.yaml` 的路径 | flow 配置文件         |

### 完整启动命令

```bash title="run_swe_agent.sh"
#!/usr/bin/env bash
set -euo pipefail

# --- 路径配置 ---
export MODEL_PATH=/path/to/Qwen3-8B
export TRAIN_DATA_PATH=./swebench_verified.parquet

# --- 启动 Ray ---
ray stop --force 2>/dev/null || true
ray start --head --num-gpus 8 \
    --object-store-memory 100000000000 \
    --memory 100000000000

# --- 启动训练 ---
python3 -m siirl.async_train \
    algorithm.adv_estimator=grpo \
    algorithm.gamma=1.0 \
    data.train_files=$TRAIN_DATA_PATH \
    data.train_batch_size=64 \
    data.max_prompt_length=4096 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    actor_ref.model.path=$MODEL_PATH \
    actor_ref.model.trust_remote_code=True \
    actor_ref.actor.optim.lr=5e-7 \
    actor_ref.actor.ppo_mini_batch_size=32 \
    actor_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_ref.actor.use_dynamic_batch=True \
    actor_ref.actor.max_tokens_per_gpu=16384 \
    actor_ref.actor.use_kl_loss=True \
    actor_ref.actor.kl_loss_coef=0.01 \
    actor_ref.actor.kl_loss_type=low_var_kl \
    actor_ref.actor.megatron.param_offload=True \
    actor_ref.actor.megatron.use_mbridge=True \
    actor_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_ref.ref.megatron.param_offload=True \
    rollout.name=sglang \
    rollout.tensor_model_parallel_size=2 \
    rollout.gpu_memory_utilization=0.6 \
    rollout.n=4 \
    rollout.trust_remote_code=True \
    rollout.max_model_len=16384 \
    rollout.flow_function=swe \
    rollout.flow_config=./swe_flow_config.yaml \
    rollout.multiturn.env_type='tool_env' \
    rollout.multiturn.max_env_turns=30 \
    rollout.multiturn.max_assistant_turns=30 \
    rollout.multiturn.max_parallel_calls=32 \
    rollout.multiturn.max_env_response_length=2048 \
    rollout.multiturn.env_response_truncate_side=middle \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.colocate=True \
    trainer.total_epochs=10 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.default_local_dir=ckpts/swe_agent \
    trainer.project_name=siirl_swe_agent \
    trainer.experiment_name=swe_grpo_qwen3_8b \
    trainer.logger="['console','tensorboard','wandb']" \
    trainer.resume_mode=auto \
    trainer.tensor_model_parallel_size=2

ray stop --force
```

### 与单轮训练的关键差异

| 方面                    | 单轮（DeepScaleR） | 多轮（SWE）           |
| --------------------- | --------- | ----------------- |
| Batch size            | 512       | 64（序列更长）          |
| `rollout.n`           | 8         | 4（每次 rollout 成本高） |
| `max_response_length` | 2048      | 8192+（多轮累积）       |
| 学习率                   | 1e-6      | 5e-7（更保守）         |
| 奖励来源                  | 字符串匹配     | 测试套件通过/失败         |
| Rollout 时间            | 秒级        | 分钟级（每个实例）         |


## 步骤 5：启动与监控

!!! tip "核心要点"
    SWE 智能体的 rollout 比单轮生成慢得多。需要同时监控沙箱健康状况和训练指标。每个 rollout 实例在容器内运行完整的智能体循环（最多 `step_limit` 轮）。

### SWE 专用指标

| 指标                     | 说明                 | 健康信号                    |
| ---------------------- | ------------------ | ----------------------- |
| `reward/mean`          | 已解决实例的比例           | 任何上升趋势都是好的；SWE-bench 很难 |
| `rollout/avg_steps`    | 每个实例的平均智能体步数       | 应随训练而降低                 |
| `rollout/submit_rate`  | 到达提交状态的 rollout 比例 | 应增加（相对于超时）              |
| `rollout/timeout_rate` | 达到 step_limit 的比例  | 应随训练而降低                 |
| `kl_divergence`        | 与参考策略的 KL 散度       | 保持在 10 以下               |

### 检查生成的 Patch

runtime 的 `diff` 阶段（`swebench.py:43`）之后，每个样本的 patch 存储在 `SWERolloutMeta.patch` 中。评估阶段（`swebench.py:47`）将 patch 应用到干净的 checkout 上并通过 `get_eval_report` 运行测试套件。完全解决的实例获得奖励 `1.0`；否则为 `0.0`（`swebench.py:88-91`）。

要检查已完成 rollout 的 patch，请查看日志输出或 checkpoint 数据。`SWERolloutResult` 枚举（`base.py:12`）记录每个实例是 `SUBMIT`、`EXCEED`（步数限制）还是 `FAILURE`。


## 预期结果

!!! warning
    SWE-bench 结果对智能体设计、prompt 工程、步数限制和基座模型能力高度敏感。以下范围为近似值，基于公开文献。

| 模型                 | GPU      | 步数限制 | 大致 SWE-bench Verified 解决率 |
| ------------------ | -------- | ---- | ------------------------- |
| Qwen3-8B (GRPO)    | 8x A100  | 30   | 5--15%                    |
| LLaMA-3-70B (GRPO) | 32x A100 | 30   | 15--25%                   |

以上为粗略范围。使用专门的脚手架和更大模型的最先进系统在 SWE-bench Verified 上可达到 30-50%+ 的解决率。


## 常见问题

| 症状                                                    | 原因                         | 修复方法                                                                                  |
| ----------------------------------------------------- | -------------------------- | ------------------------------------------------------------------------------------- |
| `RuntimeError: Container environment is closed`       | 容器在智能体完成前被清理               | 增加 `ContainerStartArgs` 中的 `container_timeout`（默认 `"2h"`）                             |
| 所有 rollout 在 step_limit 处超时                           | `step_limit` 过低，或模型未学会提交命令 | 增加配置中的 `agent.step_limit`；验证系统 prompt 中包含 `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`     |
| `docker exec failed (exit code 127)`                  | 容器内找不到命令                   | 确保 SWE-bench 镜像已安装所需工具；使用 `swebench.harness` 构建镜像                                     |
| `TimeoutError: Command timed out after 180s`          | 单个命令运行时间过长（如完整测试套件）        | 每个 `ContainerEnv.execute` 的默认超时为 180 秒；对于重型测试套件，需在 runtime 中自定义                       |
| `FormatError: EXACTLY ONE action in triple backticks` | 模型输出了多个或零个代码块              | 智能体通过 `NonTerminatingException`（`minisweagent.py:140`）自动重试；若持续出现，改进系统 prompt          |
| `kubectl cp failed` / Pod 创建错误                        | K8s 命名空间不存在或 RBAC 问题       | 验证 `kubectl create namespace swebench`；检查 Pod 创建权限                                    |
| 多轮 rollout 时 OOM                                      | 对话在多轮中累积过长                 | 减小 `rollout.multiturn.max_env_response_length`，设置 `env_response_truncate_side=middle` |
| 奖励始终为 0.0                                             | Patch 未应用或测试失败             | 检查 `SWERolloutResult` -- 如果大多数是 `EXCEED`，增加步数限制；如果是 `SUBMIT` 但 reward=0，检查 patch 质量   |
