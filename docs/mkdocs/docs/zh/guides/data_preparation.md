# 数据准备

*将训练数据准备为 Parquet 格式，确保符合 siirl-agentic 所需的 schema。*

## 数据集 Schema

!!! tip "核心要点"
    siirl-agentic 通过 `DataArguments.train_files` 读取 Parquet 文件。最少需要两列：`prompt`（聊天消息列表）和 `data_source`（映射到奖励函数）。ground truth 存放在 `reward_model` 字典列**内部**，而**不是**顶层列。

`PartitionedRLHFDataset`（`partitioned_dataset.py`）加载 Parquet 文件，并将每一行转换为 `Sample`（`sample.py`）。以下表格列出所有可识别的列：

| 列名             | 类型             | 是否必需  | 配置键                                | 说明                                                                       |
| -------------- | -------------- | ----- | ---------------------------------- | ------------------------------------------------------------------------ |
| `prompt`       | `list[dict]`   | **是** | `data.prompt_key="prompt"`         | OpenAI 聊天格式消息，如 `[{"role": "user", "content": "..."}]`                   |
| `data_source`  | `str`          | **是** | `data.reward_fn_key="data_source"` | 标识符，用于分发到对应的奖励函数（参见[自定义奖励](custom_rewards.md)）                           |
| `reward_model` | `dict`         | 推荐    | —                                  | 包含 `ground_truth` 键的字典。在奖励计算时通过 `sample.reward_model["ground_truth"]` 访问 |
| `extra_info`   | `dict` 或 `str` | 可选    | —                                  | 传递给奖励函数的任意元数据                                                            |

!!! warning "常见陷阱：`ground_truth` 的位置"
    `ground_truth` 是 `reward_model` 字典**内部**的一个键，不是 Parquet 的顶层列。如果放在顶层，奖励函数会抛出 `KeyError`。

### 最小 Parquet 示例

一行有效的训练数据在 Python 中如下所示：

```python
{
    "prompt": [
        {"role": "user", "content": "2 + 3 等于多少？"}
    ],
    "data_source": "openai/gsm8k",
    "reward_model": {"ground_truth": "5"},
    "extra_info": {}
}
```

## DataArguments 参考

!!! tip "核心要点"
    所有数据配置都在 `data.*` 命名空间下。通过 CLI 点分表示法传递参数，例如 `data.train_files=/path/to/train.parquet data.max_prompt_length=1024`。

`DataArguments`（`data_args.py`）控制数据的加载和预处理：

| 参数                          | 类型          | 默认值                                   | 说明                                                          |                                            |
| --------------------------- | ----------- | ------------------------------------- | ----------------------------------------------------------- | ------------------------------------------ |
| `train_files`               | `list[str]` | `["~/data/rlhf/gsm8k/train.parquet"]` | 训练 Parquet 文件路径（也接受逗号分隔的字符串）                                |                                            |
| `val_files`                 | `list[str]` | `["~/data/rlhf/gsm8k/test.parquet"]`  | 验证 Parquet 文件路径                                             |                                            |
| `prompt_key`                | `str`       | `"prompt"`                            | Parquet 中 prompt 列的名称                                       |                                            |
| `reward_fn_key`             | `str`       | `"data_source"`                       | 奖励函数分发所用的列名                                                 |                                            |
| `max_prompt_length`         | `int`       | `512`                                 | prompt 的最大 token 长度                                         |                                            |
| `max_response_length`       | `int`       | `512`                                 | response 的最大 token 长度                                       |                                            |
| `train_batch_size`          | `int`       | `1024`                                | 全局训练批次大小（在 DDP rank 之间均分）                                   |                                            |
| `val_batch_size`            | `int \      | None`                                 | `None`                                                      | 验证批次大小；`None` 表示使用整个验证集                    |
| `gen_batch_size`            | `int \      | None`                                 | `None`                                                      | DAPO 的生成批次大小（通常为 `train_batch_size` 的 3 倍） |
| `return_raw_chat`           | `bool`      | `True`                                | 在 `Sample.raw_prompt` 中保留原始聊天消息                             |                                            |
| `filter_overlong_prompts`   | `bool`      | `False`                               | 在分词前过滤超过 `max_prompt_length` 的 prompt                       |                                            |
| `shuffle`                   | `bool`      | `True`                                | 是否打乱训练数据                                                    |                                            |
| `truncation`                | `str`       | `"error"`                             | 截断策略：`"error"`、`"left"`、`"right"` 或 `"middle"`              |                                            |
| `streaming`                 | `bool`      | `False`                               | 启用流式数据集模式                                                   |                                            |
| `force_on_the_fly`          | `bool`      | `False`                               | 即时预处理（为大数据集节省内存）                                            |                                            |
| `mix_strategy`              | `str`       | `"concat"`                            | 数据集混合策略：`"concat"`、`"interleave_under"`、`"interleave_over"` |                                            |
| `interleave_probs`          | `str \      | None`                                 | `None`                                                      | 逗号分隔的交错采样概率                                |
| `auto_repeat`               | `bool`      | `False`                               | 数据集小于批次大小时自动重复                                              |                                            |
| `num_loader_workers`        | `int`       | `8`                                   | DataLoader 工作线程数                                            |                                            |
| `preprocessing_num_workers` | `int \      | None`                                 | `None`                                                      | 预处理并行度；`None` 使用 `cpu_count // 8`          |
| `preprocessing_batch_size`  | `int`       | `1000`                                | 每个预处理组的样本数                                                  |                                            |
| `cutoff_len`                | `int`       | `2048`                                | 分词后输入的截断长度                                                  |                                            |
| `buffer_size`               | `int`       | `16384`                               | 流式随机采样的缓冲区大小                                                |                                            |
| `train_on_prompt`           | `bool`      | `False`                               | 禁用 prompt token 上的 loss mask                                |                                            |
| `mask_history`              | `bool`      | `False`                               | 遮蔽对话历史，仅在最后一轮上训练（与 `train_on_prompt` 不兼容）                   |                                            |
| `tokenized_path`            | `str \      | None`                                 | `None`                                                      | 缓存/加载分词后数据集的路径                             |
| `overwrite_cache`           | `bool`      | `False`                               | 是否覆盖已缓存的分词数据集                                               |                                            |

## 从常见格式转换

### 从 HuggingFace Datasets 转换

```python
from datasets import load_dataset

# 加载 HuggingFace 数据集
ds = load_dataset("openai/gsm8k", "main", split="train")

# 映射到 siirl-agentic schema
def convert_row(row):
    return {
        "prompt": [{"role": "user", "content": row["question"]}],
        "data_source": "openai/gsm8k",
        "reward_model": {"ground_truth": row["answer"].split("####")[-1].strip()},
        "extra_info": {},
    }

converted = ds.map(convert_row, remove_columns=ds.column_names)
converted.to_parquet("gsm8k_train.parquet")
```

### 从 JSON / JSONL 转换

```python
import json
import pyarrow as pa
import pyarrow.parquet as pq

records = []
with open("data.jsonl") as f:
    for line in f:
        obj = json.loads(line)
        records.append({
            "prompt": [{"role": "user", "content": obj["question"]}],
            "data_source": obj.get("source", "custom"),
            "reward_model": {"ground_truth": obj["answer"]},
            "extra_info": obj.get("metadata", {}),
        })

table = pa.Table.from_pylist(records)
pq.write_table(table, "data.parquet")
```

### 多轮对话格式

siirl-agentic 要求使用 OpenAI 聊天格式——一个 `{"role": ..., "content": ...}` 字典的列表。对于多轮对话，需要包含完整的历史记录：

```python
{
    "prompt": [
        {"role": "system", "content": "你是一个乐于助人的数学辅导老师。"},
        {"role": "user", "content": "什么是微积分？"},
        {"role": "assistant", "content": "微积分是研究变化的数学学科……"},
        {"role": "user", "content": "能给我举个例子吗？"},
    ],
    "data_source": "custom",
    "reward_model": {"ground_truth": "任何有效的微积分示例"},
}
```

当 `data.return_raw_chat=True`（默认值）时，原始消息列表会保留在 `Sample.raw_prompt` 中，供奖励函数和多轮 rollout 使用。

## 数据集混合

!!! tip "核心要点"
    使用数据集混合可以在单次训练中组合多个数据源。`mix_strategy` 控制数据集的组合方式，`interleave_probs` 允许你设置权重。

当 `data.train_files` 包含多个 Parquet 路径时，`mix_strategy` 参数控制它们的组合方式：

| 策略                 | 行为               | 适用场景         |
| ------------------ | ---------------- | ------------ |
| `concat`           | 将所有数据集首尾拼接       | 默认；简单且可预测    |
| `interleave_under` | 轮询采样，在最短数据集处停止   | 防止较大数据集过度占比  |
| `interleave_over`  | 轮询采样，对较短数据集进行过采样 | 确保所有数据源被均等使用 |

**示例：加权交错**

```bash
python -m siirl.async_train \
    data.train_files="math_train.parquet,code_train.parquet" \
    data.mix_strategy=interleave_over \
    data.interleave_probs="0.7,0.3"
```

!!! note "验证约束"
    使用 `interleave_probs` 时，概率的数量必须与 `train_files` 和 `val_files` 中的文件数量匹配。概率同时应用于训练和验证数据加载。

## 数据加载流水线

siirl-agentic 中的数据加载流水线工作流程如下：

1. **Parquet 文件**由 `PartitionedRLHFDataset`（`partitioned_dataset.py`）加载，它只读取属于当前 DDP rank 的行组
2. **超长 prompt** 根据 `max_prompt_length` 进行可选过滤
3. 使用模型的 tokenizer 通过 `apply_chat_template` 和左填充进行**分词**
4. 从 attention mask 计算 **Position ID**
5. `DataLoaderNode`（`data_loader_node.py`）将数据集包装在 `StatefulDataLoader` 中，使用配置的采样器和批次大小
6. 每个批次被转换为 `TensorDict`，然后通过 `Dict2Samples`（`sample.py`）转换为 `Sample` 对象列表

## 验证与常见错误

!!! tip "核心要点"
    大多数数据问题会在第一个训练步骤中以晦涩的错误形式出现。在启动分布式运行之前，先验证你的 Parquet schema。

**快速验证脚本：**

```python
import pyarrow.parquet as pq

pf = pq.ParquetFile("train.parquet")
schema = pf.schema_arrow
print(schema)

# 读取第一行以验证结构
table = pf.read_row_group(0)
row = table.to_pydict()
print("prompt 类型:", type(row["prompt"][0]))
print("reward_model 键:", row["reward_model"][0].keys() if "reward_model" in row else "缺失")
```

| 症状                                                                  | 原因                                                     | 修复方法                                                   |
| ------------------------------------------------------------------- | ------------------------------------------------------ | ------------------------------------------------------ |
| `KeyError: 'prompt'`                                                | 列名不匹配                                                  | 设置 `data.prompt_key=你的列名`                              |
| `KeyError: 'ground_truth'`                                          | `ground_truth` 放在顶层列而不是 `reward_model` 字典内部            | 重组数据：`{"reward_model": {"ground_truth": ...}}`         |
| 预处理时 OOM                                                            | 整个数据集加载到内存                                             | 设置 `data.force_on_the_fly=True`                        |
| `RuntimeError: Prompt length ... is longer than ...`                | prompt 超过 `max_prompt_length` 且默认 `truncation="error"` | 增加 `data.max_prompt_length` 或设置 `data.truncation=left` |
| `AssertionError: Not enough data for current rank`                  | 对于 GPU 数量来说行数太少                                        | 添加更多数据或减少 GPU 数量                                       |
| `ValueError: mask_history is incompatible with train_on_prompt`     | 两个标志同时启用                                               | 只使用 `data.mask_history` 或 `data.train_on_prompt` 之一    |
| `ValueError: interleave_probs is only valid for interleaved mixing` | `interleave_probs` 设置时使用了 `mix_strategy=concat`        | 改为 `mix_strategy=interleave_under` 或 `interleave_over` |
| 不同数据源的奖励分布不均匀                                                       | 使用 `concat` 时各数据集大小差异很大                                | 切换到 `interleave_over` 并设置适当的 `interleave_probs`        |

## 成功的标志

数据成功加载后，你应该看到类似以下的日志消息：

```
INFO     | DataLoaderNode initialized:
INFO     |   Group rank: 0 / 8
INFO     |   Rollout DDP rank: 0 / 2
INFO     |   Train batches per epoch for this rank: 24
INFO     |   Total training steps (approx): 720
```

如果设置了 `filter_overlong_prompts=True`，你还会看到：

```
INFO     | Filtered prompts from 10000 to 9847 on each rank.
```
