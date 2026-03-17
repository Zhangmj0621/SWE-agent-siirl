# Data Preparation

*Prepare your training data in Parquet format with the right schema for siirl-agentic.*

## Dataset Schema

!!! tip "Key Insight"
    siirl-agentic reads Parquet files via `DataArguments.train_files`. The minimum required columns are `prompt` (list of chat messages) and `data_source` (maps to the reward function). Ground truth lives inside the `reward_model` dict column, **not** as a top-level column.

The `PartitionedRLHFDataset` (`partitioned_dataset.py`) loads Parquet files and converts each row into a `Sample` (`sample.py`). The following table lists all recognized columns:

| Column         | Type            | Required    | Config Key                         | Description                                                                                                           |
| -------------- | --------------- | ----------- | ---------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `prompt`       | `list[dict]`    | **Yes**     | `data.prompt_key="prompt"`         | OpenAI chat-format messages, e.g. `[{"role": "user", "content": "..."}]`                                              |
| `data_source`  | `str`           | **Yes**     | `data.reward_fn_key="data_source"` | Identifier that dispatches to the correct reward function (see [Custom Rewards](custom_rewards.md))                   |
| `reward_model` | `dict`          | Recommended | —                                  | Dictionary containing `ground_truth` key. Accessed as `sample.reward_model["ground_truth"]` during reward computation |
| `extra_info`   | `dict` or `str` | Optional    | —                                  | Arbitrary metadata passed through to the reward function as `extra_info`                                              |

!!! warning "Common Pitfall: `ground_truth` Location"
    `ground_truth` is a key **inside** the `reward_model` dict, not a top-level Parquet column. If you put it at the top level, the reward function will raise a `KeyError`.

### Minimal Parquet Example

A valid training row looks like this in Python:

```python
{
    "prompt": [
        {"role": "user", "content": "What is 2 + 3?"}
    ],
    "data_source": "openai/gsm8k",
    "reward_model": {"ground_truth": "5"},
    "extra_info": {}
}
```

## DataArguments Reference

!!! tip "Key Insight"
    All data configuration lives under the `data.*` namespace. Pass parameters via CLI dot-notation, e.g. `data.train_files=/path/to/train.parquet data.max_prompt_length=1024`.

`DataArguments` (`data_args.py`) controls how data is loaded and preprocessed:

| Parameter                   | Type        | Default                               | Description                                                                              |                                                                  |
| --------------------------- | ----------- | ------------------------------------- | ---------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| `train_files`               | `list[str]` | `["~/data/rlhf/gsm8k/train.parquet"]` | Training Parquet file paths (comma-separated string also accepted)                       |                                                                  |
| `val_files`                 | `list[str]` | `["~/data/rlhf/gsm8k/test.parquet"]`  | Validation Parquet file paths                                                            |                                                                  |
| `prompt_key`                | `str`       | `"prompt"`                            | Column name for prompts in Parquet                                                       |                                                                  |
| `reward_fn_key`             | `str`       | `"data_source"`                       | Column name for reward function dispatch                                                 |                                                                  |
| `max_prompt_length`         | `int`       | `512`                                 | Max token length for prompts                                                             |                                                                  |
| `max_response_length`       | `int`       | `512`                                 | Max token length for responses                                                           |                                                                  |
| `train_batch_size`          | `int`       | `1024`                                | Global training batch size (divided across DDP ranks)                                    |                                                                  |
| `val_batch_size`            | `int \      | None`                                 | `None`                                                                                   | Validation batch size; `None` uses the entire validation set     |
| `gen_batch_size`            | `int \      | None`                                 | `None`                                                                                   | Generation batch size for DAPO (typically 3x `train_batch_size`) |
| `return_raw_chat`           | `bool`      | `True`                                | Preserve raw chat messages in `Sample.raw_prompt`                                        |                                                                  |
| `filter_overlong_prompts`   | `bool`      | `False`                               | Filter prompts exceeding `max_prompt_length` before tokenization                         |                                                                  |
| `shuffle`                   | `bool`      | `True`                                | Shuffle training data                                                                    |                                                                  |
| `truncation`                | `str`       | `"error"`                             | Truncation strategy: `"error"`, `"left"`, `"right"`, or `"middle"`                       |                                                                  |
| `streaming`                 | `bool`      | `False`                               | Enable dataset streaming mode                                                            |                                                                  |
| `force_on_the_fly`          | `bool`      | `False`                               | On-the-fly preprocessing (saves memory for large datasets)                               |                                                                  |
| `mix_strategy`              | `str`       | `"concat"`                            | Dataset mixing strategy: `"concat"`, `"interleave_under"`, `"interleave_over"`           |                                                                  |
| `interleave_probs`          | `str \      | None`                                 | `None`                                                                                   | Comma-separated sampling probabilities for interleaved mixing    |
| `auto_repeat`               | `bool`      | `False`                               | Auto-repeat dataset if smaller than batch size                                           |                                                                  |
| `num_loader_workers`        | `int`       | `8`                                   | DataLoader worker count                                                                  |                                                                  |
| `preprocessing_num_workers` | `int \      | None`                                 | `None`                                                                                   | Preprocessing parallelism; `None` uses `cpu_count // 8`          |
| `preprocessing_batch_size`  | `int`       | `1000`                                | Examples per preprocessing group                                                         |                                                                  |
| `cutoff_len`                | `int`       | `2048`                                | Cutoff length of tokenized inputs                                                        |                                                                  |
| `buffer_size`               | `int`       | `16384`                               | Buffer size for streaming random sampling                                                |                                                                  |
| `train_on_prompt`           | `bool`      | `False`                               | Disable loss mask on prompt tokens                                                       |                                                                  |
| `mask_history`              | `bool`      | `False`                               | Mask conversation history; train on last turn only (incompatible with `train_on_prompt`) |                                                                  |
| `tokenized_path`            | `str \      | None`                                 | `None`                                                                                   | Path to cache/load tokenized datasets                            |
| `overwrite_cache`           | `bool`      | `False`                               | Overwrite cached tokenized datasets                                                      |                                                                  |

## Converting from Common Formats

### From HuggingFace Datasets

```python
from datasets import load_dataset

# Load a HuggingFace dataset
ds = load_dataset("openai/gsm8k", "main", split="train")

# Map to siirl-agentic schema
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

### From JSON / JSONL

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

### Multi-Turn Conversation Format

siirl-agentic expects the OpenAI chat format — a list of `{"role": ..., "content": ...}` dicts. For multi-turn conversations, include the full history:

```python
{
    "prompt": [
        {"role": "system", "content": "You are a helpful math tutor."},
        {"role": "user", "content": "What is calculus?"},
        {"role": "assistant", "content": "Calculus is the study of change..."},
        {"role": "user", "content": "Can you give me an example?"},
    ],
    "data_source": "custom",
    "reward_model": {"ground_truth": "any valid calculus example"},
}
```

When `data.return_raw_chat=True` (the default), the original message list is preserved in `Sample.raw_prompt` for use in reward functions and multi-turn rollout.

## Dataset Mixing

!!! tip "Key Insight"
    Use dataset mixing to combine multiple data sources in a single training run. The `mix_strategy` controls how datasets are combined, and `interleave_probs` lets you weight them.

When `data.train_files` contains multiple Parquet paths, the `mix_strategy` parameter controls how they are combined:

| Strategy           | Behavior                                          | When to Use                                   |
| ------------------ | ------------------------------------------------- | --------------------------------------------- |
| `concat`           | Concatenate all datasets end-to-end               | Default; simple and predictable               |
| `interleave_under` | Round-robin sampling, stop at shortest dataset    | Prevent overrepresentation of larger datasets |
| `interleave_over`  | Round-robin sampling, oversample shorter datasets | Ensure all data sources are seen equally      |

**Example: weighted interleaving**

```bash
python -m siirl.async_train \
    data.train_files="math_train.parquet,code_train.parquet" \
    data.mix_strategy=interleave_over \
    data.interleave_probs="0.7,0.3"
```

!!! note "Validation Constraint"
    When using `interleave_probs`, the number of probabilities must match the number of files in both `train_files` and `val_files`. The probabilities apply to both training and validation data loading.

## Data Loading Pipeline

The data loading pipeline in siirl-agentic works as follows:

1. **Parquet files** are loaded by `PartitionedRLHFDataset` (`partitioned_dataset.py`), which reads only the row groups belonging to the current DDP rank
2. **Overlong prompts** are optionally filtered based on `max_prompt_length`
3. **Tokenization** is applied using the model's tokenizer with `apply_chat_template` and left-padding
4. **Position IDs** are computed from the attention mask
5. The `DataLoaderNode` (`data_loader_node.py`) wraps the dataset in a `StatefulDataLoader` with the configured sampler and batch size
6. Each batch is converted to a `TensorDict` and then to a list of `Sample` objects via `Dict2Samples` (`sample.py`)

## Validation and Common Mistakes

!!! tip "Key Insight"
    Most data issues surface as cryptic errors during the first training step. Validate your Parquet schema before launching a distributed run.

**Quick validation script:**

```python
import pyarrow.parquet as pq

pf = pq.ParquetFile("train.parquet")
schema = pf.schema_arrow
print(schema)

# Read first row to verify structure
table = pf.read_row_group(0)
row = table.to_pydict()
print("prompt type:", type(row["prompt"][0]))
print("reward_model keys:", row["reward_model"][0].keys() if "reward_model" in row else "MISSING")
```

| Symptom                                                             | Cause                                                                           | Fix                                                             |
| ------------------------------------------------------------------- | ------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| `KeyError: 'prompt'`                                                | Column name mismatch                                                            | Set `data.prompt_key=your_column_name`                          |
| `KeyError: 'ground_truth'`                                          | `ground_truth` placed as top-level column instead of inside `reward_model` dict | Restructure data: `{"reward_model": {"ground_truth": ...}}`     |
| OOM during preprocessing                                            | Entire dataset loaded into memory                                               | Set `data.force_on_the_fly=True`                                |
| `RuntimeError: Prompt length ... is longer than ...`                | Prompts exceed `max_prompt_length` with default `truncation="error"`            | Increase `data.max_prompt_length` or set `data.truncation=left` |
| `AssertionError: Not enough data for current rank`                  | Too few rows for the number of GPUs                                             | Add more data or reduce GPU count                               |
| `ValueError: mask_history is incompatible with train_on_prompt`     | Both flags enabled simultaneously                                               | Use only one of `data.mask_history` or `data.train_on_prompt`   |
| `ValueError: interleave_probs is only valid for interleaved mixing` | `interleave_probs` set with `mix_strategy=concat`                               | Change to `mix_strategy=interleave_under` or `interleave_over`  |
| Uneven reward distribution across data sources                      | Datasets have very different sizes with `concat`                                | Switch to `interleave_over` with appropriate `interleave_probs` |

## What Success Looks Like

After successful data loading, you should see log messages like:

```
INFO     | DataLoaderNode initialized:
INFO     |   Group rank: 0 / 8
INFO     |   Rollout DDP rank: 0 / 2
INFO     |   Train batches per epoch for this rank: 24
INFO     |   Total training steps (approx): 720
```

If `filter_overlong_prompts=True`, you will also see:

```
INFO     | Filtered prompts from 10000 to 9847 on each rank.
```
