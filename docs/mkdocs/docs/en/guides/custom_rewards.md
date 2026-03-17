# Custom Reward Functions

*Write your own reward function and plug it in without modifying framework code.*

## Built-in Reward Functions

!!! tip "Key Insight"
    siirl-agentic includes 8 reward categories covering 28+ `data_source` patterns via `default_compute_score` (`reward_score/__init__.py`). Check if your dataset is already supported before writing a custom function.

The `default_compute_score` function dispatches on the `data_source` field of each sample. The complete mapping:

| `data_source` Pattern                                                                                                                           | Module                          | Algorithm                                                                          |
| ----------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------- | ---------------------------------------------------------------------------------- |
| `openai/gsm8k`                                                                                                                                  | `gsm8k`                         | Extract `#### <number>`, exact string match                                        |
| `lighteval/MATH`, `DigitalLearningGmbH/MATH-lighteval`, `agentica-org/DeepScaleR-Preview-Dataset`, `AIME2024`, `AIME2025`, `AIME24`, `AIME25`   | `math`                          | Extract `\boxed{...}`, normalized string equivalence                               |
| `math_dapo`, `aime*` (startswith)                                                                                                               | `math_dapo`                     | DAPO-style math scoring                                                            |
| `numina_aops_forum`, `numina_synthetic_math`, `numina_amc_aime`, `numina_synthetic_amc`, `numina_cn_k12`, `numina_olympiads`                    | `prime_math`                    | PrimeIntellect math grading with normalization                                     |
| `codecontests`, `apps`, `codeforces`, `taco`                                                                                                    | `sandbox_fusion` / `prime_code` | Sandbox code execution (if URL provided) or static code testing; `continuous=True` |
| `hiyouga/geometry3k`                                                                                                                            | `geo3k`                         | Geometry problem scoring                                                           |
| `mm_eureka`                                                                                                                                     | `mm_eureka`                     | Multi-modal math/eureka scoring                                                    |
| `searchR1_nq`, `searchR1_triviaqa`, `searchR1_popqa`, `searchR1_hotpotqa`, `searchR1_2wikimultihopqa`, `searchR1_musique`, `searchR1_bamboogle` | `search_r1_like_qa_em`          | Extract `<answer>...</answer>`, normalized exact match                             |

!!! note "Math-Verify Integration"
    For enhanced math accuracy, you can optionally install [Math-Verify](https://github.com/huggingface/Math-Verify) (`pip install math-verify`) and modify the dispatch in `__init__.py` to use the `math_verify` module instead of `math`.

## Writing a Custom Reward

### Interface

!!! tip "Key Insight"
    A custom reward function is a plain Python function with a specific signature. siirl-agentic loads it dynamically via `load_custom_reward_function` (`custom_reward.py`) using `importlib` — no framework code changes needed.

The `load_custom_reward_function` (`custom_reward.py:25`) reads the `CustomRewardArguments` from the config and dynamically imports your function. Your function must accept this signature:

```python
def reward_function(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs,
) -> float | dict:
    """
    Args:
        data_source: The data_source field from the sample (e.g. "my_dataset").
        solution_str: The model's generated response text.
        ground_truth: The ground truth from sample.reward_model["ground_truth"].
        extra_info: Optional metadata from sample.extra_info.
        **kwargs: Additional keyword arguments from CustomRewardArguments.reward_kwargs.

    Returns:
        A float score (typically 0.0 or 1.0), or a dict with detailed scoring info.
    """
    ...
```

### CustomRewardArguments

`CustomRewardArguments` (`training_args.py:23`) controls how your function is loaded:

| Parameter                              | Type   | Default             | Description                                                             |                                                         |
| -------------------------------------- | ------ | ------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------- |
| `custom_reward_function.path`          | `str \ | None`               | `None`                                                                  | Path to the Python file containing your reward function |
| `custom_reward_function.name`          | `str`  | `"reward_function"` | Function name within the file                                           |                                                         |
| `custom_reward_function.reward_kwargs` | `dict` | `{}`                | Extra keyword arguments passed to your function via `functools.partial` |                                                         |

### Minimal Example

Create a file `my_reward.py`:

```python
def reward_function(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs,
) -> float:
    """Simple exact-match reward."""
    # Extract the answer from the model output
    if ground_truth.strip().lower() in solution_str.strip().lower():
        return 1.0
    return 0.0
```

Launch training with:

```bash
python -m siirl.async_train \
    actor_ref.model.path=/path/to/model \
    data.train_files=/path/to/train.parquet \
    custom_reward_function.path=my_reward.py \
    custom_reward_function.name=reward_function
```

### Parameterized Reward Example

You can pass extra keyword arguments via `reward_kwargs`. They are applied through `functools.partial` at load time:

```python
# my_reward.py
def reward_function(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    partial_score: float = 0.5,
    case_sensitive: bool = False,
    **kwargs,
) -> float:
    """Configurable string-match reward with partial credit."""
    answer = solution_str.strip()
    target = ground_truth.strip()

    if not case_sensitive:
        answer = answer.lower()
        target = target.lower()

    if target == answer:
        return 1.0
    elif target in answer:
        return partial_score
    return 0.0
```

```bash
python -m siirl.async_train \
    custom_reward_function.path=my_reward.py \
    custom_reward_function.name=reward_function \
    custom_reward_function.reward_kwargs.partial_score=0.3 \
    custom_reward_function.reward_kwargs.case_sensitive=true
```

### Multi-Turn Reward

For multi-turn agentic tasks, the tool environment returns rewards via `EnvResponse` (`environment/base.py:67`):

```python
class EnvResponse(BaseModel):
    text: str | None = None         # Tool output text
    image: list[Any] | None = None  # Visual observations
    video: list[Any] | None = None  # Video observations
    rewards: float | None = None    # Per-step reward signal
    complete: bool = False          # Episode termination flag
    metrics: dict | None = None     # Diagnostic metrics
```

In multi-turn scenarios, the `EnvResponse.rewards` field provides intermediate rewards at each tool interaction step. The final reward is typically computed by combining these step-level rewards with an outcome-level reward from your reward function.

A multi-turn reward function can use `extra_info` to access episode metadata:

```python
def reward_function(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs,
) -> float:
    """Multi-turn reward with format and correctness components."""
    score = 0.0

    # Correctness: check if final answer matches
    if ground_truth.strip() in solution_str:
        score += 0.8

    # Format: reward proper tool-call formatting
    if "<tool_call>" in solution_str and "</tool_call>" in solution_str:
        score += 0.2

    return score
```

## Testing Your Reward

!!! tip "Key Insight"
    Always test your reward function standalone before launching a distributed training run. A broken reward function will waste GPU hours.

Test your function locally before training:

```python
from my_reward import reward_function

# Test cases
assert reward_function(
    data_source="my_dataset",
    solution_str="The answer is 42.",
    ground_truth="42",
) == 1.0

assert reward_function(
    data_source="my_dataset",
    solution_str="I don't know.",
    ground_truth="42",
) == 0.0

# Test with extra_info
result = reward_function(
    data_source="my_dataset",
    solution_str="The answer is 42.",
    ground_truth="42",
    extra_info={"difficulty": "easy"},
)
print(f"Score: {result}")
```

You can also test against the built-in dispatch to verify your `data_source` routing:

```python
from siirl.utils.reward_score import default_compute_score

# This should use the gsm8k reward
score = default_compute_score(
    data_source="openai/gsm8k",
    solution_str="Let me calculate. 2 + 3 = 5 #### 5",
    ground_truth="5",
)
print(f"GSM8K score: {score}")  # Expected: 1.0
```

## Common Mistakes

| Symptom                                                                       | Cause                                                                        | Fix                                                                  |
| ----------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `NotImplementedError: Reward function is not implemented for data_source=...` | `data_source` value not in built-in dispatch and no custom reward configured | Set `custom_reward_function.path` or use a recognized `data_source`  |
| `FileNotFoundError: Custom reward function file not found`                    | Wrong path to reward file                                                    | Use an absolute path or ensure the file is in the working directory  |
| `AttributeError: Function 'reward_function' not found`                        | Function name mismatch                                                       | Check `custom_reward_function.name` matches the actual function name |
| Reward always returns 0                                                       | Extraction logic mismatch with model output format                           | Print `solution_str` to inspect the actual model output format       |
| `TypeError: reward_function() got an unexpected keyword argument`             | Missing `**kwargs` in function signature                                     | Add `**kwargs` to accept extra arguments from `reward_kwargs`        |
| Training diverges with NaN loss                                               | Reward function returns non-finite values                                    | Ensure your function always returns a finite `float`                 |
| `RuntimeError: Failed to execute module`                                      | Import error in reward file                                                  | Test your file with `python -c "import my_reward"` first             |
