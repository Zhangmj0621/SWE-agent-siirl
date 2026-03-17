# 自定义奖励函数

*编写你自己的奖励函数并接入框架，无需修改框架代码。*

## 内置奖励函数

!!! tip "核心要点"
    siirl-agentic 通过 `default_compute_score`（`reward_score/__init__.py`）内置了 8 个奖励类别，覆盖 28+ 种 `data_source` 模式。在编写自定义函数之前，先检查你的数据集是否已被支持。

`default_compute_score` 函数根据每个样本的 `data_source` 字段进行分发。完整映射如下：

| `data_source` 模式                                                                                                                          | 模块                              | 算法                                       |
| ----------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------- | ---------------------------------------- |
| `openai/gsm8k`                                                                                                                            | `gsm8k`                         | 提取 `#### <数字>`，精确字符串匹配                   |
| `lighteval/MATH`、`DigitalLearningGmbH/MATH-lighteval`、`agentica-org/DeepScaleR-Preview-Dataset`、`AIME2024`、`AIME2025`、`AIME24`、`AIME25`   | `math`                          | 提取 `\boxed{...}`，规范化字符串等价比较              |
| `math_dapo`、`aime*`（前缀匹配）                                                                                                                 | `math_dapo`                     | DAPO 风格数学评分                              |
| `numina_aops_forum`、`numina_synthetic_math`、`numina_amc_aime`、`numina_synthetic_amc`、`numina_cn_k12`、`numina_olympiads`                   | `prime_math`                    | PrimeIntellect 数学评分，带规范化                 |
| `codecontests`、`apps`、`codeforces`、`taco`                                                                                                 | `sandbox_fusion` / `prime_code` | 沙箱代码执行（如提供 URL）或静态代码测试；`continuous=True` |
| `hiyouga/geometry3k`                                                                                                                      | `geo3k`                         | 几何题评分                                    |
| `mm_eureka`                                                                                                                               | `mm_eureka`                     | 多模态数学/eureka 评分                          |
| `searchR1_nq`、`searchR1_triviaqa`、`searchR1_popqa`、`searchR1_hotpotqa`、`searchR1_2wikimultihopqa`、`searchR1_musique`、`searchR1_bamboogle` | `search_r1_like_qa_em`          | 提取 `<answer>...</answer>`，规范化精确匹配        |

!!! note "Math-Verify 集成"
    如需更高的数学评分准确度，可以选择安装 [Math-Verify](https://github.com/huggingface/Math-Verify)（`pip install math-verify`），并修改 `__init__.py` 中的分发逻辑，使用 `math_verify` 模块替代 `math`。

## 编写自定义奖励

### 接口

!!! tip "核心要点"
    自定义奖励函数是一个具有特定签名的普通 Python 函数。siirl-agentic 通过 `load_custom_reward_function`（`custom_reward.py`）使用 `importlib` 动态加载它——无需修改框架代码。

`load_custom_reward_function`（`custom_reward.py:25`）从配置中读取 `CustomRewardArguments` 并动态导入你的函数。你的函数必须符合以下签名：

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
        data_source: 样本的 data_source 字段（例如 "my_dataset"）。
        solution_str: 模型生成的响应文本。
        ground_truth: 来自 sample.reward_model["ground_truth"] 的标准答案。
        extra_info: 来自 sample.extra_info 的可选元数据。
        **kwargs: 来自 CustomRewardArguments.reward_kwargs 的额外关键字参数。

    Returns:
        一个浮点数分数（通常为 0.0 或 1.0），或包含详细评分信息的字典。
    """
    ...
```

### CustomRewardArguments

`CustomRewardArguments`（`training_args.py:23`）控制函数的加载方式：

| 参数                                     | 类型     | 默认值                 | 说明                                   |                     |
| -------------------------------------- | ------ | ------------------- | ------------------------------------ | ------------------- |
| `custom_reward_function.path`          | `str \ | None`               | `None`                               | 包含奖励函数的 Python 文件路径 |
| `custom_reward_function.name`          | `str`  | `"reward_function"` | 文件中的函数名                              |                     |
| `custom_reward_function.reward_kwargs` | `dict` | `{}`                | 通过 `functools.partial` 传递给函数的额外关键字参数 |                     |

### 最简示例

创建文件 `my_reward.py`：

```python
def reward_function(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs,
) -> float:
    """简单精确匹配奖励。"""
    # 从模型输出中提取答案
    if ground_truth.strip().lower() in solution_str.strip().lower():
        return 1.0
    return 0.0
```

启动训练：

```bash
python -m siirl.async_train \
    actor_ref.model.path=/path/to/model \
    data.train_files=/path/to/train.parquet \
    custom_reward_function.path=my_reward.py \
    custom_reward_function.name=reward_function
```

### 参数化奖励示例

可以通过 `reward_kwargs` 传递额外的关键字参数。它们在加载时通过 `functools.partial` 应用：

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
    """可配置的字符串匹配奖励，支持部分得分。"""
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

### 多轮奖励

对于多轮 agentic 任务，工具环境通过 `EnvResponse`（`environment/base.py:67`）返回奖励：

```python
class EnvResponse(BaseModel):
    text: str | None = None         # 工具输出文本
    image: list[Any] | None = None  # 视觉观测
    video: list[Any] | None = None  # 视频观测
    rewards: float | None = None    # 每步奖励信号
    complete: bool = False          # 回合终止标志
    metrics: dict | None = None     # 诊断指标
```

在多轮场景中，`EnvResponse.rewards` 字段在每次工具交互步骤提供中间奖励。最终奖励通常是将这些步级奖励与来自你的奖励函数的结果级奖励组合而成。

多轮奖励函数可以使用 `extra_info` 访问回合元数据：

```python
def reward_function(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs,
) -> float:
    """多轮奖励，包含格式和正确性两个组成部分。"""
    score = 0.0

    # 正确性：检查最终答案是否匹配
    if ground_truth.strip() in solution_str:
        score += 0.8

    # 格式：奖励正确的工具调用格式
    if "<tool_call>" in solution_str and "</tool_call>" in solution_str:
        score += 0.2

    return score
```

## 测试你的奖励函数

!!! tip "核心要点"
    在启动分布式训练之前，务必先单独测试你的奖励函数。一个有问题的奖励函数会浪费大量 GPU 时间。

在训练前进行本地测试：

```python
from my_reward import reward_function

# 测试用例
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

# 使用 extra_info 测试
result = reward_function(
    data_source="my_dataset",
    solution_str="The answer is 42.",
    ground_truth="42",
    extra_info={"difficulty": "easy"},
)
print(f"分数: {result}")
```

也可以针对内置分发进行测试，以验证你的 `data_source` 路由：

```python
from siirl.utils.reward_score import default_compute_score

# 这应该使用 gsm8k 奖励
score = default_compute_score(
    data_source="openai/gsm8k",
    solution_str="Let me calculate. 2 + 3 = 5 #### 5",
    ground_truth="5",
)
print(f"GSM8K 分数: {score}")  # 预期: 1.0
```

## 常见错误

| 症状                                                                            | 原因                              | 修复方法                                                   |
| ----------------------------------------------------------------------------- | ------------------------------- | ------------------------------------------------------ |
| `NotImplementedError: Reward function is not implemented for data_source=...` | `data_source` 值不在内置分发中且未配置自定义奖励 | 设置 `custom_reward_function.path` 或使用已识别的 `data_source` |
| `FileNotFoundError: Custom reward function file not found`                    | 奖励文件路径错误                        | 使用绝对路径或确保文件在工作目录中                                      |
| `AttributeError: Function 'reward_function' not found`                        | 函数名不匹配                          | 检查 `custom_reward_function.name` 是否与实际函数名一致            |
| 奖励总是返回 0                                                                      | 提取逻辑与模型输出格式不匹配                  | 打印 `solution_str` 查看实际的模型输出格式                          |
| `TypeError: reward_function() got an unexpected keyword argument`             | 函数签名中缺少 `**kwargs`              | 添加 `**kwargs` 以接受来自 `reward_kwargs` 的额外参数              |
| 训练发散，出现 NaN loss                                                              | 奖励函数返回非有限值                      | 确保你的函数始终返回有限的 `float`                                  |
| `RuntimeError: Failed to execute module`                                      | 奖励文件中的导入错误                      | 先用 `python -c "import my_reward"` 测试你的文件               |
