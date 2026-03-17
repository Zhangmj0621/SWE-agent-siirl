# 算法理论

*siirl-agentic 中 PPO 和 GRPO 的数学基础。*

## 策略梯度基础

!!! tip "核心要点"
    基于人类反馈的强化学习（RLHF）通过最大化期望奖励来优化语言模型策略 $\pi_\theta$。策略梯度定理给出了最速上升方向，但朴素地沿此方向更新可能导致灾难性更新。裁剪机制限制了策略在单步中的移动幅度。

标准策略梯度目标函数为：

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t} A_t \log \pi_\theta(a_t \mid s_t) \right]
$$

其中 $A_t$ 是优势估计。实践中，我们使用当前策略与旧策略之间的重要性采样比率：

$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} = \exp\!\bigl(\log \pi_\theta(a_t \mid s_t) - \log \pi_{\theta_{\text{old}}}(a_t \mid s_t)\bigr)
$$

若不加裁剪，一个批次就可能使策略偏移过大而崩溃。PPO 和 GRPO 分别采用不同策略来解决这一问题。

## PPO：近端策略优化

### 裁剪替代目标

标准 PPO 目标将比率 $r_t$ 裁剪到区间 $[1 - \varepsilon, 1 + \varepsilon]$：

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\!\Big( r_t(\theta)\, A_t,\; \text{clip}\bigl(r_t(\theta),\, 1-\varepsilon,\, 1+\varepsilon\bigr)\, A_t \Big) \right]
$$

在 siirl-agentic 中，裁剪上下界可通过 `ActorArguments`（`siirl/params/model_args.py:99`）独立配置：

| 参数                | 默认值   | 说明                                 |
| ----------------- | ----- | ---------------------------------- |
| `clip_ratio`      | `0.2` | 对称裁剪比率 $\varepsilon$（回退值）          |
| `clip_ratio_low`  | `0.2` | 下界：$1 - \varepsilon_{\text{low}}$  |
| `clip_ratio_high` | `0.2` | 上界：$1 + \varepsilon_{\text{high}}$ |

### 双重裁剪 PPO（siirl-agentic 扩展）

!!! tip "核心要点"
    siirl-agentic 实现了**双重裁剪 PPO**（Ye et al., 2019），在优势为负时增加了一个下界 $c \cdot A_t$。这防止了策略在差的轨迹上过度修正——标准 PPO 中常见的失败模式。

`compute_policy_loss_vanilla()`（`siirl/algorithm/loss.py:98`）中实现的完整双重裁剪目标为：

$$
L^{\text{dual}}(\theta) = \mathbb{E}_t \begin{cases}
\min\!\Bigl( \max\bigl(-r_t A_t,\; -\text{clip}(r_t,\, 1{-}\varepsilon_l,\, 1{+}\varepsilon_h)\,A_t\bigr),\; -c\, A_t \Bigr) & \text{if } A_t < 0 \\[4pt]
\max\bigl(-r_t A_t,\; -\text{clip}(r_t,\, 1{-}\varepsilon_l,\, 1{+}\varepsilon_h)\,A_t\bigr) & \text{if } A_t \geq 0
\end{cases}
$$

其中 $c$ 是双重裁剪系数。实现分为三个阶段：

```
1. pg_losses1 = -A * r                              # 未裁剪
2. pg_losses2 = -A * clip(r, 1-ε_low, 1+ε_high)    # 标准裁剪
3. clip_pg_losses1 = max(pg_losses1, pg_losses2)     # 标准 PPO
4. pg_losses3 = -A * c                               # 双重裁剪下限
5. clip_pg_losses2 = min(pg_losses3, clip_pg_losses1) # A < 0 时应用下限
6. pg_losses = where(A < 0, clip_pg_losses2, clip_pg_losses1)
```

| 参数              | 默认值         | 说明                                   |
| --------------- | ----------- | ------------------------------------ |
| `clip_ratio_c`  | `3.0`       | 双重裁剪下界系数 $c$（必须 > 1.0）               |
| `entropy_coeff` | `0.0`       | 熵奖励系数                                |
| `loss_mode`     | `"vanilla"` | 策略损失函数（注册在 `POLICY_LOSS_REGISTRY` 中） |

为数值稳定性，比率被裁剪：`negative_approx_kl = clamp(log_prob - old_log_prob, -20, 20)`。

### 广义优势估计（GAE）

!!! tip "核心要点"
    GAE 通过 $\lambda$ 参数在低偏差（蒙特卡洛）和低方差（单步 TD）优势估计之间平滑插值。在 siirl-agentic 中，$\gamma$ 和 $\lambda$ 默认均为 1.0，对应无偏蒙特卡洛优势。

GAE 公式通过对时间维度反向累加折扣 TD 残差来计算优势：

$$
\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{T-t-1} (\gamma \lambda)^l \, \delta_{t+l}
$$

其中 TD 残差为：

$$
\delta_t = r_t + \gamma\, V(s_{t+1}) - V(s_t)
$$

此计算在 `compute_ppo_advantage_return()`（`siirl/algorithm/advantage.py:99`）中实现，对响应长度进行反向迭代。实现正确处理了掩码位置（如 `response_mask = 0` 的工具响应 token），通过在未掩码位置延续前一个 next-value 和 last-GAE-lambda。

回报计算为 $G_t = \hat{A}_t + V(s_t)$，优势在掩码响应 token 上通过 `masked_whiten()` 进行白化（零均值、单位方差）。

| 参数      | 默认值   | 位置                                                     |
| ------- | ----- | ------------------------------------------------------ |
| `gamma` | `1.0` | `AlgorithmArguments`（`siirl/params/model_args.py:248`） |
| `lam`   | `1.0` | `AlgorithmArguments`（`siirl/params/model_args.py:249`） |

### 值函数损失

Critic 使用裁剪值损失进行训练，防止值函数的破坏性更新。在 `compute_value_loss()`（`siirl/algorithm/loss.py:73`）中实现：

$$
L^{\text{VF}} = \mathbb{E}_t \left[ \max\!\Bigl( (V_\theta(s_t) - G_t)^2,\; (\bar{V}_\theta(s_t) - G_t)^2 \Bigr) \right]
$$

其中 $\bar{V}_\theta$ 是裁剪后的值预测：

$$
\bar{V}_\theta(s_t) = \text{clip}\bigl(V_\theta(s_t),\; V_{\text{old}}(s_t) - \varepsilon_v,\; V_{\text{old}}(s_t) + \varepsilon_v\bigr)
$$

| 参数                | 默认值   | 位置                                                  |
| ----------------- | ----- | --------------------------------------------------- |
| `cliprange_value` | `0.5` | `CriticArguments`（`siirl/params/model_args.py:322`） |

## GRPO：群组相对策略优化

### 群组采样

!!! tip "核心要点"
    GRPO 完全消除了 Critic 模型。它为每个 prompt 采样 $N$ 个响应，使用**组内统计量**计算优势。这将 GPU 显存需求减半，但牺牲了样本效率。

对于每个 prompt $x$，从当前策略采样一组 $N$ 个响应 $\{y_1, y_2, \ldots, y_N\}$。每个响应获得标量奖励 $R_i$。响应 $i$ 的组内优势为：

$$
\hat{A}_i = \frac{R_i - \mu_{\text{group}}}{\sigma_{\text{group}} + \epsilon}
$$

其中 $\mu_{\text{group}} = \frac{1}{N} \sum_j R_j$，$\sigma_{\text{group}} = \text{std}(\{R_j\})$，$\epsilon = 10^{-6}$。

组大小 $N$ 由 `rollout.n`（`RolloutArguments` 中默认为 `1`）控制。使用 GRPO 时，通常应设为 4--16。

实现：`compute_grpo_outcome_advantage()`（`siirl/algorithm/advantage.py:149`）。函数通过 `uid` 索引（在 `preprocess_dataloader()` 中赋值）对样本分组，并计算每组统计量。

### 优势归一化

| 变体       | `norm_adv_by_std_in_grpo` | 公式                                              | 参考文献                                                  |
| -------- | ------------------------- | ----------------------------------------------- | ----------------------------------------------------- |
| 标准 GRPO  | `True`（默认）                | $\hat{A}_i = (R_i - \mu) / (\sigma + \epsilon)$ | [Shao et al., 2024](https://arxiv.org/abs/2402.03300) |
| Dr. GRPO | `False`                   | $\hat{A}_i = R_i - \mu$                         | [Liu et al., 2025](https://arxiv.org/abs/2503.20783)  |

Dr. GRPO 变体移除了标准差归一化。这避免了在大多数奖励相近（低方差）的组中抑制学习信号——这在使用二值通过/失败奖励的 Agentic 任务中很常见。

计算得到的优势通过 `scores.unsqueeze(-1) * response_mask` 广播到响应中的所有 token，产生与策略损失形状相同的 **token 级优势张量**。

| 参数                        | 默认值      | 位置                                                     |
| ------------------------- | -------- | ------------------------------------------------------ |
| `adv_estimator`           | `"grpo"` | `AlgorithmArguments`（`siirl/params/model_args.py:247`） |
| `norm_adv_by_std_in_grpo` | `True`   | `AlgorithmArguments`（`siirl/params/model_args.py:251`） |

## KL 惩罚

!!! tip "核心要点"
    KL 惩罚防止策略偏离参考模型过远。siirl-agentic 支持四种 KL 公式和一个可选的直通梯度技巧以降低方差。

KL 惩罚以 $r_t' = r_t - \beta \cdot \text{KL}_t$ 的形式添加到每个 token 的奖励中，其中 $\beta$ 是 KL 系数，KL 项按 token 计算。

所有变体在 `kl_penalty()` 和 `kl_penalty_forward()`（`siirl/algorithm/kl_penalty.py:18`）中实现：

| 名称     | 别名                     | 公式                                                                                      | 说明        |
| ------ | ---------------------- | --------------------------------------------------------------------------------------- | --------- |
| 前向 KL  | `"kl"`, `"k1"`         | $\log \pi_\theta - \log \pi_{\text{ref}}$                                               | 标准；可能方差较高 |
| 绝对值    | `"abs"`                | $\lvert \log \pi_\theta - \log \pi_{\text{ref}} \rvert$                                 | 对称惩罚      |
| 均方误差   | `"mse"`, `"k2"`        | $\frac{1}{2}(\log \pi_\theta - \log \pi_{\text{ref}})^2$                                | 零点附近更平滑   |
| 低方差 KL | `"low_var_kl"`, `"k3"` | $\text{clamp}(e^{d} - d - 1, -10, 10)$，其中 $d = \log \pi_{\text{ref}} - \log \pi_\theta$ | 推荐使用；数值稳定 |

**直通变体：** 在任何名称后追加 `"+"`（如 `"kl+"`、`"k3+"`）将应用直通技巧：前向传播使用所选 KL 公式，但反向传播使用 $\frac{1}{2}(\log \pi_\theta - \log \pi_{\text{ref}})^2$。这在保留前向 KL 信号的同时降低了梯度方差。MSE（`"mse"`、`"k2"`）忽略 `"+"` 后缀，因为它本身就是自己的反向传播。

| 参数             | 默认值            | 位置                                                     |
| -------------- | -------------- | ------------------------------------------------------ |
| `use_kl_loss`  | `False`        | `ActorArguments`（`siirl/params/model_args.py:125`）     |
| `kl_loss_coef` | `0.001`        | `ActorArguments`（`siirl/params/model_args.py:126`）     |
| `kl_loss_type` | `"low_var_kl"` | `ActorArguments`（`siirl/params/model_args.py:127`）     |
| `kl_penalty`   | `"kl"`         | `AlgorithmArguments`（`siirl/params/model_args.py:250`） |

## 损失聚合模式

!!! tip "核心要点"
    损失聚合模式控制 per-token 损失如何归约为标量。默认的 `token-mean` 适用于定长批次；对于变长 Agentic 轨迹，`seq-mean-token-sum` 通常能提供更稳定的训练。

所有聚合模式在 `agg_loss()`（`siirl/algorithm/loss.py:27`）中实现：

| 模式                        | 公式                                                                      | 适用场景                                              |
| ------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------- |
| `token-mean`              | $\frac{\sum_i L_i \cdot m_i}{\sum_i m_i}$                               | 默认。均衡对待所有 token。                                  |
| `seq-mean-token-sum`      | $\frac{1}{B}\sum_j \sum_i L_{j,i} \cdot m_{j,i}$                        | 变长序列；按序列贡献加权。                                     |
| `seq-mean-token-mean`     | $\frac{1}{B}\sum_j \frac{\sum_i L_{j,i} \cdot m_{j,i}}{\sum_i m_{j,i}}$ | 不论长度，每个序列等权。                                      |
| `seq-mean-token-sum-norm` | $\frac{\sum_j \sum_i L_{j,i} \cdot m_{j,i}}{S}$                         | 由 `loss_scale_factor` $S$ 归一化（默认为序列长度）。长度差异大时更稳定。 |

其中 $L_{j,i}$ 是序列 $j$ 中 token $i$ 的损失，$m_{j,i}$ 是响应掩码，$B$ 是有效序列数。

在 `dp_size > 1` 的分布式训练中，`token-mean` 和 `seq-mean-token-sum` 模式支持全局分母（`batch_num_tokens` 和 `global_valid_seqs`），确保数据并行各 rank 间损失缩放一致。

| 参数                  | 默认值            | 位置                                                 |
| ------------------- | -------------- | -------------------------------------------------- |
| `loss_agg_mode`     | `"token-mean"` | `ActorArguments`（`siirl/params/model_args.py:132`） |
| `loss_scale_factor` | `None`         | `ActorArguments`（`siirl/params/model_args.py:113`） |

## PPO 与 GRPO：如何选择

| 维度                | PPO                            | GRPO                           |
| ----------------- | ------------------------------ | ------------------------------ |
| 是否需要 Critic 模型    | 是（`CriticArguments`）           | 否                              |
| GPU 开销            | 较高（actor + critic + reference） | 较低（actor + reference）          |
| 样本效率              | 较高（GAE 逐 token 优势）             | 较低（结果级优势）                      |
| 优势粒度              | Token 级                        | 序列级（所有 token 共享相同优势）           |
| 适用场景              | 复杂奖励景观、过程奖励                    | 简单标量奖励、通过/失败任务                 |
| 组大小依赖             | 无                              | 需要 `rollout.n >= 4` 以获得有意义的统计量 |
| `adv_estimator` 值 | `"ppo"`                        | `"grpo"`                       |
