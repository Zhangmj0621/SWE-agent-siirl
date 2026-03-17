# Algorithm Theory

*Mathematical foundations of PPO and GRPO as implemented in siirl-agentic.*

## Policy Gradient Foundation

!!! tip "Key Takeaway"
    Reinforcement learning from human feedback (RLHF) optimises a language model policy $\pi_\theta$ by maximising expected reward. The policy gradient theorem gives the direction of steepest ascent, but naively following it can cause destructive updates. Clipping mechanisms bound how far the policy moves in a single step.

The standard policy gradient objective is:

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t} A_t \log \pi_\theta(a_t \mid s_t) \right]
$$

where $A_t$ is the advantage estimate. In practice, we work with the importance-sampled ratio between the current and old policy:

$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} = \exp\!\bigl(\log \pi_\theta(a_t \mid s_t) - \log \pi_{\theta_{\text{old}}}(a_t \mid s_t)\bigr)
$$

Without clipping, a single batch could shift the policy so far that it collapses. PPO and GRPO address this with different strategies.

## PPO: Proximal Policy Optimization

### Clipped Surrogate Objective

The standard PPO objective clips the ratio $r_t$ to the interval $[1 - \varepsilon, 1 + \varepsilon]$:

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\!\Big( r_t(\theta)\, A_t,\; \text{clip}\bigl(r_t(\theta),\, 1-\varepsilon,\, 1+\varepsilon\bigr)\, A_t \Big) \right]
$$

In siirl-agentic the clipping bounds are independently configurable via `ActorArguments` (`siirl/params/model_args.py:99`):

| Parameter         | Default | Description                                     |
| ----------------- | ------- | ----------------------------------------------- |
| `clip_ratio`      | `0.2`   | Symmetric fallback clipping ratio $\varepsilon$ |
| `clip_ratio_low`  | `0.2`   | Lower bound: $1 - \varepsilon_{\text{low}}$     |
| `clip_ratio_high` | `0.2`   | Upper bound: $1 + \varepsilon_{\text{high}}$    |

### Dual-Clip PPO (siirl-agentic Extension)

!!! tip "Key Takeaway"
    siirl-agentic implements **dual-clip PPO** (Ye et al., 2019), which adds a lower bound $c \cdot A_t$ when the advantage is negative. This prevents the policy from over-correcting on bad trajectories, a common failure mode in standard PPO.

The full dual-clip objective implemented in `compute_policy_loss_vanilla()` (`siirl/algorithm/loss.py:98`) is:

$$
L^{\text{dual}}(\theta) = \mathbb{E}_t \begin{cases}
\min\!\Bigl( \max\bigl(-r_t A_t,\; -\text{clip}(r_t,\, 1{-}\varepsilon_l,\, 1{+}\varepsilon_h)\,A_t\bigr),\; -c\, A_t \Bigr) & \text{if } A_t < 0 \\[4pt]
\max\bigl(-r_t A_t,\; -\text{clip}(r_t,\, 1{-}\varepsilon_l,\, 1{+}\varepsilon_h)\,A_t\bigr) & \text{if } A_t \geq 0
\end{cases}
$$

where $c$ is the dual-clip coefficient. The implementation proceeds in three stages:

```
1. pg_losses1 = -A * r                              # unclipped
2. pg_losses2 = -A * clip(r, 1-ε_low, 1+ε_high)    # standard clip
3. clip_pg_losses1 = max(pg_losses1, pg_losses2)     # standard PPO
4. pg_losses3 = -A * c                               # dual-clip floor
5. clip_pg_losses2 = min(pg_losses3, clip_pg_losses1) # apply floor when A < 0
6. pg_losses = where(A < 0, clip_pg_losses2, clip_pg_losses1)
```

| Parameter       | Default     | Description                                                 |
| --------------- | ----------- | ----------------------------------------------------------- |
| `clip_ratio_c`  | `3.0`       | Dual-clip lower bound coefficient $c$ (must be > 1.0)       |
| `entropy_coeff` | `0.0`       | Entropy bonus coefficient                                   |
| `loss_mode`     | `"vanilla"` | Policy loss function (registered in `POLICY_LOSS_REGISTRY`) |

The ratio is clamped for numerical stability: `negative_approx_kl = clamp(log_prob - old_log_prob, -20, 20)`.

### Generalized Advantage Estimation (GAE)

!!! tip "Key Takeaway"
    GAE smoothly interpolates between low-bias (Monte Carlo) and low-variance (one-step TD) advantage estimates via the $\lambda$ parameter. In siirl-agentic, both $\gamma$ and $\lambda$ default to 1.0, which corresponds to the unbiased Monte Carlo advantage.

The GAE formula computes advantages by accumulating discounted TD residuals backwards through time:

$$
\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{T-t-1} (\gamma \lambda)^l \, \delta_{t+l}
$$

where the TD residual is:

$$
\delta_t = r_t + \gamma\, V(s_{t+1}) - V(s_t)
$$

This is computed in `compute_ppo_advantage_return()` (`siirl/algorithm/advantage.py:99`), which iterates in reverse over the response length. The implementation correctly handles masked positions (e.g., tool-response tokens with `response_mask = 0`) by carrying forward the previous next-value and last-GAE-lambda through unmasked positions.

Returns are computed as $G_t = \hat{A}_t + V(s_t)$, and advantages are whitened (zero-mean, unit-variance) over the masked response tokens via `masked_whiten()`.

| Parameter | Default | Location                                                |
| --------- | ------- | ------------------------------------------------------- |
| `gamma`   | `1.0`   | `AlgorithmArguments` (`siirl/params/model_args.py:248`) |
| `lam`     | `1.0`   | `AlgorithmArguments` (`siirl/params/model_args.py:249`) |

### Value Function Loss

The critic is trained with a clipped value loss to prevent destructive updates to the value function. Implemented in `compute_value_loss()` (`siirl/algorithm/loss.py:73`):

$$
L^{\text{VF}} = \mathbb{E}_t \left[ \max\!\Bigl( (V_\theta(s_t) - G_t)^2,\; (\bar{V}_\theta(s_t) - G_t)^2 \Bigr) \right]
$$

where $\bar{V}_\theta$ is the clipped value prediction:

$$
\bar{V}_\theta(s_t) = \text{clip}\bigl(V_\theta(s_t),\; V_{\text{old}}(s_t) - \varepsilon_v,\; V_{\text{old}}(s_t) + \varepsilon_v\bigr)
$$

| Parameter         | Default | Location                                             |
| ----------------- | ------- | ---------------------------------------------------- |
| `cliprange_value` | `0.5`   | `CriticArguments` (`siirl/params/model_args.py:322`) |

## GRPO: Group Relative Policy Optimization

### Group Sampling

!!! tip "Key Takeaway"
    GRPO eliminates the critic model entirely. Instead, it samples $N$ responses per prompt and uses **intra-group statistics** to compute advantages. This halves GPU memory requirements at the cost of sample efficiency.

For each prompt $x$, sample a group of $N$ responses $\{y_1, y_2, \ldots, y_N\}$ from the current policy. Each response receives a scalar reward $R_i$. The group advantage for response $i$ is:

$$
\hat{A}_i = \frac{R_i - \mu_{\text{group}}}{\sigma_{\text{group}} + \epsilon}
$$

where $\mu_{\text{group}} = \frac{1}{N} \sum_j R_j$ and $\sigma_{\text{group}} = \text{std}(\{R_j\})$, with $\epsilon = 10^{-6}$.

The group size $N$ is controlled by `rollout.n` (default: `1` in `RolloutArguments`). For GRPO, this should typically be set to 4--16.

Implementation: `compute_grpo_outcome_advantage()` (`siirl/algorithm/advantage.py:149`). The function groups samples by their `uid` index (assigned in `preprocess_dataloader()`) and computes per-group statistics.

### Advantage Normalization

| Variant       | `norm_adv_by_std_in_grpo` | Formula                                         | Reference                                             |
| ------------- | ------------------------- | ----------------------------------------------- | ----------------------------------------------------- |
| Standard GRPO | `True` (default)          | $\hat{A}_i = (R_i - \mu) / (\sigma + \epsilon)$ | [Shao et al., 2024](https://arxiv.org/abs/2402.03300) |
| Dr. GRPO      | `False`                   | $\hat{A}_i = R_i - \mu$                         | [Liu et al., 2025](https://arxiv.org/abs/2503.20783)  |

The Dr. GRPO variant removes standard-deviation normalization. This avoids suppressing learning signals in groups where most rewards are similar (low variance), which is common in agentic tasks with binary pass/fail rewards.

The resulting advantage is broadcast to all tokens in the response via `scores.unsqueeze(-1) * response_mask`, giving a **token-level advantage tensor** with the same shape as the policy loss.

| Parameter                 | Default  | Location                                                |
| ------------------------- | -------- | ------------------------------------------------------- |
| `adv_estimator`           | `"grpo"` | `AlgorithmArguments` (`siirl/params/model_args.py:247`) |
| `norm_adv_by_std_in_grpo` | `True`   | `AlgorithmArguments` (`siirl/params/model_args.py:251`) |

## KL Penalty

!!! tip "Key Takeaway"
    The KL penalty prevents the policy from drifting too far from a reference model. siirl-agentic supports four KL formulations and an optional straight-through gradient trick for lower variance.

The KL penalty is added to per-token rewards as $r_t' = r_t - \beta \cdot \text{KL}_t$, where $\beta$ is the KL coefficient and the KL term is computed per token.

All variants are implemented in `kl_penalty()` and `kl_penalty_forward()` (`siirl/algorithm/kl_penalty.py:18`):

| Name            | Aliases                | Formula                                                                                    | Notes                            |
| --------------- | ---------------------- | ------------------------------------------------------------------------------------------ | -------------------------------- |
| Forward KL      | `"kl"`, `"k1"`         | $\log \pi_\theta - \log \pi_{\text{ref}}$                                                  | Standard; may have high variance |
| Absolute        | `"abs"`                | $\lvert \log \pi_\theta - \log \pi_{\text{ref}} \rvert$                                    | Symmetric penalty                |
| MSE             | `"mse"`, `"k2"`        | $\frac{1}{2}(\log \pi_\theta - \log \pi_{\text{ref}})^2$                                   | Smoother near zero               |
| Low-variance KL | `"low_var_kl"`, `"k3"` | $\text{clamp}(e^{d} - d - 1, -10, 10)$ where $d = \log \pi_{\text{ref}} - \log \pi_\theta$ | Recommended; numerically stable  |

**Straight-through variants:** Appending `"+"` to any name (e.g., `"kl+"`, `"k3+"`) applies the straight-through trick: the forward pass uses the chosen KL formula, but the backward pass uses $\frac{1}{2}(\log \pi_\theta - \log \pi_{\text{ref}})^2$. This reduces gradient variance while preserving the forward KL signal. MSE (`"mse"`, `"k2"`) ignores the `"+"` suffix since it is already its own backward pass.

| Parameter      | Default        | Location                                                |
| -------------- | -------------- | ------------------------------------------------------- |
| `use_kl_loss`  | `False`        | `ActorArguments` (`siirl/params/model_args.py:125`)     |
| `kl_loss_coef` | `0.001`        | `ActorArguments` (`siirl/params/model_args.py:126`)     |
| `kl_loss_type` | `"low_var_kl"` | `ActorArguments` (`siirl/params/model_args.py:127`)     |
| `kl_penalty`   | `"kl"`         | `AlgorithmArguments` (`siirl/params/model_args.py:250`) |

## Loss Aggregation Modes

!!! tip "Key Takeaway"
    The loss aggregation mode controls how per-token losses are reduced to a scalar. The default `token-mean` works well for fixed-length batches; for variable-length agentic trajectories, `seq-mean-token-sum` often gives more stable training.

All aggregation modes are implemented in `agg_loss()` (`siirl/algorithm/loss.py:27`):

| Mode                      | Formula                                                                 | When to Use                                                                                              |
| ------------------------- | ----------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `token-mean`              | $\frac{\sum_i L_i \cdot m_i}{\sum_i m_i}$                               | Default. Balanced treatment of all tokens.                                                               |
| `seq-mean-token-sum`      | $\frac{1}{B}\sum_j \sum_i L_{j,i} \cdot m_{j,i}$                        | Variable-length sequences; weights by sequence contribution.                                             |
| `seq-mean-token-mean`     | $\frac{1}{B}\sum_j \frac{\sum_i L_{j,i} \cdot m_{j,i}}{\sum_i m_{j,i}}$ | Equal weight per sequence regardless of length.                                                          |
| `seq-mean-token-sum-norm` | $\frac{\sum_j \sum_i L_{j,i} \cdot m_{j,i}}{S}$                         | Normalised by `loss_scale_factor` $S$ (defaults to sequence length). Stable with very different lengths. |

Where $L_{j,i}$ is the loss at token $i$ of sequence $j$, $m_{j,i}$ is the response mask, and $B$ is the number of valid sequences.

When distributed training with `dp_size > 1`, the `token-mean` and `seq-mean-token-sum` modes support global denominators (`batch_num_tokens` and `global_valid_seqs`) to ensure consistent loss scaling across data-parallel ranks.

| Parameter           | Default        | Location                                            |
| ------------------- | -------------- | --------------------------------------------------- |
| `loss_agg_mode`     | `"token-mean"` | `ActorArguments` (`siirl/params/model_args.py:132`) |
| `loss_scale_factor` | `None`         | `ActorArguments` (`siirl/params/model_args.py:113`) |

## PPO vs GRPO: When to Use Which

| Dimension             | PPO                                        | GRPO                                             |
| --------------------- | ------------------------------------------ | ------------------------------------------------ |
| Requires critic model | Yes (`CriticArguments`)                    | No                                               |
| GPU overhead          | Higher (actor + critic + reference)        | Lower (actor + reference)                        |
| Sample efficiency     | Higher (per-token advantage via GAE)       | Lower (outcome-level advantage)                  |
| Advantage granularity | Token-level                                | Sequence-level (same advantage for all tokens)   |
| Best for              | Complex reward landscapes, process rewards | Simple scalar rewards, pass/fail tasks           |
| Group size dependency | N/A                                        | Needs `rollout.n >= 4` for meaningful statistics |
| `adv_estimator` value | `"ppo"`                                    | `"grpo"`                                         |
