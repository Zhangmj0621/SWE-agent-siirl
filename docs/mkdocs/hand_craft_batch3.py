#!/usr/bin/env python3
"""Hand-crafted SVG rendering functions for Batch 3 diagrams:
  - user_guide/grpo_training (2 diagrams)
  - user_guide/ppo_training (2 diagrams)
  - user_guide/agentic_multiturn (2 diagrams)
  - user_guide/aio_tool_infrastructure (2 diagrams)
  - user_guide/configuration_system (2 diagrams)
  - user_guide/deployment_modes (4 diagrams)
  - user_guide/checkpoint_resume (2 diagrams)
  - user_guide/tool_env_and_swe (2 diagrams)
  - user_guide/metrics_and_evaluation (1 diagram)
  - user_guide/validate_reuse_and_eval_scaling (2 diagrams)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from svg_renderer import (
    BOX_GAP,
    BOX_HEIGHT,
    BOX_TEXT_SIZE,
    LABEL_SIZE,
    PALETTES,
    REGION_PADDING,
    REGION_TITLE_HEIGHT,
    REGION_TITLE_SIZE,
    SvgCanvas,
    estimate_text_width,
)

# ===================================================================
# user_guide/grpo_training.md — Diagram 1
# GRPO Group Sampling and Training
# ===================================================================


def render_grpo_1() -> str:
    c = SvgCanvas(
        width=900,
        height=580,
        title="GRPO Training Pipeline",
        caption="Figure 1: Group Relative Policy Optimization workflow",
    )

    # Group Sampling Phase (top)
    r_sample = c.region(x=30, y=55, w=840, h=200, title="Group Sampling Phase", palette="rollout")
    b_prompt = c.box(x=50, y=100, w=120, h=BOX_HEIGHT, label="Prompt x_i", bold=True, palette="data")
    b_sglang = c.box(x=200, y=100, w=140, h=BOX_HEIGHT, label="SGLang Engine", palette="rollout")
    # Group samples
    b_y1 = c.box(x=380, y=80, w=100, h=26, label="y_1 : r=0.8", palette="rollout")
    b_y2 = c.box(x=500, y=80, w=100, h=26, label="y_2 : r=0.2", palette="rollout")
    b_y3 = c.box(x=620, y=80, w=100, h=26, label="y_3 : r=1.0", palette="rollout")
    b_y4 = c.box(x=740, y=80, w=100, h=26, label="y_4 : r=0.4", palette="rollout")
    b_y5 = c.box(x=380, y=120, w=100, h=26, label="y_5 : r=0.7", palette="rollout")
    b_y6 = c.box(x=500, y=120, w=100, h=26, label="y_6 : r=0.3", palette="rollout")
    b_y7 = c.box(x=620, y=120, w=100, h=26, label="y_7 : r=0.9", palette="rollout")
    b_y8 = c.box(x=740, y=120, w=100, h=26, label="y_8 : r=0.5", palette="rollout")

    b_group = c.box(x=50, y=180, w=180, h=BOX_HEIGHT, label="Group of 8 samples", bold=True, palette="data")

    # Advantage Computation (middle)
    r_adv = c.region(x=30, y=280, w=400, h=120, title="Group Advantage Computation", palette="training")
    b_stats = c.box(x=50, y=320, w=150, h=BOX_HEIGHT, label="Group Statistics", palette="training")
    b_adv = c.box(x=230, y=320, w=180, h=BOX_HEIGHT, label="Advantage_i = r_i - μ", palette="training")

    # Policy Update (bottom)
    r_update = c.region(x=30, y=420, w=400, h=100, title="Policy Update", palette="training")
    b_ref = c.box(x=50, y=460, w=120, h=BOX_HEIGHT, label="Ref Forward", palette="training")
    b_loss = c.box(x=200, y=460, w=150, h=BOX_HEIGHT, label="Clipped PPO Loss", palette="training")

    # Arrows
    c.arrow(b_prompt, b_sglang, label="", src_side="right", dst_side="left")
    c.arrow(b_sglang, b_y1, label="G samples", src_side="right", dst_side="left")
    c.arrow(b_group, b_stats, label="", src_side="bottom", dst_side="top")
    c.arrow(b_stats, b_adv, label="", src_side="right", dst_side="left")
    c.arrow(b_adv, b_ref, label="", src_side="bottom", dst_side="top")
    c.arrow(b_ref, b_loss, label="", src_side="right", dst_side="left")

    return c.render()


# ===================================================================
# user_guide/grpo_training.md — Diagram 2
# GRPO Loss Formula
# ===================================================================


def render_grpo_2() -> str:
    c = SvgCanvas(
        width=700,
        height=280,
        title="GRPO Advantage Calculation",
        caption="Figure 2: How group-relative advantages are computed",
    )

    # Linear flow
    b_rewards = c.box(x=40, y=100, w=130, h=BOX_HEIGHT, label="Group Rewards", bold=True, palette="data")
    b_mean = c.box(x=210, y=100, w=100, h=BOX_HEIGHT, label="Mean (μ)", palette="training")
    b_std = c.box(x=340, y=100, w=100, h=BOX_HEIGHT, label="Std (σ)", palette="training")
    b_adv = c.box(x=480, y=100, w=170, h=BOX_HEIGHT, label="A_i = (r_i - μ) / σ", bold=True, palette="highlight")

    c.arrow(b_rewards, b_mean, label="", src_side="right", dst_side="left")
    c.arrow(b_rewards, b_std, label="", src_side="right", dst_side="left")
    c.arrow(b_mean, b_adv, label="", src_side="right", dst_side="left")
    c.arrow(b_std, b_adv, label="", src_side="right", dst_side="left")

    return c.render()


# ===================================================================
# user_guide/ppo_training.md — Diagram 1
# PPO Training Pipeline
# ===================================================================


def render_ppo_1() -> str:
    c = SvgCanvas(
        width=500,
        height=700,
        title="PPO Training Pipeline",
        caption="Figure 1: Proximal Policy Optimization workflow",
    )

    # Rollout Phase (top)
    r_rollout = c.region(x=30, y=55, w=440, h=140, title="Rollout Phase", palette="rollout")
    b_rm = c.box(x=50, y=100, w=140, h=BOX_HEIGHT, label="RolloutManager", palette="rollout")
    b_samples = c.box(x=220, y=100, w=160, h=BOX_HEIGHT, label="Completed Samples", palette="rollout")

    # Forward Pass (middle)
    r_forward = c.region(x=30, y=220, w=440, h=160, title="Forward Pass (3 Models)", palette="infra")
    b_actor = c.box(x=50, y=265, w=120, h=BOX_HEIGHT, label="Actor Forward", palette="training")
    b_ref = c.box(x=190, y=265, w=110, h=BOX_HEIGHT, label="Ref Forward", palette="training")
    b_critic = c.box(x=320, y=265, w=120, h=BOX_HEIGHT, label="Critic Forward", palette="training")
    b_gae = c.box(x=150, y=320, w=150, h=BOX_HEIGHT, label="GAE Computation", bold=True, palette="training")

    # Policy Update (bottom)
    r_update = c.region(x=30, y=400, w=440, h=160, title="Policy & Value Update", palette="training")
    b_actor_loss = c.box(x=50, y=445, w=110, h=BOX_HEIGHT, label="Actor Loss", palette="training")
    b_critic_loss = c.box(x=180, y=445, w=110, h=BOX_HEIGHT, label="Critic Loss", palette="training")
    b_backward = c.box(x=310, y=445, w=130, h=BOX_HEIGHT, label="Backward", palette="training")
    b_sync = c.box(x=150, y=510, w=150, h=BOX_HEIGHT, label="Param Sync", bold=True, palette="highlight")

    # Arrows
    c.arrow(b_rm, b_samples, label="", src_side="right", dst_side="left")
    c.arrow(b_samples, b_actor, label="", src_side="bottom", dst_side="top")
    c.arrow(b_actor, b_gae, label="", src_side="bottom", dst_side="top")
    c.arrow(b_ref, b_gae, label="", src_side="bottom", dst_side="top")
    c.arrow(b_critic, b_gae, label="", src_side="bottom", dst_side="top")
    c.arrow(b_gae, b_actor_loss, label="", src_side="bottom", dst_side="top")
    c.arrow(b_gae, b_critic_loss, label="", src_side="bottom", dst_side="top")
    c.arrow(b_actor_loss, b_backward, label="", src_side="right", dst_side="left")
    c.arrow(b_critic_loss, b_backward, label="", src_side="right", dst_side="left")
    c.arrow(b_backward, b_sync, label="", src_side="bottom", dst_side="top")

    return c.render()


# ===================================================================
# user_guide/ppo_training.md — Diagram 2
# PPO Loss Formula
# ===================================================================


def render_ppo_2() -> str:
    c = SvgCanvas(
        width=700,
        height=260,
        title="PPO Loss Components",
        caption="Figure 2: Actor and Critic loss calculation",
    )

    b_ratio = c.box(x=40, y=100, w=120, h=BOX_HEIGHT, label="Ratio π/π_old", palette="training")
    b_adv = c.box(x=200, y=100, w=100, h=BOX_HEIGHT, label="Advantage", palette="training")
    b_clip = c.box(x=340, y=100, w=130, h=BOX_HEIGHT, label="Clip(ratio, ε)", palette="highlight")
    b_loss = c.box(x=510, y=100, w=140, h=BOX_HEIGHT, label="min(r*A, clip*A)", bold=True, palette="training")

    c.arrow(b_ratio, b_clip, label="", src_side="right", dst_side="left")
    c.arrow(b_adv, b_loss, label="", src_side="right", dst_side="left")
    c.arrow(b_clip, b_loss, label="", src_side="right", dst_side="left")

    return c.render()


# ===================================================================
# user_guide/agentic_multiturn.md — Diagram 1
# Multi-turn State Machine
# ===================================================================


def render_multiturn_1() -> str:
    c = SvgCanvas(
        width=500,
        height=360,
        title="Multi-turn Conversation States",
        caption="Figure 1: State transitions during multi-turn generation",
    )

    sd = c.state_diagram()
    sd.start_dot(x=250, y=70, name="start")
    sd.state("GEN", "Generating", x=150, y=110, w=200, h=36, palette_name="rollout")
    sd.state("TOOL", "Tool Execution", x=150, y=180, w=200, h=36, palette_name="tools")
    sd.state("DONE", "Completed", x=150, y=250, w=200, h=36, palette_name="training")
    sd.end_dot(x=250, y=320, name="end")

    sd.transition("start", "GEN", "", src_side="bottom", dst_side="top", waypoints=[])
    sd.transition("GEN", "TOOL", "tool_call", src_side="bottom", dst_side="top", waypoints=[])
    sd.transition("TOOL", "GEN", "response", src_side="left", dst_side="left", waypoints=[(100, 198), (100, 128)])
    sd.transition("GEN", "DONE", "EOS", src_side="right", dst_side="right", waypoints=[(380, 128), (380, 268)])
    sd.transition("DONE", "end", "", src_side="bottom", dst_side="top", waypoints=[])

    return c.render()


# ===================================================================
# user_guide/agentic_multiturn.md — Diagram 2
# Multi-turn Sequence
# ===================================================================


def render_multiturn_2() -> str:
    c = SvgCanvas(
        width=800,
        height=500,
        title="Multi-turn Sequence",
        caption="Figure 2: Message flow in multi-turn conversation",
    )

    seq = c.sequence(
        actors=[
            {"name": "User", "short": "User", "palette": "data"},
            {"name": "Model", "short": "Model", "palette": "rollout"},
            {"name": "Tool", "short": "Tool", "palette": "tools"},
        ],
        x=100,
        y=60,
        actor_spacing=250,
        msg_spacing=45,
    )

    seq.message("User", "Model", "User prompt")
    seq.message("Model", "Tool", "tool_call(args)")
    seq.message("Tool", "Model", "tool_response")
    seq.message("Model", "Tool", "tool_call(args)")
    seq.message("Tool", "Model", "tool_response")
    seq.message("Model", "User", "Final answer")

    return c.render()


# ===================================================================
# user_guide/aio_tool_infrastructure.md — Diagram 1
# AIO Architecture
# ===================================================================


def render_aio_tool_1() -> str:
    c = SvgCanvas(
        width=800,
        height=420,
        title="AIO Tool Infrastructure",
        caption="Figure 1: All-in-One tool environment",
    )

    # Rollout side
    r_rollout = c.region(x=30, y=55, w=250, h=140, title="Rollout Workers", palette="rollout")
    b_w1 = c.box(x=50, y=100, w=100, h=BOX_HEIGHT, label="Worker 1", palette="rollout")
    b_w2 = c.box(x=50, y=145, w=100, h=BOX_HEIGHT, label="Worker N", palette="rollout")

    # AIO Proxy
    r_aio = c.region(x=320, y=55, w=180, h=140, title="AIO Proxy", palette="highlight")
    b_router = c.box(x=340, y=100, w=140, h=BOX_HEIGHT, label="Request Router", bold=True, palette="highlight")

    # Tool Environments
    r_tools = c.region(x=540, y=55, w=230, h=240, title="Tool Environments", palette="tools")
    b_env1 = c.box(x=560, y=100, w=180, h=BOX_HEIGHT, label="ToolEnv Container 1", palette="tools")
    b_env2 = c.box(x=560, y=145, w=180, h=BOX_HEIGHT, label="ToolEnv Container 2", palette="tools")
    b_envn = c.box(x=560, y=190, w=180, h=BOX_HEIGHT, label="ToolEnv Container N", palette="tools")
    b_sandbox = c.box(x=560, y=245, w=180, h=BOX_HEIGHT, label="Sandbox Isolation", palette="infra")

    c.arrow(b_w1, b_router, label="tool_call", src_side="right", dst_side="left")
    c.arrow(b_w2, b_router, label="", src_side="right", dst_side="left")
    c.arrow(b_router, b_env1, label="dispatch", src_side="right", dst_side="left")
    c.arrow(b_router, b_env2, label="", src_side="right", dst_side="left")

    return c.render()


# ===================================================================
# user_guide/aio_tool_infrastructure.md — Diagram 2
# Tool Execution Sequence
# ===================================================================


def render_aio_tool_2() -> str:
    c = SvgCanvas(
        width=750,
        height=400,
        title="Tool Execution Sequence",
        caption="Figure 2: How tool calls are processed",
    )

    seq = c.sequence(
        actors=[
            {"name": "Worker", "short": "Worker", "palette": "rollout"},
            {"name": "AIO Proxy", "short": "AIO Proxy", "palette": "highlight"},
            {"name": "ToolEnv", "short": "ToolEnv", "palette": "tools"},
        ],
        x=80,
        y=60,
        actor_spacing=250,
        msg_spacing=45,
    )

    seq.message("Worker", "AIO Proxy", "tool_call(name, args)")
    seq.message("AIO Proxy", "ToolEnv", "dispatch()")
    seq.note("ToolEnv", "Execute in sandbox", side="right")
    seq.message("ToolEnv", "AIO Proxy", "result")
    seq.message("AIO Proxy", "Worker", "response")

    return c.render()


# ===================================================================
# user_guide/configuration_system.md — Diagram 1
# Config Hierarchy
# ===================================================================


def render_config_1() -> str:
    c = SvgCanvas(
        width=800,
        height=520,
        title="Configuration System",
        caption="Figure 1: Hierarchical configuration structure",
    )

    # Top level
    b_main = c.box(x=300, y=70, w=200, h=40, label="TrainConfig", bold=True, palette="infra")

    # Second level
    b_model = c.box(x=50, y=170, w=150, h=BOX_HEIGHT, label="ModelConfig", palette="training")
    b_rollout = c.box(x=230, y=170, w=150, h=BOX_HEIGHT, label="RolloutConfig", palette="rollout")
    b_data = c.box(x=410, y=170, w=150, h=BOX_HEIGHT, label="DataConfig", palette="data")
    b_train = c.box(x=590, y=170, w=150, h=BOX_HEIGHT, label="TrainerConfig", palette="training")

    # Third level (under ModelConfig)
    b_actor = c.box(x=50, y=260, w=130, h=BOX_HEIGHT, label="ActorConfig", palette="training")
    b_critic = c.box(x=50, y=310, w=130, h=BOX_HEIGHT, label="CriticConfig", palette="training")

    # Third level (under RolloutConfig)
    b_sglang = c.box(x=230, y=260, w=130, h=BOX_HEIGHT, label="SGLangConfig", palette="rollout")
    b_tool = c.box(x=230, y=310, w=130, h=BOX_HEIGHT, label="ToolConfig", palette="tools")

    c.arrow(b_main, b_model, label="", src_side="bottom", dst_side="top")
    c.arrow(b_main, b_rollout, label="", src_side="bottom", dst_side="top")
    c.arrow(b_main, b_data, label="", src_side="bottom", dst_side="top")
    c.arrow(b_main, b_train, label="", src_side="bottom", dst_side="top")
    c.arrow(b_model, b_actor, label="", src_side="bottom", dst_side="top")
    c.arrow(b_model, b_critic, label="", src_side="bottom", dst_side="top")
    c.arrow(b_rollout, b_sglang, label="", src_side="bottom", dst_side="top")
    c.arrow(b_rollout, b_tool, label="", src_side="bottom", dst_side="top")

    return c.render()


# ===================================================================
# user_guide/configuration_system.md — Diagram 2
# Config Loading Flow
# ===================================================================


def render_config_2() -> str:
    c = SvgCanvas(
        width=700,
        height=240,
        title="Config Loading",
        caption="Figure 2: How configuration is loaded",
    )

    b_yaml = c.box(x=40, y=100, w=100, h=BOX_HEIGHT, label="YAML File", palette="data")
    b_parse = c.box(x=180, y=100, w=100, h=BOX_HEIGHT, label="Parse", palette="infra")
    b_validate = c.box(x=320, y=100, w=100, h=BOX_HEIGHT, label="Validate", palette="highlight")
    b_config = c.box(x=460, y=100, w=140, h=BOX_HEIGHT, label="TrainConfig", bold=True, palette="infra")

    c.arrow(b_yaml, b_parse, label="", src_side="right", dst_side="left")
    c.arrow(b_parse, b_validate, label="", src_side="right", dst_side="left")
    c.arrow(b_validate, b_config, label="", src_side="right", dst_side="left")

    return c.render()


# ===================================================================
# user_guide/deployment_modes.md — Diagram 1
# Single Node Mode
# ===================================================================


def render_deploy_1() -> str:
    c = SvgCanvas(
        width=600,
        height=300,
        title="Single Node Deployment",
        caption="Figure 1: All components on one machine",
    )

    r_node = c.region(x=50, y=55, w=500, h=160, title="Single Node", palette="infra")
    b_train = c.box(x=70, y=100, w=130, h=BOX_HEIGHT, label="TrainerGroup", palette="training")
    b_rollout = c.box(x=230, y=100, w=140, h=BOX_HEIGHT, label="RolloutManager", palette="rollout")
    b_data = c.box(x=400, y=100, w=130, h=BOX_HEIGHT, label="DataCoordinator", palette="data")
    b_tool = c.box(x=230, y=160, w=140, h=BOX_HEIGHT, label="ToolEnv", palette="tools")

    c.arrow(b_train, b_rollout, label="", src_side="right", dst_side="left")
    c.arrow(b_rollout, b_data, label="", src_side="right", dst_side="left")
    c.arrow(b_rollout, b_tool, label="", src_side="bottom", dst_side="top")

    return c.render()


# ===================================================================
# user_guide/deployment_modes.md — Diagram 2
# Multi-Node Mode
# ===================================================================


def render_deploy_2() -> str:
    c = SvgCanvas(
        width=700,
        height=300,
        title="Multi-Node Deployment",
        caption="Figure 2: Distributed across multiple machines",
    )

    r_train = c.region(x=30, y=55, w=200, h=120, title="Training Node", palette="training")
    b_tg = c.box(x=50, y=100, w=160, h=BOX_HEIGHT, label="TrainerGroup", bold=True, palette="training")

    r_rollout = c.region(x=260, y=55, w=200, h=120, title="Rollout Node", palette="rollout")
    b_rm = c.box(x=280, y=100, w=160, h=BOX_HEIGHT, label="RolloutManager", bold=True, palette="rollout")

    r_tool = c.region(x=490, y=55, w=180, h=120, title="Tool Node", palette="tools")
    b_tool = c.box(x=510, y=100, w=140, h=BOX_HEIGHT, label="ToolEnv Pool", bold=True, palette="tools")

    c.arrow(b_tg, b_rm, label="weights", src_side="right", dst_side="left")
    c.arrow(b_rm, b_tool, label="tool_call", src_side="right", dst_side="left")

    return c.render()


# ===================================================================
# user_guide/deployment_modes.md — Diagram 3
# Colocated Mode
# ===================================================================


def render_deploy_3() -> str:
    c = SvgCanvas(
        width=600,
        height=300,
        title="Colocated Deployment",
        caption="Figure 3: Training and rollout share GPUs",
    )

    r_node = c.region(x=50, y=55, w=500, h=160, title="Colocated Node (Shared GPUs)", palette="highlight")
    b_train = c.box(x=70, y=100, w=200, h=BOX_HEIGHT, label="Training + Rollout", bold=True, palette="highlight")
    b_data = c.box(x=300, y=100, w=130, h=BOX_HEIGHT, label="DataCoordinator", palette="data")
    b_tool = c.box(x=70, y=160, w=100, h=BOX_HEIGHT, label="ToolEnv", palette="tools")

    c.arrow(b_train, b_data, label="", src_side="right", dst_side="left")
    c.arrow(b_train, b_tool, label="", src_side="bottom", dst_side="top")

    return c.render()


# ===================================================================
# user_guide/deployment_modes.md — Diagram 4
# Deployment Sequence
# ===================================================================


def render_deploy_4() -> str:
    c = SvgCanvas(
        width=800,
        height=450,
        title="Deployment Startup Sequence",
        caption="Figure 4: Order of component initialization",
    )

    seq = c.sequence(
        actors=[
            {"name": "CLI", "short": "CLI", "palette": "infra"},
            {"name": "Ray", "short": "Ray", "palette": "infra"},
            {"name": "Trainer", "short": "Trainer", "palette": "training"},
            {"name": "Rollout", "short": "Rollout", "palette": "rollout"},
        ],
        x=60,
        y=60,
        actor_spacing=180,
        msg_spacing=40,
    )

    seq.message("CLI", "Ray", "ray.init()")
    seq.message("Ray", "Trainer", "spawn TrainerGroup")
    seq.message("Ray", "Rollout", "spawn RolloutManager")
    seq.message("Trainer", "Rollout", "register_weights()")
    seq.message("Rollout", "Trainer", "ready()")
    seq.message("CLI", "Trainer", "start_training()")

    return c.render()


# ===================================================================
# user_guide/checkpoint_resume.md — Diagram 1
# Checkpoint Structure
# ===================================================================


def render_ckpt_1() -> str:
    c = SvgCanvas(
        width=700,
        height=400,
        title="Checkpoint Structure",
        caption="Figure 1: What's saved in a checkpoint",
    )

    # Checkpoint box
    r_ckpt = c.region(x=200, y=55, w=300, h=250, title="Checkpoint", palette="data")
    b_model = c.box(x=220, y=100, w=130, h=BOX_HEIGHT, label="Model Weights", palette="training")
    b_opt = c.box(x=220, y=150, w=130, h=BOX_HEIGHT, label="Optimizer State", palette="training")
    b_config = c.box(x=220, y=200, w=130, h=BOX_HEIGHT, label="Config", palette="infra")
    b_step = c.box(x=220, y=250, w=130, h=BOX_HEIGHT, label="Global Step", palette="highlight")

    # Load/Save arrows
    b_save = c.box(x=40, y=150, w=100, h=BOX_HEIGHT, label="save()", bold=True, palette="highlight")
    b_load = c.box(x=560, y=150, w=100, h=BOX_HEIGHT, label="load()", bold=True, palette="highlight")

    c.arrow(b_save, b_model, label="", src_side="right", dst_side="left")
    c.arrow(b_model, b_load, label="", src_side="right", dst_side="left")

    return c.render()


# ===================================================================
# user_guide/checkpoint_resume.md — Diagram 2
# Resume Flow
# ===================================================================


def render_ckpt_2() -> str:
    c = SvgCanvas(
        width=700,
        height=260,
        title="Resume Training Flow",
        caption="Figure 2: How training resumes from checkpoint",
    )

    b_load = c.box(x=40, y=100, w=110, h=BOX_HEIGHT, label="Load Ckpt", bold=True, palette="data")
    b_restore = c.box(x=190, y=100, w=140, h=BOX_HEIGHT, label="Restore State", palette="training")
    b_sync = c.box(x=370, y=100, w=120, h=BOX_HEIGHT, label="Sync Weights", palette="rollout")
    b_resume = c.box(x=530, y=100, w=120, h=BOX_HEIGHT, label="Resume Loop", bold=True, palette="training")

    c.arrow(b_load, b_restore, label="", src_side="right", dst_side="left")
    c.arrow(b_restore, b_sync, label="", src_side="right", dst_side="left")
    c.arrow(b_sync, b_resume, label="", src_side="right", dst_side="left")

    return c.render()


# ===================================================================
# user_guide/tool_env_and_swe.md — Diagram 1
# Tool Environment Architecture
# ===================================================================


def render_tool_swe_1() -> str:
    c = SvgCanvas(
        width=700,
        height=380,
        title="Tool Environment Architecture",
        caption="Figure 1: How tools are executed",
    )

    # Flow region
    r_flow = c.region(x=30, y=55, w=640, h=140, title="NaiveFlow", palette="rollout")
    b_gen = c.box(x=50, y=100, w=100, h=BOX_HEIGHT, label="Generate", palette="rollout")
    b_parse = c.box(x=180, y=100, w=100, h=BOX_HEIGHT, label="Parse", palette="rollout")
    b_call = c.box(x=310, y=100, w=110, h=BOX_HEIGHT, label="Tool Call", palette="highlight")
    b_result = c.box(x=450, y=100, w=100, h=BOX_HEIGHT, label="Result", palette="data")

    # Tool layer
    r_tool = c.region(x=30, y=220, w=640, h=100, title="Tool Environment", palette="tools")
    b_sandbox = c.box(x=50, y=260, w=150, h=BOX_HEIGHT, label="Sandbox Container", palette="tools")
    b_exec = c.box(x=230, y=260, w=120, h=BOX_HEIGHT, label="Execute Code", palette="tools")
    b_capture = c.box(x=380, y=260, w=140, h=BOX_HEIGHT, label="Capture Output", palette="data")

    c.arrow(b_gen, b_parse, label="", src_side="right", dst_side="left")
    c.arrow(b_parse, b_call, label="", src_side="right", dst_side="left")
    c.arrow(b_call, b_sandbox, label="", src_side="bottom", dst_side="top")
    c.arrow(b_sandbox, b_exec, label="", src_side="right", dst_side="left")
    c.arrow(b_exec, b_capture, label="", src_side="right", dst_side="left")
    c.arrow(b_capture, b_result, label="", src_side="top", dst_side="bottom")

    return c.render()


# ===================================================================
# user_guide/tool_env_and_swe.md — Diagram 2
# SWE Agent Flow
# ===================================================================


def render_tool_swe_2() -> str:
    c = SvgCanvas(
        width=700,
        height=260,
        title="SWE Agent Workflow",
        caption="Figure 2: Software engineering agent flow",
    )

    b_issue = c.box(x=40, y=100, w=100, h=BOX_HEIGHT, label="Issue", bold=True, palette="data")
    b_analyze = c.box(x=180, y=100, w=100, h=BOX_HEIGHT, label="Analyze", palette="rollout")
    b_edit = c.box(x=320, y=100, w=100, h=BOX_HEIGHT, label="Edit Code", palette="tools")
    b_test = c.box(x=460, y=100, w=100, h=BOX_HEIGHT, label="Run Tests", palette="tools")
    b_pr = c.box(x=600, y=100, w=70, h=BOX_HEIGHT, label="PR", bold=True, palette="highlight")

    c.arrow(b_issue, b_analyze, label="", src_side="right", dst_side="left")
    c.arrow(b_analyze, b_edit, label="", src_side="right", dst_side="left")
    c.arrow(b_edit, b_test, label="", src_side="right", dst_side="left")
    c.arrow(b_test, b_edit, label="fix", src_side="bottom", dst_side="bottom")
    c.arrow(b_test, b_pr, label="pass", src_side="right", dst_side="left")

    return c.render()


# ===================================================================
# user_guide/metrics_and_evaluation.md — Diagram 1
# Metrics Pipeline
# ===================================================================


def render_metrics_1() -> str:
    c = SvgCanvas(
        width=800,
        height=380,
        title="Metrics Collection Pipeline",
        caption="Figure 1: How metrics are collected and reported",
    )

    # Sources
    r_sources = c.region(x=30, y=55, w=250, h=180, title="Metric Sources", palette="infra")
    b_train = c.box(x=50, y=100, w=120, h=BOX_HEIGHT, label="TrainerGroup", palette="training")
    b_rollout = c.box(x=50, y=150, w=140, h=BOX_HEIGHT, label="RolloutManager", palette="rollout")
    b_reward = c.box(x=50, y=200, w=120, h=BOX_HEIGHT, label="Reward Fn", palette="highlight")

    # Collector
    r_collect = c.region(x=320, y=55, w=180, h=180, title="MetricWorker", palette="data")
    b_agg = c.box(x=340, y=120, w=140, h=BOX_HEIGHT, label="Aggregate", palette="data")
    b_log = c.box(x=340, y=180, w=140, h=BOX_HEIGHT, label="Log", palette="data")

    # Outputs
    r_out = c.region(x=540, y=55, w=220, h=180, title="Outputs", palette="highlight")
    b_wandb = c.box(x=560, y=100, w=100, h=BOX_HEIGHT, label="W&B", palette="highlight")
    b_tb = c.box(x=560, y=150, w=120, h=BOX_HEIGHT, label="TensorBoard", palette="highlight")
    b_file = c.box(x=560, y=200, w=100, h=BOX_HEIGHT, label="JSON File", palette="data")

    c.arrow(b_train, b_agg, label="", src_side="right", dst_side="left")
    c.arrow(b_rollout, b_agg, label="", src_side="right", dst_side="left")
    c.arrow(b_reward, b_agg, label="", src_side="right", dst_side="left")
    c.arrow(b_log, b_wandb, label="", src_side="right", dst_side="left")
    c.arrow(b_log, b_tb, label="", src_side="right", dst_side="left")
    c.arrow(b_log, b_file, label="", src_side="right", dst_side="left")

    return c.render()


# ===================================================================
# user_guide/validate_reuse_and_eval_scaling.md — Diagram 1
# Validation Pipeline
# ===================================================================


def render_validate_1() -> str:
    c = SvgCanvas(
        width=800,
        height=340,
        title="Validation Pipeline",
        caption="Figure 1: How validation runs during training",
    )

    # Training loop
    r_train = c.region(x=30, y=55, w=350, h=140, title="Training Loop", palette="training")
    b_step = c.box(x=50, y=100, w=120, h=BOX_HEIGHT, label="Training Step", palette="training")
    b_check = c.box(x=200, y=100, w=150, h=BOX_HEIGHT, label="Check Interval", palette="infra")

    # Validation
    r_val = c.region(x=420, y=55, w=350, h=140, title="Validation", palette="rollout")
    b_eval = c.box(x=440, y=100, w=140, h=BOX_HEIGHT, label="Run Eval Set", palette="rollout")
    b_metric = c.box(x=610, y=100, w=130, h=BOX_HEIGHT, label="Compute Metrics", palette="highlight")

    c.arrow(b_step, b_check, label="", src_side="right", dst_side="left")
    c.arrow(b_check, b_eval, label="if interval", src_side="right", dst_side="left")
    c.arrow(b_eval, b_metric, label="", src_side="right", dst_side="left")
    c.arrow(b_check, b_step, label="continue", src_side="bottom", dst_side="bottom")

    return c.render()


# ===================================================================
# user_guide/validate_reuse_and_eval_scaling.md — Diagram 2
# Eval Scaling Gantt
# ===================================================================


def render_validate_2() -> str:
    c = SvgCanvas(
        width=700,
        height=300,
        title="Eval Scaling Timeline",
        caption="Figure 2: Parallel evaluation scaling",
    )

    # Manual gantt-style chart
    y_base = 100
    bar_h = 30
    gap = 15

    # Labels
    c._elements.append(f'<text x="30" y="{y_base + 20}" fill="#424242" font-size="{BOX_TEXT_SIZE}">Eval Worker 1</text>')
    c._elements.append(f'<text x="30" y="{y_base + bar_h + gap + 20}" fill="#424242" font-size="{BOX_TEXT_SIZE}">Eval Worker 2</text>')
    c._elements.append(f'<text x="30" y="{y_base + 2*(bar_h + gap) + 20}" fill="#424242" font-size="{BOX_TEXT_SIZE}">Eval Worker 3</text>')

    # Bars (staggered for parallelism)
    pal = PALETTES["rollout"]
    c._elements.append(
        f'<rect x="150" y="{y_base}" width="200" height="{bar_h}" rx="2" fill="{pal["bg"]}" stroke="{pal["border"]}" stroke-width="1"/>'
    )
    c._elements.append(
        f'<text x="250" y="{y_base + 20}" text-anchor="middle" fill="{pal["title_color"]}" font-size="{LABEL_SIZE}">Batch 1</text>'
    )

    c._elements.append(
        f'<rect x="180" y="{y_base + bar_h + gap}" width="200" height="{bar_h}" rx="2" fill="{pal["bg"]}" stroke="{pal["border"]}" stroke-width="1"/>'
    )
    c._elements.append(
        f'<text x="280" y="{y_base + bar_h + gap + 20}" text-anchor="middle" fill="{pal["title_color"]}" font-size="{LABEL_SIZE}">Batch 2</text>'
    )

    c._elements.append(
        f'<rect x="210" y="{y_base + 2*(bar_h + gap)}" width="200" height="{bar_h}" rx="2" fill="{pal["bg"]}" stroke="{pal["border"]}" stroke-width="1"/>'
    )
    c._elements.append(
        f'<text x="310" y="{y_base + 2*(bar_h + gap) + 20}" text-anchor="middle" fill="{pal["title_color"]}" font-size="{LABEL_SIZE}">Batch 3</text>'
    )

    # Time axis
    c._elements.append(
        f'<line x1="150" y1="{y_base + 3*(bar_h + gap)}" x2="600" y2="{y_base + 3*(bar_h + gap)}" stroke="#78909C" stroke-width="1" marker-end="url(#arr)"/>'
    )
    c._elements.append(
        f'<text x="375" y="{y_base + 3*(bar_h + gap) + 20}" text-anchor="middle" fill="#607D8B" font-size="{LABEL_SIZE}">Time →</text>'
    )

    return c.render()


# ===================================================================
# Registry: map output filenames to render functions
# ===================================================================

REGISTRY = {
    "user_guidegrpo_training_1.svg": render_grpo_1,
    "user_guidegrpo_training_2.svg": render_grpo_2,
    "user_guideppo_training_1.svg": render_ppo_1,
    "user_guideppo_training_2.svg": render_ppo_2,
    "user_guideagentic_multiturn_1.svg": render_multiturn_1,
    "user_guideagentic_multiturn_2.svg": render_multiturn_2,
    "user_guideaio_tool_infrastructure_1.svg": render_aio_tool_1,
    "user_guideaio_tool_infrastructure_2.svg": render_aio_tool_2,
    "user_guideconfiguration_system_1.svg": render_config_1,
    "user_guideconfiguration_system_2.svg": render_config_2,
    "user_guidedeployment_modes_1.svg": render_deploy_1,
    "user_guidedeployment_modes_2.svg": render_deploy_2,
    "user_guidedeployment_modes_3.svg": render_deploy_3,
    "user_guidedeployment_modes_4.svg": render_deploy_4,
    "user_guidecheckpoint_resume_1.svg": render_ckpt_1,
    "user_guidecheckpoint_resume_2.svg": render_ckpt_2,
    "user_guidetool_env_and_swe_1.svg": render_tool_swe_1,
    "user_guidetool_env_and_swe_2.svg": render_tool_swe_2,
    "user_guidemetrics_and_evaluation_1.svg": render_metrics_1,
    "user_guidevalidate_reuse_and_eval_scaling_1.svg": render_validate_1,
    "user_guidevalidate_reuse_and_eval_scaling_2.svg": render_validate_2,
}


if __name__ == "__main__":
    from pathlib import Path

    out_dir = Path(__file__).parent / "docs/assets/images/diagrams"
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname, func in REGISTRY.items():
        svg = func()
        (out_dir / fname).write_text(svg)
        print(f"Generated {fname}")
