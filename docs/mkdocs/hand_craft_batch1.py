#!/usr/bin/env python3
"""Hand-crafted SVG rendering functions for Batch 1 diagrams:
  - overview (2 diagrams)
  - concepts/architecture_overview (4 diagrams)
  - concepts/async_training_lifecycle (4 diagrams)
  - concepts/design_philosophy (2 diagrams)
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
# overview.md — Diagram 1
# High-level architecture: Training/Rollout/Data layers
# ===================================================================


def render_overview_1() -> str:
    c = SvgCanvas(
        width=900,
        height=340,
        title="SIIRL-Agentic High-Level Architecture",
        caption="Figure 1: Three-layer architecture — Training, Rollout, and Data",
    )

    # --- Training Layer (left) ---
    r_train = c.region(x=30, y=55, w=160, h=130, title="Training Layer", palette="training")
    b_trainer = r_train.box("TrainerGroup", bold=True)

    # --- Rollout Layer (center, large) ---
    r_rollout = c.region(x=210, y=55, w=440, h=200, title="Rollout Layer", palette="rollout")
    b_rm = c.box(x=230, y=100, w=140, h=BOX_HEIGHT, label="RolloutManager", bold=True, palette="rollout")
    b_sglang = c.box(x=230, y=180, w=140, h=BOX_HEIGHT, label="SGLang Engines", palette="rollout")

    # Decision diamond
    b_decision = c.diamond(x=440, y=140, size=70, label="Tool calls?", palette="rollout")

    b_toolenv = c.box(x=530, y=100, w=110, h=BOX_HEIGHT, label="AIO Tool Env", palette="tools")
    b_reward = c.box(x=530, y=180, w=110, h=BOX_HEIGHT, label="Reward Compute", palette="rollout")

    # --- Data Layer (right) ---
    r_data = c.region(x=670, y=55, w=200, h=130, title="Data Layer", palette="data")
    b_dc = c.box(x=690, y=100, w=160, h=BOX_HEIGHT, label="DataCoordinator", bold=True, palette="data")

    # --- Arrows ---
    c.arrow(b_dc, b_rm, label="prompts", src_side="left", dst_side="right")
    c.arrow(b_rm, b_sglang, label="", src_side="bottom", dst_side="top")
    c.arrow(b_sglang, b_decision, label="generate", src_side="right", dst_side="left")
    c.arrow(b_decision, b_toolenv, label="Yes", src_side="right", dst_side="left")
    c.arrow(b_toolenv, b_sglang, label="response", src_side="bottom", dst_side="right")
    c.arrow(b_decision, b_reward, label="No", src_side="bottom", dst_side="top")
    c.arrow(b_reward, b_dc, label="scored samples", src_side="right", dst_side="bottom")
    c.arrow(b_dc, b_trainer, label="training batch", src_side="left", dst_side="right")
    c.arrow(b_trainer, b_rm, label="weights", src_side="right", dst_side="left")

    return c.render()


# ===================================================================
# overview.md — Diagram 2
# Sequence diagram: Async training loop
# ===================================================================


def render_overview_2() -> str:
    c = SvgCanvas(
        width=900,
        height=620,
        title="Async Training Loop",
        caption="Figure 2: Sequence of async rollout and training",
    )

    # Create sequence diagram
    seq = c.sequence(
        actors=[
            {"name": "DataCoordinator", "short": "DataCoordinator", "palette": "data"},
            {"name": "RolloutManager", "short": "RolloutManager", "palette": "rollout"},
            {"name": "SGLang Engine", "short": "SGLang Engine", "palette": "rollout"},
            {"name": "Tool Environment", "short": "Tool Environment", "palette": "tools"},
            {"name": "TrainerGroup", "short": "TrainerGroup", "palette": "training"},
        ],
        x=60,
        y=60,
        actor_spacing=170,
        msg_spacing=40,
    )

    seq.loop("Async Training Loop")
    seq.message("DataCoordinator", "RolloutManager", "Send prompt batch")
    seq.message("RolloutManager", "SGLang Engine", "Submit generation requests")
    seq.message("SGLang Engine", "RolloutManager", "Model response (with tool calls)")
    seq.message("RolloutManager", "Tool Environment", "Execute tool calls")
    seq.message("Tool Environment", "RolloutManager", "Tool responses")
    seq.message("RolloutManager", "SGLang Engine", "Continue generation (next turn)")
    seq.message("SGLang Engine", "RolloutManager", "Final response")
    seq.note("RolloutManager", "Compute reward", side="right")
    seq.message("RolloutManager", "DataCoordinator", "Return scored samples")
    seq.message("DataCoordinator", "TrainerGroup", "Provide training batch")
    seq.note("TrainerGroup", "Forward + backward + optimizer step", side="right")
    seq.message("TrainerGroup", "RolloutManager", "Sync updated weights")
    seq.end_loop()

    return c.render()


# ===================================================================
# concepts/architecture_overview.md — Diagram 1
# Full system architecture with all components
# ===================================================================


def render_arch_overview_1() -> str:
    c = SvgCanvas(
        width=900,
        height=820,
        title="System Architecture Overview",
        caption="Figure 1: Complete SIIRL-Agentic architecture with all components",
    )

    # --- Driver Process (top) ---
    r_driver = c.region(x=30, y=55, w=200, h=140, title="Driver Process", palette="infra")
    b_main = c.box(x=50, y=95, w=100, h=BOX_HEIGHT, label="main()", palette="infra")
    b_runner = c.box(x=50, y=135, w=120, h=BOX_HEIGHT, label="MainRunner", bold=True, palette="infra")

    # --- Resource Allocation ---
    b_task = c.box(x=260, y=95, w=150, h=BOX_HEIGHT, label="TaskCoordinator", palette="infra")
    b_alloc = c.box(x=260, y=135, w=150, h=BOX_HEIGHT, label="allocate_resources", palette="infra")

    # --- Monitoring (top right) ---
    r_monitor = c.region(x=450, y=55, w=200, h=90, title="Monitoring", palette="infra")
    b_metric = c.box(x=470, y=95, w=130, h=BOX_HEIGHT, label="MetricWorker", palette="infra")

    b_validate = c.box(x=700, y=95, w=150, h=BOX_HEIGHT, label="ValidateMonitor", palette="infra")

    # --- Training Cluster ---
    r_train = c.region(x=30, y=220, w=400, h=180, title="Training Cluster", palette="training")
    b_tg = c.box(x=50, y=265, w=140, h=BOX_HEIGHT, label="TrainerGroup", bold=True, palette="training")
    b_t1 = c.box(x=50, y=310, w=100, h=BOX_HEIGHT, label="Trainer 1", palette="training")
    b_t2 = c.box(x=170, y=310, w=100, h=BOX_HEIGHT, label="Trainer 2", palette="training")
    b_tn = c.box(x=290, y=310, w=100, h=BOX_HEIGHT, label="Trainer N", palette="training")
    b_actor = c.box(x=50, y=355, w=100, h=BOX_HEIGHT, label="Actor Model", palette="training")
    b_ref = c.box(x=170, y=355, w=90, h=BOX_HEIGHT, label="Ref Model", palette="training")
    b_critic = c.box(x=280, y=355, w=100, h=BOX_HEIGHT, label="Critic Model", palette="training")

    # --- Rollout Cluster ---
    r_rollout = c.region(x=30, y=420, w=840, h=180, title="Rollout Cluster", palette="rollout")
    b_rm = c.box(x=50, y=465, w=150, h=BOX_HEIGHT, label="RolloutManager", bold=True, palette="rollout")
    b_rw1 = c.box(x=230, y=465, w=130, h=BOX_HEIGHT, label="RolloutWorker 1", palette="rollout")
    b_rw2 = c.box(x=380, y=465, w=130, h=BOX_HEIGHT, label="RolloutWorker 2", palette="rollout")
    b_rwn = c.box(x=530, y=465, w=130, h=BOX_HEIGHT, label="RolloutWorker N", palette="rollout")
    b_sg1 = c.box(x=230, y=510, w=130, h=BOX_HEIGHT, label="SGLang Engine", palette="rollout")
    b_sg2 = c.box(x=380, y=510, w=130, h=BOX_HEIGHT, label="SGLang Engine", palette="rollout")
    b_sgn = c.box(x=530, y=510, w=130, h=BOX_HEIGHT, label="SGLang Engine", palette="rollout")
    b_nf1 = c.box(x=230, y=555, w=100, h=BOX_HEIGHT, label="NaiveFlow", palette="rollout")
    b_nf2 = c.box(x=350, y=555, w=100, h=BOX_HEIGHT, label="NaiveFlow", palette="rollout")
    b_nf3 = c.box(x=470, y=555, w=100, h=BOX_HEIGHT, label="NaiveFlow", palette="rollout")

    # --- Data Layer ---
    r_data = c.region(x=680, y=465, w=180, h=120, title="Data Layer", palette="data")
    b_dc = c.box(x=700, y=505, w=140, h=BOX_HEIGHT, label="DataCoordinator", bold=True, palette="data")
    b_loader = c.box(x=700, y=545, w=100, h=BOX_HEIGHT, label="Dataloader", palette="data")

    # --- Tool Layer ---
    r_tool = c.region(x=30, y=620, w=200, h=120, title="Tool Layer", palette="tools")
    b_toolenv = c.box(x=50, y=660, w=100, h=BOX_HEIGHT, label="ToolEnv", palette="tools")
    b_aio = c.box(x=50, y=700, w=100, h=BOX_HEIGHT, label="AIO Proxy", palette="tools")

    # --- Key Arrows ---
    c.arrow(b_main, b_runner, label="", src_side="bottom", dst_side="top")
    c.arrow(b_runner, b_task, label="", src_side="right", dst_side="left")
    c.arrow(b_task, b_alloc, label="", src_side="bottom", dst_side="top")
    c.arrow(b_alloc, b_tg, label="spawn", src_side="bottom", dst_side="top")
    c.arrow(b_alloc, b_rm, label="spawn", src_side="bottom", dst_side="top")
    c.arrow(b_alloc, b_metric, label="spawn", src_side="right", dst_side="bottom")
    c.arrow(b_tg, b_t1, label="", src_side="bottom", dst_side="top")
    c.arrow(b_tg, b_t2, label="", src_side="bottom", dst_side="top")
    c.arrow(b_tg, b_tn, label="", src_side="bottom", dst_side="top")
    c.arrow(b_rm, b_rw1, label="", src_side="right", dst_side="left")
    c.arrow(b_rm, b_rw2, label="", src_side="right", dst_side="left")
    c.arrow(b_rw1, b_sg1, label="", src_side="bottom", dst_side="top")
    c.arrow(b_rw2, b_sg2, label="", src_side="bottom", dst_side="top")
    c.arrow(b_dc, b_rm, label="prompts", src_side="left", dst_side="right")
    c.arrow(b_rm, b_dc, label="samples", src_side="right", dst_side="left")
    c.arrow(b_dc, b_tg, label="batch", src_side="top", dst_side="bottom")
    c.arrow(b_tg, b_rm, label="weights", src_side="bottom", dst_side="top")
    c.arrow(b_nf1, b_toolenv, label="tool call", src_side="bottom", dst_side="top")
    c.arrow(b_toolenv, b_aio, label="", src_side="bottom", dst_side="top")

    return c.render()


# ===================================================================
# concepts/architecture_overview.md — Diagram 2
# Data flow: left-to-right training pipeline
# ===================================================================


def render_arch_overview_2() -> str:
    c = SvgCanvas(
        width=880,
        height=320,
        title="Training Data Flow",
        caption="Figure 2: Data flow from prompts to trained model",
    )

    # Horizontal pipeline
    b_prompts = c.box(x=40, y=120, w=100, h=BOX_HEIGHT, label="Prompts", bold=True, palette="data")
    b_rm = c.box(x=180, y=120, w=140, h=BOX_HEIGHT, label="RolloutManager", palette="rollout")
    b_sglang = c.box(x=360, y=120, w=120, h=BOX_HEIGHT, label="SGLang", palette="rollout")
    b_reward = c.box(x=520, y=120, w=100, h=BOX_HEIGHT, label="Reward", palette="highlight")
    b_dc = c.box(x=660, y=120, w=130, h=BOX_HEIGHT, label="DataCoordinator", palette="data")

    # Bottom row: training
    b_trainer = c.box(x=660, y=200, w=130, h=BOX_HEIGHT, label="TrainerGroup", bold=True, palette="training")
    b_model = c.box(x=520, y=200, w=100, h=BOX_HEIGHT, label="Actor Model", palette="training")

    # Arrows
    c.arrow(b_prompts, b_rm, label="batch", src_side="right", dst_side="left")
    c.arrow(b_rm, b_sglang, label="generate", src_side="right", dst_side="left")
    c.arrow(b_sglang, b_reward, label="output", src_side="right", dst_side="left")
    c.arrow(b_reward, b_dc, label="scored", src_side="right", dst_side="left")
    c.arrow(b_dc, b_trainer, label="batch", src_side="bottom", dst_side="top")
    c.arrow(b_trainer, b_model, label="update", src_side="left", dst_side="right")
    c.arrow(b_model, b_rm, label="sync weights", src_side="top", dst_side="bottom")

    return c.render()


# ===================================================================
# concepts/architecture_overview.md — Diagram 3
# Simple component overview
# ===================================================================


def render_arch_overview_3() -> str:
    c = SvgCanvas(
        width=700,
        height=260,
        title="Core Components",
        caption="Figure 3: Three core components of SIIRL-Agentic",
    )

    b_rm = c.box(x=50, y=100, w=160, h=40, label="RolloutManager", bold=True, palette="rollout")
    b_dc = c.box(x=270, y=100, w=160, h=40, label="DataCoordinator", bold=True, palette="data")
    b_tg = c.box(x=490, y=100, w=160, h=40, label="TrainerGroup", bold=True, palette="training")

    c.arrow(b_rm, b_dc, label="samples", src_side="right", dst_side="left")
    c.arrow(b_dc, b_tg, label="batch", src_side="right", dst_side="left")
    c.arrow(b_tg, b_rm, label="weights", src_side="top", dst_side="top")

    return c.render()


# ===================================================================
# concepts/architecture_overview.md — Diagram 4
# Resource allocation flow
# ===================================================================


def render_arch_overview_4() -> str:
    c = SvgCanvas(
        width=800,
        height=360,
        title="Resource Allocation",
        caption="Figure 4: How resources are allocated across clusters",
    )

    # Top: coordinator
    r_coord = c.region(x=280, y=55, w=240, h=80, title="Coordinator", palette="infra")
    b_alloc = c.box(x=300, y=90, w=200, h=BOX_HEIGHT, label="allocate_resources()", bold=True, palette="infra")

    # Left: Training GPUs
    r_train = c.region(x=40, y=170, w=200, h=120, title="Training GPUs", palette="training")
    b_t1 = c.box(x=60, y=210, w=80, h=BOX_HEIGHT, label="GPU 0", palette="training")
    b_t2 = c.box(x=60, y=250, w=80, h=BOX_HEIGHT, label="GPU 1", palette="training")

    # Right: Rollout GPUs
    r_rollout = c.region(x=280, y=170, w=200, h=120, title="Rollout GPUs", palette="rollout")
    b_r1 = c.box(x=300, y=210, w=80, h=BOX_HEIGHT, label="GPU 2", palette="rollout")
    b_r2 = c.box(x=300, y=250, w=80, h=BOX_HEIGHT, label="GPU 3", palette="rollout")

    # Far right: Tool CPUs
    r_tool = c.region(x=520, y=170, w=200, h=120, title="Tool CPUs", palette="tools")
    b_cpu = c.box(x=540, y=210, w=160, h=BOX_HEIGHT, label="CPU Workers", palette="tools")

    c.arrow(b_alloc, b_t1, label="spawn", src_side="bottom", dst_side="top")
    c.arrow(b_alloc, b_r1, label="spawn", src_side="bottom", dst_side="top")
    c.arrow(b_alloc, b_cpu, label="spawn", src_side="bottom", dst_side="top")

    return c.render()


# ===================================================================
# concepts/async_training_lifecycle.md — Diagram 1
# Training lifecycle states
# ===================================================================


def render_async_lifecycle_1() -> str:
    c = SvgCanvas(
        width=800,
        height=300,
        title="Async Training Lifecycle",
        caption="Figure 1: States in the async training loop",
    )

    # Linear flow of states
    b_init = c.box(x=40, y=120, w=100, h=BOX_HEIGHT, label="Initialize", bold=True, palette="infra")
    b_rollout = c.box(x=180, y=120, w=100, h=BOX_HEIGHT, label="Rollout", palette="rollout")
    b_collect = c.box(x=320, y=120, w=100, h=BOX_HEIGHT, label="Collect", palette="data")
    b_train = c.box(x=460, y=120, w=100, h=BOX_HEIGHT, label="Train", palette="training")
    b_sync = c.box(x=600, y=120, w=100, h=BOX_HEIGHT, label="Sync", palette="highlight")

    c.arrow(b_init, b_rollout, label="", src_side="right", dst_side="left")
    c.arrow(b_rollout, b_collect, label="", src_side="right", dst_side="left")
    c.arrow(b_collect, b_train, label="", src_side="right", dst_side="left")
    c.arrow(b_train, b_sync, label="", src_side="right", dst_side="left")
    c.arrow(b_sync, b_rollout, label="loop", src_side="top", dst_side="top")

    return c.render()


# ===================================================================
# concepts/async_training_lifecycle.md — Diagram 2
# Detailed sequence of one training iteration
# ===================================================================


def render_async_lifecycle_2() -> str:
    c = SvgCanvas(
        width=900,
        height=700,
        title="Training Iteration Sequence",
        caption="Figure 2: Detailed sequence of one async training iteration",
    )

    seq = c.sequence(
        actors=[
            {"name": "MainRunner", "short": "MainRunner", "palette": "infra"},
            {"name": "RolloutManager", "short": "RolloutManager", "palette": "rollout"},
            {"name": "SGLang", "short": "SGLang", "palette": "rollout"},
            {"name": "DataCoordinator", "short": "DataCoordinator", "palette": "data"},
            {"name": "TrainerGroup", "short": "TrainerGroup", "palette": "training"},
        ],
        x=50,
        y=60,
        actor_spacing=170,
        msg_spacing=38,
    )

    seq.message("MainRunner", "RolloutManager", "start_rollout()")
    seq.message("RolloutManager", "SGLang", "generate(prompts)")
    seq.note("SGLang", "Multi-turn generation", side="right")
    seq.message("SGLang", "RolloutManager", "completions")
    seq.note("RolloutManager", "Compute rewards", side="right")
    seq.message("RolloutManager", "DataCoordinator", "submit_samples()")
    seq.message("DataCoordinator", "TrainerGroup", "get_batch()")
    seq.message("TrainerGroup", "TrainerGroup", "forward + backward")
    seq.message("TrainerGroup", "DataCoordinator", "batch_done()")
    seq.message("TrainerGroup", "RolloutManager", "sync_weights()")
    seq.message("RolloutManager", "MainRunner", "iteration_complete()")

    return c.render()


# ===================================================================
# concepts/async_training_lifecycle.md — Diagram 3
# Async overlap: rollout vs training
# ===================================================================


def render_async_lifecycle_3() -> str:
    c = SvgCanvas(
        width=800,
        height=360,
        title="Async Overlap",
        caption="Figure 3: Rollout and training can overlap in time",
    )

    # Rollout timeline (top)
    r_rollout = c.region(x=40, y=55, w=720, h=90, title="Rollout Workers", palette="rollout")
    b_r1 = c.box(x=60, y=95, w=150, h=BOX_HEIGHT, label="Batch 1 rollout", palette="rollout")
    b_r2 = c.box(x=230, y=95, w=150, h=BOX_HEIGHT, label="Batch 2 rollout", palette="rollout")
    b_r3 = c.box(x=400, y=95, w=150, h=BOX_HEIGHT, label="Batch 3 rollout", palette="rollout")

    # Training timeline (bottom)
    r_train = c.region(x=40, y=170, w=720, h=90, title="Training Workers", palette="training")
    b_t1 = c.box(x=150, y=210, w=130, h=BOX_HEIGHT, label="Train Batch 1", palette="training")
    b_t2 = c.box(x=300, y=210, w=130, h=BOX_HEIGHT, label="Train Batch 2", palette="training")
    b_t3 = c.box(x=450, y=210, w=130, h=BOX_HEIGHT, label="Train Batch 3", palette="training")

    # Arrows showing data flow
    c.arrow(b_r1, b_t1, label="", src_side="bottom", dst_side="top", dashed=True)
    c.arrow(b_r2, b_t2, label="", src_side="bottom", dst_side="top", dashed=True)
    c.arrow(b_r3, b_t3, label="", src_side="bottom", dst_side="top", dashed=True)

    return c.render()


# ===================================================================
# concepts/async_training_lifecycle.md — Diagram 4
# Weight sync mechanism
# ===================================================================


def render_async_lifecycle_4() -> str:
    c = SvgCanvas(
        width=800,
        height=360,
        title="Weight Synchronization",
        caption="Figure 4: How model weights are synced between clusters",
    )

    # Training side
    r_train = c.region(x=40, y=80, w=250, h=180, title="Training Cluster", palette="training")
    b_actor = c.box(x=60, y=125, w=120, h=BOX_HEIGHT, label="Actor Model", bold=True, palette="training")
    b_opt = c.box(x=60, y=170, w=120, h=BOX_HEIGHT, label="Optimizer", palette="training")
    b_grad = c.box(x=60, y=215, w=120, h=BOX_HEIGHT, label="Gradients", palette="training")

    # Sync arrow
    b_sync = c.box(x=340, y=145, w=120, h=BOX_HEIGHT, label="sync_weights()", bold=True, palette="highlight")

    # Rollout side
    r_rollout = c.region(x=510, y=80, w=250, h=180, title="Rollout Cluster", palette="rollout")
    b_infer = c.box(x=530, y=125, w=140, h=BOX_HEIGHT, label="Inference Model", bold=True, palette="rollout")
    b_sglang = c.box(x=530, y=170, w=140, h=BOX_HEIGHT, label="SGLang Engine", palette="rollout")

    c.arrow(b_opt, b_actor, label="update", src_side="top", dst_side="bottom")
    c.arrow(b_actor, b_sync, label="params", src_side="right", dst_side="left")
    c.arrow(b_sync, b_infer, label="broadcast", src_side="right", dst_side="left")
    c.arrow(b_infer, b_sglang, label="", src_side="bottom", dst_side="top")

    return c.render()


# ===================================================================
# concepts/design_philosophy.md — Diagram 1
# Design principles
# ===================================================================


def render_design_philosophy_1() -> str:
    c = SvgCanvas(
        width=800,
        height=380,
        title="Design Philosophy",
        caption="Figure 1: Core design principles of SIIRL-Agentic",
    )

    # Top: main principle
    r_main = c.region(x=250, y=55, w=300, h=80, title="Core Principle", palette="highlight")
    b_async = c.box(x=270, y=90, w=260, h=BOX_HEIGHT, label="Async-First Architecture", bold=True, palette="highlight")

    # Bottom row: three pillars
    r_decouple = c.region(x=40, y=180, w=220, h=120, title="Decoupling", palette="rollout")
    b_dec = c.box(x=60, y=220, w=180, h=BOX_HEIGHT, label="Separate rollout/train", palette="rollout")

    r_scale = c.region(x=290, y=180, w=220, h=120, title="Scalability", palette="training")
    b_scale = c.box(x=310, y=220, w=180, h=BOX_HEIGHT, label="Linear GPU scaling", palette="training")

    r_flex = c.region(x=540, y=180, w=220, h=120, title="Flexibility", palette="tools")
    b_flex = c.box(x=560, y=220, w=180, h=BOX_HEIGHT, label="Pluggable components", palette="tools")

    c.arrow(b_async, b_dec, label="", src_side="bottom", dst_side="top")
    c.arrow(b_async, b_scale, label="", src_side="bottom", dst_side="top")
    c.arrow(b_async, b_flex, label="", src_side="bottom", dst_side="top")

    return c.render()


# ===================================================================
# concepts/design_philosophy.md — Diagram 2
# Comparison: sync vs async
# ===================================================================


def render_design_philosophy_2() -> str:
    c = SvgCanvas(
        width=800,
        height=360,
        title="Sync vs Async Training",
        caption="Figure 2: Comparison of synchronous and asynchronous approaches",
    )

    # Left: Sync (bad)
    r_sync = c.region(x=40, y=55, w=340, h=200, title="Synchronous (Blocking)", palette="error")
    b_s1 = c.box(x=60, y=100, w=130, h=BOX_HEIGHT, label="Rollout", palette="rollout")
    b_s2 = c.box(x=60, y=150, w=130, h=BOX_HEIGHT, label="Wait...", palette="infra")
    b_s3 = c.box(x=60, y=200, w=130, h=BOX_HEIGHT, label="Train", palette="training")
    c.box(x=220, y=150, w=130, h=BOX_HEIGHT, label="GPU Idle!", palette="error")

    # Right: Async (good)
    r_async = c.region(x=420, y=55, w=340, h=200, title="Asynchronous (Overlap)", palette="training")
    b_a1 = c.box(x=440, y=100, w=130, h=BOX_HEIGHT, label="Rollout 1", palette="rollout")
    b_a2 = c.box(x=440, y=150, w=130, h=BOX_HEIGHT, label="Rollout 2", palette="rollout")
    b_a3 = c.box(x=600, y=100, w=120, h=BOX_HEIGHT, label="Train 1", palette="training")
    b_a4 = c.box(x=600, y=150, w=120, h=BOX_HEIGHT, label="Train 2", palette="training")

    c.arrow(b_s1, b_s2, label="", src_side="bottom", dst_side="top")
    c.arrow(b_s2, b_s3, label="", src_side="bottom", dst_side="top")
    c.arrow(b_a1, b_a3, label="", src_side="right", dst_side="left", dashed=True)
    c.arrow(b_a2, b_a4, label="", src_side="right", dst_side="left", dashed=True)

    return c.render()


# ===================================================================
# Registry: map output filenames to render functions
# ===================================================================

REGISTRY = {
    "overview_1.svg": render_overview_1,
    "overview_2.svg": render_overview_2,
    "conceptsarchitecture_overview_1.svg": render_arch_overview_1,
    "conceptsarchitecture_overview_2.svg": render_arch_overview_2,
    "conceptsarchitecture_overview_3.svg": render_arch_overview_3,
    "conceptsarchitecture_overview_4.svg": render_arch_overview_4,
    "conceptsasync_training_lifecycle_1.svg": render_async_lifecycle_1,
    "conceptsasync_training_lifecycle_2.svg": render_async_lifecycle_2,
    "conceptsasync_training_lifecycle_3.svg": render_async_lifecycle_3,
    "conceptsasync_training_lifecycle_4.svg": render_async_lifecycle_4,
    "conceptsdesign_philosophy_1.svg": render_design_philosophy_1,
    "conceptsdesign_philosophy_2.svg": render_design_philosophy_2,
}


if __name__ == "__main__":
    from pathlib import Path

    out_dir = Path(__file__).parent / "docs/assets/images/diagrams"
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname, func in REGISTRY.items():
        svg = func()
        (out_dir / fname).write_text(svg)
        print(f"Generated {fname}")
