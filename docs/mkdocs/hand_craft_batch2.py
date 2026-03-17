#!/usr/bin/env python3
"""Hand-crafted SVG rendering functions for Batch 2 diagrams:
  - highlights/mpmd_async_execution_engine (2 diagrams)
  - highlights/native_agentic_trajectory_training (2 diagrams)
  - highlights/pluggable_agentflow_protocol (3 diagrams)
  - highlights/aio_elastic_agentic_tool_infrastructure (2 diagrams)
  - highlights/why_agentic_rl (2 diagrams)
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
# highlights/mpmd_async_execution_engine.md — Diagram 1
# MPMD Architecture Overview
# ===================================================================


def render_mpmd_1() -> str:
    c = SvgCanvas(
        width=900,
        height=700,
        title="MPMD Async Execution Engine",
        caption="Figure 1: Multi-Program Multiple-Data architecture",
    )

    # --- MainRunner Region (top-left) ---
    r_main = c.region(x=30, y=55, w=180, h=140, title="MainRunner", palette="infra")
    b_config = c.box(x=50, y=95, w=120, h=BOX_HEIGHT, label="Parse Config", palette="infra")
    b_alloc = c.box(x=50, y=135, w=140, h=BOX_HEIGHT, label="Allocate Resources", palette="infra")

    # --- TrainerGroup Region (top-center) ---
    r_train = c.region(x=240, y=55, w=280, h=140, title="TrainerGroup (Training GPUs)", palette="training")
    b_t1 = c.box(x=260, y=95, w=110, h=BOX_HEIGHT, label="Trainer Actor 1", palette="training")
    b_t2 = c.box(x=390, y=95, w=110, h=BOX_HEIGHT, label="Trainer Actor N", palette="training")
    b_critic = c.box(x=260, y=135, w=110, h=BOX_HEIGHT, label="Critic Model", palette="training")

    # --- Monitoring Region (top-right) ---
    r_monitor = c.region(x=550, y=55, w=320, h=90, title="Monitoring", palette="infra")
    b_metric = c.box(x=570, y=95, w=120, h=BOX_HEIGHT, label="MetricWorker", palette="infra")
    b_validate = c.box(x=710, y=95, w=140, h=BOX_HEIGHT, label="ValidateMonitor", palette="infra")

    # --- RolloutManager Region (middle) ---
    r_rollout = c.region(x=30, y=220, w=840, h=200, title="RolloutManager (Rollout GPUs)", palette="rollout")
    b_router = c.box(x=50, y=265, w=100, h=BOX_HEIGHT, label="Router", palette="rollout")
    b_rw1 = c.box(x=180, y=265, w=130, h=BOX_HEIGHT, label="RolloutWorker 1", palette="rollout")
    b_rw2 = c.box(x=330, y=265, w=130, h=BOX_HEIGHT, label="RolloutWorker 2", palette="rollout")
    b_rwn = c.box(x=480, y=265, w=130, h=BOX_HEIGHT, label="RolloutWorker N", palette="rollout")
    b_sg1 = c.box(x=180, y=310, w=130, h=BOX_HEIGHT, label="SGLang Engine", palette="rollout")
    b_sg2 = c.box(x=330, y=310, w=130, h=BOX_HEIGHT, label="SGLang Engine", palette="rollout")
    b_sgn = c.box(x=480, y=310, w=130, h=BOX_HEIGHT, label="SGLang Engine", palette="rollout")
    b_nf1 = c.box(x=180, y=355, w=100, h=BOX_HEIGHT, label="NaiveFlow", palette="rollout")
    b_nf2 = c.box(x=300, y=355, w=100, h=BOX_HEIGHT, label="NaiveFlow", palette="rollout")
    b_nf3 = c.box(x=420, y=355, w=100, h=BOX_HEIGHT, label="NaiveFlow", palette="rollout")

    # --- DataCoordinator Region (middle-right) ---
    r_data = c.region(x=640, y=265, w=220, h=120, title="DataCoordinator", palette="data")
    b_loader = c.box(x=660, y=305, w=100, h=BOX_HEIGHT, label="Dataloader", palette="data")
    b_buffer = c.box(x=660, y=345, w=120, h=BOX_HEIGHT, label="Sample Buffer", palette="data")

    # --- Tool Environment Region (bottom) ---
    r_tool = c.region(x=30, y=450, w=300, h=120, title="Tool Environment", palette="tools")
    b_toolenv = c.box(x=50, y=490, w=100, h=BOX_HEIGHT, label="ToolEnv", palette="tools")
    b_aio = c.box(x=170, y=490, w=120, h=BOX_HEIGHT, label="AIO Proxy", palette="tools")

    # --- Key Arrows ---
    c.arrow(b_config, b_alloc, label="", src_side="bottom", dst_side="top")
    c.arrow(b_alloc, b_t1, label="spawn", src_side="right", dst_side="left")
    c.arrow(b_alloc, b_router, label="spawn", src_side="bottom", dst_side="top")
    c.arrow(b_router, b_rw1, label="", src_side="right", dst_side="left")
    c.arrow(b_router, b_rw2, label="", src_side="right", dst_side="left")
    c.arrow(b_rw1, b_sg1, label="", src_side="bottom", dst_side="top")
    c.arrow(b_rw2, b_sg2, label="", src_side="bottom", dst_side="top")
    c.arrow(b_sg1, b_nf1, label="", src_side="bottom", dst_side="top")
    c.arrow(b_nf1, b_toolenv, label="tool call", src_side="bottom", dst_side="top")
    c.arrow(b_toolenv, b_aio, label="", src_side="right", dst_side="left")
    c.arrow(b_buffer, b_t1, label="batch", src_side="top", dst_side="bottom")
    c.arrow(b_t1, b_router, label="weights", src_side="bottom", dst_side="top")

    return c.render()


# ===================================================================
# highlights/mpmd_async_execution_engine.md — Diagram 2
# MPMD Sequence Diagram
# ===================================================================


def render_mpmd_2() -> str:
    c = SvgCanvas(
        width=900,
        height=580,
        title="MPMD Execution Sequence",
        caption="Figure 2: Async execution flow between components",
    )

    seq = c.sequence(
        actors=[
            {"name": "MainRunner", "short": "MainRunner", "palette": "infra"},
            {"name": "RolloutManager", "short": "RolloutManager", "palette": "rollout"},
            {"name": "SGLang", "short": "SGLang", "palette": "rollout"},
            {"name": "ToolEnv", "short": "ToolEnv", "palette": "tools"},
            {"name": "Trainer", "short": "Trainer", "palette": "training"},
        ],
        x=50,
        y=60,
        actor_spacing=170,
        msg_spacing=38,
    )

    seq.message("MainRunner", "RolloutManager", "dispatch_prompts()")
    seq.message("RolloutManager", "SGLang", "generate(batch)")
    seq.message("SGLang", "ToolEnv", "execute_tool()")
    seq.message("ToolEnv", "SGLang", "tool_response")
    seq.message("SGLang", "RolloutManager", "completions")
    seq.note("RolloutManager", "Compute rewards", side="right")
    seq.message("RolloutManager", "Trainer", "submit_batch()")
    seq.message("Trainer", "Trainer", "forward + backward")
    seq.message("Trainer", "RolloutManager", "sync_weights()")

    return c.render()


# ===================================================================
# highlights/native_agentic_trajectory_training.md — Diagram 1
# Sample State Machine
# ===================================================================


def render_agentic_training_1() -> str:
    c = SvgCanvas(
        width=450,
        height=620,
        title="Sample State Machine",
        caption="Figure 1: States during agentic trajectory generation",
    )

    # Use state diagram
    sd = c.state_diagram()

    # States (vertical layout)
    s_start = sd.start_dot(x=225, y=70, name="start")
    s_pending = sd.state("PENDING", "PENDING", x=125, y=110, w=200, h=36, palette_name="rollout")
    s_gen = sd.state("GENERATING", "GENERATING", x=125, y=180, w=200, h=36, palette_name="rollout")
    s_before = sd.state("BEFORE_ENV", "BEFORE_PROCESSING_ENV", x=100, y=250, w=250, h=36, palette_name="rollout")
    s_proc = sd.state("PROCESSING", "PROCESSING_ENV", x=125, y=320, w=200, h=36, palette_name="tools")
    s_term = sd.state("TERMINATED", "TERMINATED", x=125, y=400, w=200, h=36, palette_name="training")
    s_abort = sd.state("ABORTED", "ABORTED", x=125, y=470, w=200, h=36, palette_name="error")
    s_end = sd.end_dot(x=225, y=550, name="end")

    # Transitions
    sd.transition("start", "PENDING", "Sample created", src_side="bottom", dst_side="top", waypoints=[])
    sd.transition("PENDING", "GENERATING", "apply_chat_template", src_side="bottom", dst_side="top", waypoints=[])
    sd.transition("GENERATING", "BEFORE_ENV", "Tool calls detected", src_side="bottom", dst_side="top", waypoints=[])
    sd.transition("GENERATING", "TERMINATED", "No tool / EOS", src_side="right", dst_side="right", waypoints=[(350, 215), (350, 418)])
    sd.transition("BEFORE_ENV", "PROCESSING", "Validate calls", src_side="bottom", dst_side="top", waypoints=[])
    sd.transition("PROCESSING", "GENERATING", "Execute → append", src_side="left", dst_side="left", waypoints=[(80, 338), (80, 198)])
    sd.transition("PROCESSING", "ABORTED", "Tool error", src_side="bottom", dst_side="top", waypoints=[])
    sd.transition("TERMINATED", "end", "Compute reward", src_side="bottom", dst_side="top", waypoints=[])
    sd.transition("ABORTED", "end", "Log error", src_side="bottom", dst_side="top", waypoints=[])

    return c.render()


# ===================================================================
# highlights/native_agentic_trajectory_training.md — Diagram 2
# Trajectory Training Flow
# ===================================================================


def render_agentic_training_2() -> str:
    c = SvgCanvas(
        width=800,
        height=340,
        title="Trajectory Training Flow",
        caption="Figure 2: How agentic trajectories are trained",
    )

    # Horizontal flow
    r_rollout = c.region(x=30, y=55, w=450, h=180, title="Rollout Phase", palette="rollout")
    b_prompt = c.box(x=50, y=100, w=100, h=BOX_HEIGHT, label="Prompt", palette="data")
    b_gen = c.box(x=180, y=100, w=100, h=BOX_HEIGHT, label="Generate", palette="rollout")
    b_tool = c.box(x=310, y=100, w=100, h=BOX_HEIGHT, label="Tool Call", palette="tools")
    b_traj = c.box(x=180, y=160, w=230, h=BOX_HEIGHT, label="Complete Trajectory", bold=True, palette="rollout")

    r_train = c.region(x=510, y=55, w=260, h=180, title="Training Phase", palette="training")
    b_reward = c.box(x=530, y=100, w=100, h=BOX_HEIGHT, label="Reward", palette="highlight")
    b_loss = c.box(x=650, y=100, w=100, h=BOX_HEIGHT, label="Loss", palette="training")
    b_update = c.box(x=590, y=160, w=100, h=BOX_HEIGHT, label="Update", bold=True, palette="training")

    c.arrow(b_prompt, b_gen, label="", src_side="right", dst_side="left")
    c.arrow(b_gen, b_tool, label="", src_side="right", dst_side="left")
    c.arrow(b_tool, b_gen, label="response", src_side="bottom", dst_side="bottom")
    c.arrow(b_gen, b_traj, label="", src_side="bottom", dst_side="top")
    c.arrow(b_traj, b_reward, label="", src_side="right", dst_side="left")
    c.arrow(b_reward, b_loss, label="", src_side="right", dst_side="left")
    c.arrow(b_loss, b_update, label="", src_side="bottom", dst_side="top")

    return c.render()


# ===================================================================
# highlights/pluggable_agentflow_protocol.md — Diagram 1
# AgentFlow Class Diagram
# ===================================================================


def render_agentflow_1() -> str:
    c = SvgCanvas(
        width=800,
        height=480,
        title="AgentFlow Protocol Classes",
        caption="Figure 1: Core protocol and data classes",
    )

    # AgentFlow protocol (top-left)
    c._elements.append(
        f'<rect x="40" y="60" width="200" height="120" rx="2" fill="{PALETTES["rollout"]["bg"]}" '
        f'stroke="{PALETTES["rollout"]["border"]}" stroke-width="1.2"/>'
    )
    c._elements.append(f'<rect x="40" y="60" width="200" height="26" rx="2" fill="{PALETTES["rollout"]["border"]}" />')
    c._elements.append(
        f'<text x="140" y="78" text-anchor="middle" fill="#fff" font-size="{BOX_TEXT_SIZE}" ' f'font-weight="bold">AgentFlow</text>'
    )
    c._elements.append(f'<text x="50" y="98" fill="#424242" font-size="{LABEL_SIZE}">&lt;&lt;Protocol&gt;&gt;</text>')
    c._elements.append(f'<line x1="40" y1="106" x2="240" y2="106" stroke="{PALETTES["rollout"]["border"]}" stroke-width="0.5"/>')
    c._elements.append(f'<text x="50" y="122" fill="#424242" font-size="{LABEL_SIZE}">preprocess(sample) → Sample</text>')
    c._elements.append(f'<text x="50" y="140" fill="#424242" font-size="{LABEL_SIZE}">generate(sample) → async</text>')
    c._elements.append(f'<text x="50" y="158" fill="#424242" font-size="{LABEL_SIZE}">reward(sample) → async</text>')

    # Sample dataclass (center)
    c._elements.append(
        f'<rect x="290" y="60" width="200" height="180" rx="2" fill="{PALETTES["data"]["bg"]}" '
        f'stroke="{PALETTES["data"]["border"]}" stroke-width="1.2"/>'
    )
    c._elements.append(f'<rect x="290" y="60" width="200" height="26" rx="2" fill="{PALETTES["data"]["border"]}" />')
    c._elements.append(
        f'<text x="390" y="78" text-anchor="middle" fill="#fff" font-size="{BOX_TEXT_SIZE}" ' f'font-weight="bold">Sample</text>'
    )
    c._elements.append(f'<text x="300" y="98" fill="#424242" font-size="{LABEL_SIZE}">&lt;&lt;dataclass&gt;&gt;</text>')
    for i, attr in enumerate(
        [
            "tokens: list[int]",
            "loss_mask: list[int]",
            "rollout_log_probs: list[float]",
            "conversations: list[dict]",
            "reward: float | None",
            "model: Model",
            "status: Status",
        ]
    ):
        c._elements.append(f'<text x="300" y="{116 + i*18}" fill="#424242" font-size="{LABEL_SIZE}">{attr}</text>')

    # Model protocol (top-right)
    c._elements.append(
        f'<rect x="540" y="60" width="220" height="100" rx="2" fill="{PALETTES["rollout"]["bg"]}" '
        f'stroke="{PALETTES["rollout"]["border"]}" stroke-width="1.2"/>'
    )
    c._elements.append(f'<rect x="540" y="60" width="220" height="26" rx="2" fill="{PALETTES["rollout"]["border"]}" />')
    c._elements.append(
        f'<text x="650" y="78" text-anchor="middle" fill="#fff" font-size="{BOX_TEXT_SIZE}" ' f'font-weight="bold">Model</text>'
    )
    c._elements.append(f'<text x="550" y="98" fill="#424242" font-size="{LABEL_SIZE}">&lt;&lt;Protocol&gt;&gt;</text>')
    c._elements.append(f'<line x1="540" y1="106" x2="760" y2="106" stroke="{PALETTES["rollout"]["border"]}" stroke-width="0.5"/>')
    c._elements.append(f'<text x="550" y="122" fill="#424242" font-size="{LABEL_SIZE}">generate(convs, params) → Response</text>')
    c._elements.append(f'<text x="550" y="140" fill="#424242" font-size="{LABEL_SIZE}">tokenize(text) → list[int]</text>')

    # ModelResponse (bottom-left)
    c._elements.append(
        f'<rect x="40" y="260" width="200" height="100" rx="2" fill="{PALETTES["data"]["bg"]}" '
        f'stroke="{PALETTES["data"]["border"]}" stroke-width="1.2"/>'
    )
    c._elements.append(f'<rect x="40" y="260" width="200" height="26" rx="2" fill="{PALETTES["data"]["border"]}" />')
    c._elements.append(
        f'<text x="140" y="278" text-anchor="middle" fill="#fff" font-size="{BOX_TEXT_SIZE}" ' f'font-weight="bold">ModelResponse</text>'
    )
    c._elements.append(f'<text x="50" y="298" fill="#424242" font-size="{LABEL_SIZE}">&lt;&lt;dataclass&gt;&gt;</text>')
    c._elements.append(f'<text x="50" y="316" fill="#424242" font-size="{LABEL_SIZE}">text: str</text>')
    c._elements.append(f'<text x="50" y="334" fill="#424242" font-size="{LABEL_SIZE}">tokens: list[int]</text>')
    c._elements.append(f'<text x="50" y="352" fill="#424242" font-size="{LABEL_SIZE}">log_probs: list[float]</text>')

    # Arrows
    c._elements.append(f'<line x1="240" y1="120" x2="290" y2="120" stroke="#78909C" stroke-width="1.2" marker-end="url(#arr)"/>')
    c._elements.append(f'<text x="265" y="112" text-anchor="middle" fill="#607D8B" font-size="{LABEL_SIZE}">creates</text>')
    c._elements.append(f'<line x1="490" y1="150" x2="540" y2="110" stroke="#78909C" stroke-width="1.2" marker-end="url(#arr)"/>')
    c._elements.append(f'<text x="515" y="122" text-anchor="middle" fill="#607D8B" font-size="{LABEL_SIZE}">uses</text>')
    c._elements.append(f'<line x1="650" y1="160" x2="140" y2="260" stroke="#78909C" stroke-width="1.2" marker-end="url(#arr)"/>')
    c._elements.append(f'<text x="395" y="202" text-anchor="middle" fill="#607D8B" font-size="{LABEL_SIZE}">returns</text>')

    return c.render()


# ===================================================================
# highlights/pluggable_agentflow_protocol.md — Diagram 2
# NaiveFlow Implementation
# ===================================================================


def render_agentflow_2() -> str:
    c = SvgCanvas(
        width=800,
        height=340,
        title="NaiveFlow Implementation",
        caption="Figure 2: Default AgentFlow implementation",
    )

    # NaiveFlow region
    r_naive = c.region(x=30, y=55, w=740, h=180, title="NaiveFlow (implements AgentFlow)", palette="rollout")

    b_pre = c.box(x=50, y=100, w=130, h=BOX_HEIGHT, label="preprocess()", palette="rollout")
    b_gen = c.box(x=210, y=100, w=130, h=BOX_HEIGHT, label="generate()", palette="rollout")
    b_tool = c.box(x=370, y=100, w=130, h=BOX_HEIGHT, label="execute_tools()", palette="tools")
    b_reward = c.box(x=530, y=100, w=130, h=BOX_HEIGHT, label="reward()", palette="highlight")

    b_loop = c.box(x=290, y=170, w=200, h=BOX_HEIGHT, label="Multi-turn Loop", bold=True, palette="rollout")

    c.arrow(b_pre, b_gen, label="", src_side="right", dst_side="left")
    c.arrow(b_gen, b_tool, label="if tool_calls", src_side="right", dst_side="left")
    c.arrow(b_tool, b_gen, label="append response", src_side="bottom", dst_side="bottom")
    c.arrow(b_gen, b_reward, label="if done", src_side="right", dst_side="left")
    c.arrow(b_gen, b_loop, label="", src_side="bottom", dst_side="top", dashed=True)
    c.arrow(b_tool, b_loop, label="", src_side="bottom", dst_side="top", dashed=True)

    return c.render()


# ===================================================================
# highlights/pluggable_agentflow_protocol.md — Diagram 3
# Custom Flow Extension
# ===================================================================


def render_agentflow_3() -> str:
    c = SvgCanvas(
        width=700,
        height=300,
        title="Custom AgentFlow Extension",
        caption="Figure 3: How to extend AgentFlow for custom logic",
    )

    # Base protocol
    b_proto = c.box(x=260, y=80, w=180, h=40, label="AgentFlow Protocol", bold=True, palette="rollout")

    # Implementations
    b_naive = c.box(x=80, y=180, w=140, h=40, label="NaiveFlow", palette="rollout")
    b_custom = c.box(x=280, y=180, w=140, h=40, label="CustomFlow", palette="highlight")
    b_eval = c.box(x=480, y=180, w=140, h=40, label="EvalFlow", palette="data")

    c.arrow(b_proto, b_naive, label="implements", src_side="bottom", dst_side="top")
    c.arrow(b_proto, b_custom, label="implements", src_side="bottom", dst_side="top")
    c.arrow(b_proto, b_eval, label="implements", src_side="bottom", dst_side="top")

    return c.render()


# ===================================================================
# highlights/aio_elastic_agentic_tool_infrastructure.md — Diagram 1
# AIO Architecture
# ===================================================================


def render_aio_1() -> str:
    c = SvgCanvas(
        width=900,
        height=520,
        title="AIO Elastic Tool Infrastructure",
        caption="Figure 1: All-in-One tool environment architecture",
    )

    # Rollout Layer (top)
    r_rollout = c.region(x=30, y=55, w=400, h=150, title="Rollout Workers", palette="rollout")
    b_rw1 = c.box(x=50, y=100, w=110, h=BOX_HEIGHT, label="Worker 1", palette="rollout")
    b_rw2 = c.box(x=180, y=100, w=110, h=BOX_HEIGHT, label="Worker 2", palette="rollout")
    b_rwn = c.box(x=50, y=150, w=110, h=BOX_HEIGHT, label="Worker N", palette="rollout")

    # AIO Proxy (center)
    r_aio = c.region(x=460, y=55, w=200, h=150, title="AIO Proxy", palette="highlight")
    b_router = c.box(x=480, y=100, w=160, h=BOX_HEIGHT, label="Request Router", bold=True, palette="highlight")
    b_queue = c.box(x=480, y=150, w=160, h=BOX_HEIGHT, label="Task Queue", palette="highlight")

    # Tool Environments (bottom)
    r_tools = c.region(x=30, y=250, w=840, h=160, title="Tool Environment Pool (Auto-scaling)", palette="tools")
    b_env1 = c.box(x=50, y=295, w=160, h=BOX_HEIGHT, label="ToolEnv Container 1", palette="tools")
    b_env2 = c.box(x=240, y=295, w=160, h=BOX_HEIGHT, label="ToolEnv Container 2", palette="tools")
    b_envn = c.box(x=430, y=295, w=160, h=BOX_HEIGHT, label="ToolEnv Container N", palette="tools")
    b_sandbox = c.box(x=50, y=355, w=540, h=BOX_HEIGHT, label="Sandboxed Execution (Docker / Process Isolation)", palette="infra")

    # External Services (right)
    r_ext = c.region(x=690, y=55, w=180, h=150, title="External APIs", palette="data")
    b_api = c.box(x=710, y=100, w=140, h=BOX_HEIGHT, label="HTTP Services", palette="data")
    b_db = c.box(x=710, y=150, w=140, h=BOX_HEIGHT, label="Databases", palette="data")

    c.arrow(b_rw1, b_router, label="tool_call", src_side="right", dst_side="left")
    c.arrow(b_rw2, b_router, label="", src_side="right", dst_side="left")
    c.arrow(b_router, b_queue, label="", src_side="bottom", dst_side="top")
    c.arrow(b_queue, b_env1, label="dispatch", src_side="bottom", dst_side="top")
    c.arrow(b_queue, b_env2, label="", src_side="bottom", dst_side="top")
    c.arrow(b_env1, b_sandbox, label="", src_side="bottom", dst_side="top")
    c.arrow(b_router, b_api, label="external", src_side="right", dst_side="left")

    return c.render()


# ===================================================================
# highlights/aio_elastic_agentic_tool_infrastructure.md — Diagram 2
# Tool Execution Flow
# ===================================================================


def render_aio_2() -> str:
    c = SvgCanvas(
        width=800,
        height=300,
        title="Tool Execution Flow",
        caption="Figure 2: How tool calls are processed",
    )

    # Linear flow
    b_call = c.box(x=40, y=120, w=120, h=BOX_HEIGHT, label="Tool Call", bold=True, palette="rollout")
    b_parse = c.box(x=190, y=120, w=100, h=BOX_HEIGHT, label="Parse JSON", palette="infra")
    b_route = c.box(x=320, y=120, w=100, h=BOX_HEIGHT, label="Route", palette="highlight")
    b_exec = c.box(x=450, y=120, w=100, h=BOX_HEIGHT, label="Execute", palette="tools")
    b_resp = c.box(x=580, y=120, w=120, h=BOX_HEIGHT, label="Response", bold=True, palette="data")

    c.arrow(b_call, b_parse, label="", src_side="right", dst_side="left")
    c.arrow(b_parse, b_route, label="", src_side="right", dst_side="left")
    c.arrow(b_route, b_exec, label="", src_side="right", dst_side="left")
    c.arrow(b_exec, b_resp, label="", src_side="right", dst_side="left")

    return c.render()


# ===================================================================
# highlights/why_agentic_rl.md — Diagram 1
# Traditional RL vs Agentic RL
# ===================================================================


def render_why_rl_1() -> str:
    c = SvgCanvas(
        width=800,
        height=340,
        title="Traditional vs Agentic RL",
        caption="Figure 1: Comparison of training approaches",
    )

    # Traditional (left)
    r_trad = c.region(x=30, y=55, w=340, h=180, title="Traditional RL", palette="infra")
    b_t_prompt = c.box(x=50, y=100, w=100, h=BOX_HEIGHT, label="Prompt", palette="data")
    b_t_gen = c.box(x=180, y=100, w=100, h=BOX_HEIGHT, label="Generate", palette="rollout")
    b_t_reward = c.box(x=115, y=160, w=100, h=BOX_HEIGHT, label="Reward", palette="highlight")

    # Agentic (right)
    r_agent = c.region(x=430, y=55, w=340, h=180, title="Agentic RL", palette="training")
    b_a_prompt = c.box(x=450, y=100, w=90, h=BOX_HEIGHT, label="Prompt", palette="data")
    b_a_gen = c.box(x=560, y=100, w=90, h=BOX_HEIGHT, label="Generate", palette="rollout")
    b_a_tool = c.box(x=670, y=100, w=80, h=BOX_HEIGHT, label="Tool", palette="tools")
    b_a_reward = c.box(x=560, y=160, w=90, h=BOX_HEIGHT, label="Reward", palette="highlight")

    c.arrow(b_t_prompt, b_t_gen, label="", src_side="right", dst_side="left")
    c.arrow(b_t_gen, b_t_reward, label="", src_side="bottom", dst_side="top")

    c.arrow(b_a_prompt, b_a_gen, label="", src_side="right", dst_side="left")
    c.arrow(b_a_gen, b_a_tool, label="", src_side="right", dst_side="left")
    c.arrow(b_a_tool, b_a_gen, label="loop", src_side="bottom", dst_side="bottom")
    c.arrow(b_a_gen, b_a_reward, label="", src_side="bottom", dst_side="top")

    return c.render()


# ===================================================================
# highlights/why_agentic_rl.md — Diagram 2
# Benefits of Agentic RL
# ===================================================================


def render_why_rl_2() -> str:
    c = SvgCanvas(
        width=800,
        height=400,
        title="Benefits of Agentic RL",
        caption="Figure 2: Why agentic training improves model capabilities",
    )

    # Center: Agentic Training
    b_center = c.box(x=300, y=80, w=200, h=40, label="Agentic Training", bold=True, palette="training")

    # Benefits (fan out)
    r_tools = c.region(x=40, y=180, w=180, h=100, title="Tool Use", palette="tools")
    b_tool = c.box(x=60, y=220, w=140, h=BOX_HEIGHT, label="Learn tool APIs", palette="tools")

    r_reason = c.region(x=250, y=180, w=180, h=100, title="Reasoning", palette="rollout")
    b_reason = c.box(x=270, y=220, w=140, h=BOX_HEIGHT, label="Multi-step logic", palette="rollout")

    r_ground = c.region(x=460, y=180, w=180, h=100, title="Grounding", palette="data")
    b_ground = c.box(x=480, y=220, w=140, h=BOX_HEIGHT, label="Real-world data", palette="data")

    r_robust = c.region(x=670, y=180, w=100, h=100, title="Robustness", palette="highlight")
    b_robust = c.box(x=690, y=220, w=60, h=BOX_HEIGHT, label="Errors", palette="highlight")

    c.arrow(b_center, b_tool, label="", src_side="bottom", dst_side="top")
    c.arrow(b_center, b_reason, label="", src_side="bottom", dst_side="top")
    c.arrow(b_center, b_ground, label="", src_side="bottom", dst_side="top")
    c.arrow(b_center, b_robust, label="", src_side="bottom", dst_side="top")

    return c.render()


# ===================================================================
# Registry: map output filenames to render functions
# ===================================================================

REGISTRY = {
    "highlightsmpmd_async_execution_engine_1.svg": render_mpmd_1,
    "highlightsmpmd_async_execution_engine_2.svg": render_mpmd_2,
    "highlightsnative_agentic_trajectory_training_1.svg": render_agentic_training_1,
    "highlightsnative_agentic_trajectory_training_2.svg": render_agentic_training_2,
    "highlightspluggable_agentflow_protocol_1.svg": render_agentflow_1,
    "highlightspluggable_agentflow_protocol_2.svg": render_agentflow_2,
    "highlightspluggable_agentflow_protocol_3.svg": render_agentflow_3,
    "highlightsaio_elastic_agentic_tool_infrastructure_1.svg": render_aio_1,
    "highlightsaio_elastic_agentic_tool_infrastructure_2.svg": render_aio_2,
    "highlightswhy_agentic_rl_1.svg": render_why_rl_1,
    "highlightswhy_agentic_rl_2.svg": render_why_rl_2,
}


if __name__ == "__main__":
    from pathlib import Path

    out_dir = Path(__file__).parent / "docs/assets/images/diagrams"
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname, func in REGISTRY.items():
        svg = func()
        (out_dir / fname).write_text(svg)
        print(f"Generated {fname}")
