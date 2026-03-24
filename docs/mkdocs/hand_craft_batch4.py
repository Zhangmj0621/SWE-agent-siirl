#!/usr/bin/env python3
"""Hand-crafted SVG rendering functions for Batch 4 diagrams:
  - developer_guide/code_structure (5 diagrams)
  - developer_guide/adding_new_executor_or_flow (2 diagrams)
  - developer_guide/contributing (1 diagram)
  - advanced/failure_propagation (3 diagrams)
  - advanced/performance_tuning (2 diagrams)
  - faq/troubleshooting (1 diagram)
  - reference/module_map (1 diagram)
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
# developer_guide/code_structure  — Diagram 1
# Training step call graph (top-down)
# ===================================================================


def render_code_structure_1() -> str:
    c = SvgCanvas(
        width=860,
        height=680,
        title="Training Step Call Graph",
        caption="Figure 1: MainRunner → TrainerGroup → forward / advantage / loss / backward",
    )

    # --- Region: MainRunner (top center) ---
    r_main = c.region(x=300, y=52, w=260, h=68, title="MainRunner", palette="infra")
    b_run = r_main.box("MainRunner.run()", bold=True)

    # --- Region: Training Step (large middle) ---
    r_train = c.region(x=50, y=150, w=760, h=340, title="Training Step", palette="training")
    # Row 1: TrainerGroup
    b_tg = c.box(x=330, y=195, w=200, h=BOX_HEIGHT, label="TrainerGroup.train()", bold=True, palette="training")
    # Row 2: DataCoordinator, Actor, Critic
    b_dc = c.box(x=68, y=260, w=210, h=BOX_HEIGHT, label="DataCoordinator.get()", palette="data")
    b_actor = c.box(x=325, y=260, w=200, h=BOX_HEIGHT, label="Actor.forward()", palette="training")
    b_critic = c.box(x=575, y=260, w=210, h=BOX_HEIGHT, label="Critic.forward()", palette="rollout")
    # Row 3: compute_advantage
    b_adv = c.box(x=325, y=325, w=200, h=BOX_HEIGHT, label="compute_advantage()", palette="training")
    # Row 4: compute_loss
    b_loss = c.box(x=325, y=390, w=200, h=BOX_HEIGHT, label="compute_loss()", palette="training")
    # Row 5: backward
    b_bw = c.box(x=325, y=455, w=200, h=BOX_HEIGHT, label="Actor.backward()", palette="training")

    # --- Region: Reporting (bottom) ---
    r_rep = c.region(x=300, y=530, w=260, h=68, title="Reporting", palette="highlight")
    b_metric = r_rep.box("MetricWorker.report()")

    # --- Arrows ---
    c.arrow(b_run, b_tg, label="launch training", src_side="bottom", dst_side="top")
    c.arrow(b_tg, b_dc, label="fetch batch", src_side="bottom", dst_side="top")
    c.arrow(b_tg, b_actor, label="model inference", src_side="bottom", dst_side="top")
    c.arrow(b_tg, b_critic, label="value estimation (PPO)", src_side="bottom", dst_side="top")
    c.arrow(b_actor, b_adv, label="", src_side="bottom", dst_side="top")
    c.arrow(b_critic, b_adv, label="", src_side="bottom", dst_side="top")
    c.arrow(b_adv, b_loss, label="GAE / GRPO", src_side="bottom", dst_side="top")
    c.arrow(b_loss, b_bw, label="dual-clip PPO loss", src_side="bottom", dst_side="top")
    c.arrow(b_bw, b_metric, label="gradient update", src_side="bottom", dst_side="top")

    return c.render()


# ===================================================================
# developer_guide/code_structure  — Diagram 2
# Rollout / NaiveFlow state machine (top-down)
# ===================================================================


def render_code_structure_2() -> str:
    c = SvgCanvas(
        width=880,
        height=720,
        title="Rollout & NaiveFlow Execution Path",
        caption="Figure 2: RolloutManager → AgentExecutor → NaiveFlow state machine",
    )

    # --- Region: RolloutManager ---
    r_rm = c.region(x=50, y=52, w=260, h=68, title="RolloutManager", palette="rollout")
    b_rm = r_rm.box("RolloutManager.rollout()", bold=True)

    # --- Region: AgentExecutor ---
    r_ae = c.region(x=370, y=52, w=260, h=68, title="AgentExecutor", palette="tools")
    b_ae = r_ae.box("AgentExecutor.execute()", bold=True)

    # --- Region: NaiveFlow state machine ---
    r_nf = c.region(x=50, y=155, w=780, h=480, title="NaiveFlow State Machine", palette="training")

    b_pend = c.box(x=330, y=200, w=240, h=BOX_HEIGHT, label="_handle_pending_state()", palette="training")
    b_gen = c.box(x=330, y=265, w=240, h=BOX_HEIGHT, label="_handle_generating_state()", palette="training")
    b_parse = c.box(x=330, y=330, w=260, h=BOX_HEIGHT, label="ToolParser.extract_tool_calls()", palette="tools")
    b_proc = c.box(x=330, y=395, w=260, h=BOX_HEIGHT, label="_handle_processing_envs_state()", palette="training")
    b_tool = c.box(x=330, y=460, w=260, h=BOX_HEIGHT, label="ToolEnv.create() + step() + release()", palette="tools")

    # Diamond: More turns?
    d_more = c.diamond(x=392, y=520, size=50, label="More?", palette="highlight")

    # Compute reward
    b_reward = c.box(x=530, y=530, w=170, h=BOX_HEIGHT, label="Compute reward", palette="data")

    # --- Arrows ---
    c.arrow(b_rm, b_ae, label="dispatch", src_side="right", dst_side="left")
    c.arrow(b_ae, b_pend, label="invoke flow", src_side="bottom", dst_side="top")
    c.arrow(b_pend, b_gen, label="tokenize prompt", src_side="bottom", dst_side="top")
    c.arrow(b_gen, b_parse, label="SGLang inference", src_side="bottom", dst_side="top")
    c.arrow(b_parse, b_proc, label="tool calls found", src_side="bottom", dst_side="top")
    c.arrow(b_proc, b_tool, label="execute tools", src_side="bottom", dst_side="top")
    c.arrow(b_tool, d_more, label="", src_side="bottom", dst_side="top")
    # Yes loop back to _handle_generating_state
    c.arrow(d_more, b_gen, label="Yes", src_side="left", dst_side="left", waypoints=[(200, 545), (200, 280)])
    # No -> Compute reward
    c.arrow(d_more, b_reward, label="No: TERMINATED", src_side="right", dst_side="left")

    return c.render()


# ===================================================================
# developer_guide/code_structure  — Diagram 3
# AgentFlow loading & three-stage protocol (top-down)
# ===================================================================


def render_code_structure_3() -> str:
    c = SvgCanvas(
        width=860,
        height=580,
        title="AgentFlow Loading & Three-Stage Protocol",
        caption="Figure 3: Dynamic import → instantiation → method injection → 3-stage protocol",
    )

    # --- Region: Loading ---
    r_load = c.region(x=50, y=52, w=760, h=200, title="AgentFlow Loading", palette="rollout")
    b_load = c.box(x=70, y=100, w=250, h=BOX_HEIGHT, label="load_agentflow(config, model)", bold=True, palette="rollout")
    b_imp = c.box(x=360, y=100, w=200, h=BOX_HEIGHT, label="Import AgentFlow class", palette="rollout")
    b_inst = c.box(x=600, y=100, w=190, h=BOX_HEIGHT, label="Instantiate agent", palette="rollout")
    b_inject = c.box(x=250, y=170, w=360, h=BOX_HEIGHT, label="Inject preprocess_fn, generate_fn, reward_fn", palette="rollout")

    # --- Region: Three-Stage Protocol ---
    r_proto = c.region(x=50, y=290, w=760, h=200, title="Three-Stage Protocol", palette="training")
    b_pre = c.box(x=80, y=340, w=210, h=BOX_HEIGHT, label="agent.preprocess(sample)", bold=True, palette="training")
    b_gen = c.box(x=330, y=340, w=210, h=BOX_HEIGHT, label="agent.generate(sample)", bold=True, palette="training")
    b_rew = c.box(x=580, y=340, w=210, h=BOX_HEIGHT, label="agent.reward(sample)", bold=True, palette="training")

    # Step numbers
    c.step_number(140, 375, 1)
    c.step_number(390, 375, 2)
    c.step_number(640, 375, 3)

    # Notes under each stage
    c.text(185, 410, "returns Sample", size=10, color="#607D8B", style="italic")
    c.text(435, 410, "async inference + tool interaction", size=10, color="#607D8B", style="italic")
    c.text(685, 410, "scalar reward", size=10, color="#607D8B", style="italic")

    # --- Arrows: Loading chain ---
    c.arrow(b_load, b_imp, label="dynamic import", src_side="right", dst_side="left")
    c.arrow(b_imp, b_inst, label="AgentFlowClass()", src_side="right", dst_side="left")
    c.arrow(b_inst, b_inject, label="MethodType injection", src_side="bottom", dst_side="right", waypoints=[(695, 185)])
    # Loading -> Protocol
    c.arrow(b_inject, b_pre, label="", src_side="bottom", dst_side="top", waypoints=[(430, 270), (185, 270)])
    # Protocol chain
    c.arrow(b_pre, b_gen, label="returns Sample", src_side="right", dst_side="left")
    c.arrow(b_gen, b_rew, label="async inference + tool", src_side="right", dst_side="left")

    return c.render()


# ===================================================================
# developer_guide/code_structure  — Diagram 4
# Config parsing (left-to-right)
# ===================================================================


def render_code_structure_4() -> str:
    c = SvgCanvas(
        width=900,
        height=420,
        title="Configuration Parsing Pipeline",
        caption="Figure 4: CLI args → parser.py → OmegaConf → SiiRLArguments dataclass tree",
    )

    # --- Region: CLI Input ---
    r_cli = c.region(x=40, y=52, w=200, h=80, title="CLI Input", palette="infra")
    b_cli = r_cli.box("CLI args (dot-notation)", bold=True)

    # --- Region: Parsing ---
    r_parse = c.region(x=290, y=52, w=260, h=120, title="Config Parsing", palette="rollout")
    b_parser = r_parse.box("parser.py", bold=True)
    b_omega = r_parse.box("argparse + OmegaConf.from_cli()")

    # --- Region: Output ---
    r_out = c.region(x=600, y=52, w=270, h=320, title="SiiRLArguments", palette="training")
    b_data = r_out.box(".data: DataArguments")
    b_actor = r_out.box(".actor_ref: ActorRefArguments")
    b_rollout = r_out.box(".rollout: RolloutArguments")
    b_critic = r_out.box(".critic: CriticArguments")
    b_trainer = r_out.box(".trainer: TrainingArguments")

    # --- Arrows ---
    c.arrow(b_cli, b_parser, label="key=value pairs", src_side="right", dst_side="left")
    c.arrow(b_parser, b_omega, label="parse & merge", src_side="bottom", dst_side="top")
    # Fan-out from omega to each arg
    c.arrow(b_omega, b_data, label="recursive dataclass", src_side="right", dst_side="left")
    c.arrow(b_omega, b_actor, label="", src_side="right", dst_side="left", waypoints=[(580, 148)])
    c.arrow(b_omega, b_rollout, label="", src_side="right", dst_side="left", waypoints=[(580, 188)])
    c.arrow(b_omega, b_critic, label="", src_side="right", dst_side="left", waypoints=[(580, 228)])
    c.arrow(b_omega, b_trainer, label="", src_side="right", dst_side="left", waypoints=[(580, 268)])

    return c.render()


# ===================================================================
# developer_guide/code_structure  — Diagram 5
# Extension points (top-bottom)
# ===================================================================


def render_code_structure_5() -> str:
    c = SvgCanvas(
        width=860,
        height=500,
        title="Extension Points Architecture",
        caption="Figure 5: Core components and their pluggable extension points",
    )

    # --- Region: Core ---
    r_core = c.region(x=230, y=52, w=400, h=130, title="siirl-agentic Core", palette="infra")
    b_rm = c.box(x=250, y=100, w=165, h=BOX_HEIGHT, label="RolloutManager", bold=True, palette="rollout")
    b_tg = c.box(x=445, y=100, w=165, h=BOX_HEIGHT, label="TrainerGroup", bold=True, palette="training")
    b_dc = c.box(x=345, y=140, w=170, h=BOX_HEIGHT, label="DataCoordinator", bold=True, palette="data")

    # --- Region: Extension Points ---
    r_ext = c.region(x=50, y=230, w=760, h=200, title="Extension Points", palette="highlight")
    b_flow = c.box(x=70, y=280, w=200, h=38, label="New AgentFlow", bold=True, palette="highlight")
    c.text(170, 330, "agentflow/my_flow.py", size=10, color="#607D8B", style="italic")

    b_tool = c.box(x=290, y=280, w=200, h=38, label="New ToolEnv", bold=True, palette="highlight")
    c.text(390, 330, "tool_env/my_tool.py", size=10, color="#607D8B", style="italic")

    b_rewfn = c.box(x=510, y=280, w=190, h=38, label="New Reward Fn", bold=True, palette="highlight")
    c.text(605, 330, "reward_score/my_reward.py", size=10, color="#607D8B", style="italic")

    b_tp = c.box(x=70, y=360, w=200, h=38, label="New ToolParser", bold=True, palette="highlight")
    c.text(170, 410, "@ToolParser.register", size=10, color="#607D8B", style="italic")

    b_algo = c.box(x=510, y=360, w=190, h=38, label="New Algorithm", bold=True, palette="highlight")
    c.text(605, 410, "algorithm/my_algo.py", size=10, color="#607D8B", style="italic")

    # --- Arrows: plugs into ---
    c.arrow(b_flow, b_rm, label="plugs into", src_side="top", dst_side="bottom", waypoints=[(170, 250), (332, 250)])
    c.arrow(b_tool, b_rm, label="plugs into", src_side="top", dst_side="bottom", waypoints=[(390, 250), (332, 250)])
    c.arrow(b_rewfn, b_rm, label="plugs into", src_side="top", dst_side="bottom", waypoints=[(605, 250), (332, 250)])
    c.arrow(b_tp, b_rm, label="plugs into", src_side="top", dst_side="left", waypoints=[(170, 250), (232, 250), (232, 115)])
    c.arrow(b_algo, b_tg, label="plugs into", src_side="top", dst_side="bottom", waypoints=[(605, 250), (527, 250)])

    return c.render()


# ===================================================================
# developer_guide/adding_new_executor_or_flow  — Diagram 1
# Class diagram (AgentFlow, Sample, Model, Status)
# ===================================================================


def _class_box(
    canvas: SvgCanvas, x: float, y: float, w: float, title: str, stereotype: str, members: list[str], palette_name: str = "default"
) -> "BoxInfo":
    """Draw a UML-style class box with header and members."""
    from svg_renderer import PALETTES, BoxInfo, Rect, _esc

    pal = PALETTES.get(palette_name, PALETTES["default"])
    header_h = 38 if stereotype else 28
    line_h = 18
    body_h = max(len(members) * line_h + 10, 30)
    total_h = header_h + body_h

    # Header bg
    canvas._elements.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{total_h}" '
        f'rx="2" fill="{pal["box_bg"]}" stroke="{pal["border"]}" stroke-width="1.2"/>'
    )
    canvas._elements.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{header_h}" ' f'rx="2" fill="{pal["bg"]}" stroke="{pal["border"]}" stroke-width="1.2"/>'
    )
    # Stereotype
    ty = y + 14
    if stereotype:
        canvas._elements.append(
            f'<text x="{x + w / 2}" y="{ty}" text-anchor="middle" '
            f'fill="{pal["title_color"]}" font-size="9" font-style="italic">'
            f"{_esc(stereotype)}</text>"
        )
        ty += 16
    # Title
    canvas._elements.append(
        f'<text x="{x + w / 2}" y="{ty}" text-anchor="middle" '
        f'fill="{pal["title_color"]}" font-size="{BOX_TEXT_SIZE}" font-weight="bold">'
        f"{_esc(title)}</text>"
    )
    # Divider
    div_y = y + header_h
    canvas._elements.append(f'<line x1="{x}" y1="{div_y}" x2="{x + w}" y2="{div_y}" ' f'stroke="{pal["border"]}" stroke-width="0.8"/>')
    # Members
    for i, m in enumerate(members):
        canvas._elements.append(
            f'<text x="{x + 8}" y="{div_y + 16 + i * line_h}" text-anchor="start" '
            f'fill="{pal["box_text"]}" font-size="10">{_esc(m)}</text>'
        )

    return BoxInfo(rect=Rect(x, y, w, total_h), label=title)


def render_adding_new_executor_1() -> str:
    c = SvgCanvas(
        width=900,
        height=620,
        title="AgentFlow Class Diagram",
        caption="Figure 1: Core class relationships — AgentFlow, Sample, Model, Status",
    )

    # AgentFlow class (top-left)
    b_af = _class_box(
        c,
        x=50,
        y=60,
        w=260,
        title="AgentFlow",
        stereotype="<<Protocol>>",
        members=[
            "+preprocess(sample: dict) → Sample",
            "+generate(sample: Sample)*",
            "+reward(sample: Sample)*",
        ],
        palette_name="rollout",
    )

    # Sample class (top-right)
    b_sample = _class_box(
        c,
        x=380,
        y=60,
        w=280,
        title="Sample<AgentMeta>",
        stereotype="",
        members=[
            "+m: AgentMeta",
            "+model: Model",
            "+status: Status",
            "+tokens: list[int]",
            "+loss_mask: list[int]",
            "+rollout_log_probs: list[float]",
            "+conversations: list[dict]",
            "+reward: float | None",
            "+append_input_tokens(tokens)",
            "+add_message(role, content)",
        ],
        palette_name="training",
    )

    # Model class (bottom-left)
    b_model = _class_box(
        c,
        x=50,
        y=350,
        w=260,
        title="Model",
        stereotype="",
        members=[
            "+tokenizer: Tokenizer",
            "+query(input_tokens, messages)",
            "  → ModelResponse",
        ],
        palette_name="data",
    )

    # Status enum (bottom-right)
    b_status = _class_box(
        c,
        x=550,
        y=350,
        w=200,
        title="Status",
        stereotype="<<enumeration>>",
        members=[
            "PENDING",
            "ROLLEDOUT",
            "COMPLETED",
            "TRUNCATED",
            "ABORTED",
            "FAILED",
        ],
        palette_name="tools",
    )

    # Relationship arrows
    c.arrow(b_af, b_sample, label="produces & consumes", src_side="right", dst_side="left")
    c.arrow(b_af, b_model, label="uses for inference", src_side="bottom", dst_side="top")
    c.arrow(b_sample, b_status, label="tracks lifecycle", src_side="bottom", dst_side="top")
    c.arrow(b_sample, b_model, label="holds reference", src_side="bottom", dst_side="top", waypoints=[(520, 340), (280, 340)])

    return c.render()


# ===================================================================
# developer_guide/adding_new_executor_or_flow  — Diagram 2
# 3-stage flow (left-to-right with loop in Stage 2)
# ===================================================================


def render_adding_new_executor_2() -> str:
    c = SvgCanvas(
        width=900,
        height=520,
        title="Three-Stage AgentFlow Protocol",
        caption="Figure 2: Preprocess → Generate (with tool loop) → Reward",
    )

    # --- Stage 1: Preprocess ---
    r_s1 = c.region(x=30, y=52, w=200, h=200, title="Stage 1: Preprocess", palette="rollout")
    b_raw = r_s1.box("Raw dict from dataset")
    b_create = r_s1.box("Create Sample + metadata")
    b_conv = r_s1.box("Build initial conversation")

    # --- Stage 2: Generate ---
    r_s2 = c.region(x=260, y=52, w=320, h=320, title="Stage 2: Generate", palette="training")
    b_tok = c.box(x=280, y=100, w=280, h=BOX_HEIGHT, label="Tokenize conversation", palette="training")
    b_query = c.box(x=280, y=150, w=280, h=BOX_HEIGHT, label="model.query() — LLM inference", palette="training")

    d_tool = c.diamond(x=380, y=200, size=48, label="Tool?", palette="highlight")

    b_exec = c.box(x=280, y=270, w=280, h=BOX_HEIGHT, label="Execute tool via environment", palette="tools")
    b_append = c.box(x=280, y=316, w=280, h=BOX_HEIGHT, label="Append tool response", palette="tools")
    b_rec = c.box(x=280, y=200, w=80, h=BOX_HEIGHT, label="Record", palette="training")

    # Place "Record output tokens + log_probs" to the right of diamond
    # Actually let me adjust - diamond right goes to Record box
    # Let me re-layout Stage 2 more carefully
    # Tokenize -> query -> diamond
    # diamond Yes -> exec tool -> append -> loop back to query
    # diamond No -> record tokens

    # --- Stage 3: Reward ---
    r_s3 = c.region(x=620, y=52, w=250, h=200, title="Stage 3: Reward", palette="data")
    b_eval = r_s3.box("Evaluate trajectory")
    b_assign = r_s3.box("Assign scalar reward")
    b_status = r_s3.box("Set status = COMPLETED")

    # --- Arrows: Stage 1 internal ---
    c.arrow(b_raw, b_create, label="", src_side="bottom", dst_side="top")
    c.arrow(b_create, b_conv, label="", src_side="bottom", dst_side="top")

    # Stage 1 -> Stage 2
    c.arrow(b_conv, b_tok, label="Sample (PENDING)", src_side="right", dst_side="left")

    # Stage 2 internal
    c.arrow(b_tok, b_query, label="", src_side="bottom", dst_side="top")
    c.arrow(b_query, d_tool, label="", src_side="bottom", dst_side="top")
    # Yes -> execute tool
    c.arrow(d_tool, b_exec, label="Yes", src_side="bottom", dst_side="top")
    c.arrow(b_exec, b_append, label="", src_side="bottom", dst_side="top")
    # Loop back to query
    c.arrow(b_append, b_query, label="", src_side="left", dst_side="left", waypoints=[(265, 331), (265, 165)])
    # No -> record
    c.arrow(d_tool, b_rec, label="No", src_side="left", dst_side="bottom", waypoints=[(320, 224)])
    # Record -> Stage 3
    c.arrow(b_rec, b_eval, label="Sample (ROLLEDOUT)", src_side="top", dst_side="left", waypoints=[(320, 108)])

    # Stage 3 internal
    c.arrow(b_eval, b_assign, label="", src_side="bottom", dst_side="top")
    c.arrow(b_assign, b_status, label="", src_side="bottom", dst_side="top")

    return c.render()


# ===================================================================
# developer_guide/contributing  — Diagram 1
# Contribution workflow (top-down, 4 stages with feedback loop)
# ===================================================================


def render_contributing_1() -> str:
    c = SvgCanvas(
        width=860,
        height=760,
        title="Contribution Workflow",
        caption="Figure 1: Fork → Validate → Submit → Review cycle",
    )

    # --- Stage 1: Prepare ---
    r_prep = c.region(x=290, y=52, w=280, h=160, title="1. Prepare", palette="rollout")
    b_fork = r_prep.box("Fork repository")
    b_branch = r_prep.box("Create feature branch")
    b_impl = r_prep.box("Implement changes")

    # --- Stage 2: Validate ---
    r_val = c.region(x=290, y=240, w=280, h=160, title="2. Validate", palette="training")
    b_hooks = r_val.box("Run pre-commit hooks")
    b_test = r_val.box("Run pytest")
    b_docs = r_val.box("Update documentation")

    # --- Stage 3: Submit ---
    r_sub = c.region(x=290, y=428, w=280, h=130, title="3. Submit", palette="data")
    b_commit = r_sub.box("Write clear commit message")
    b_push = r_sub.box("Push to fork → Open PR")

    # --- Stage 4: Review ---
    r_rev = c.region(x=245, y=586, w=370, h=110, title="4. Review", palette="tools")
    b_auto = c.box(x=265, y=630, w=160, h=BOX_HEIGHT, label="Automated checks", palette="tools")
    b_review = c.box(x=435, y=630, w=160, h=BOX_HEIGHT, label="Code review", palette="tools")

    # Diamond: Changes requested?
    d_change = c.diamond(x=405, y=700, size=44, label="OK?", palette="highlight")

    # Merge
    b_merge = c.box(x=520, y=700, w=140, h=BOX_HEIGHT, label="Merge to master", bold=True, palette="training")

    # --- Arrows ---
    c.arrow(b_fork, b_branch, label="git fork", src_side="bottom", dst_side="top")
    c.arrow(b_branch, b_impl, label="checkout -b", src_side="bottom", dst_side="top")
    c.arrow(b_impl, b_hooks, label="code ready", src_side="bottom", dst_side="top")
    c.arrow(b_hooks, b_test, label="all pass", src_side="bottom", dst_side="top")
    c.arrow(b_test, b_docs, label="tests pass", src_side="bottom", dst_side="top")
    c.arrow(b_docs, b_commit, label="docs updated", src_side="bottom", dst_side="top")
    c.arrow(b_commit, b_push, label="", src_side="bottom", dst_side="top")
    c.arrow(b_push, b_auto, label="PR created", src_side="bottom", dst_side="top")
    c.arrow(b_auto, b_review, label="", src_side="right", dst_side="left")
    c.arrow(b_review, d_change, label="", src_side="bottom", dst_side="top")

    # Feedback loop: Changes requested -> back to Implement
    c.arrow(d_change, b_impl, label="Changes requested", src_side="left", dst_side="left", waypoints=[(240, 722), (240, 150)])
    # Approved -> merge
    c.arrow(d_change, b_merge, label="Approved", src_side="right", dst_side="left")

    return c.render()


# ===================================================================
# advanced/failure_propagation  — Diagram 1
# State diagram: system lifecycle states
# ===================================================================


def render_failure_propagation_1() -> str:
    c = SvgCanvas(
        width=820,
        height=440,
        title="System Lifecycle State Diagram",
        caption="Figure 1: RUNNING → COMPLETED / FAILED / SHUTDOWN state transitions",
    )

    sd = c.state_diagram()

    # Start dot
    sd.start_dot(100, 180, name="start")

    # States
    sd.state("running", "RUNNING", x=180, y=164, w=150, h=32, palette_name="training")
    sd.state("completed", "COMPLETED", x=460, y=80, w=160, h=32, palette_name="rollout")
    sd.state("failed", "FAILED", x=460, y=180, w=160, h=32, palette_name="error")
    sd.state("shutdown", "SHUTDOWN", x=460, y=280, w=160, h=32, palette_name="data")

    # End dots
    sd.end_dot(720, 80, name="end1")
    sd.end_dot(720, 180, name="end2")
    sd.end_dot(720, 280, name="end3")

    # Transitions
    sd.transition("start", "running", label="System initialized", src_side="right", dst_side="left")
    sd.transition("running", "completed", label="All epochs finished", src_side="right", dst_side="left", waypoints=[(400, 96)])
    sd.transition("running", "failed", label="Error in any component", src_side="right", dst_side="left")
    sd.transition("running", "shutdown", label="Graceful shutdown", src_side="right", dst_side="left", waypoints=[(400, 296)])
    sd.transition("completed", "end1", label="", src_side="right", dst_side="left")
    sd.transition("failed", "end2", label="", src_side="right", dst_side="left")
    sd.transition("shutdown", "end3", label="", src_side="right", dst_side="left")

    # Notes
    c.text(255, 220, "should_stop() → False", size=10, color="#607D8B", style="italic")
    c.text(540, 230, "should_stop() → True", size=10, color="#C62828", style="italic")
    c.text(540, 248, "failure_reason logged", size=10, color="#C62828", style="italic")

    return c.render()


# ===================================================================
# advanced/failure_propagation  — Diagram 2
# Sequence diagram: failure propagation across components
# ===================================================================


def render_failure_propagation_2() -> str:
    c = SvgCanvas(
        width=900,
        height=580,
        title="Failure Propagation Sequence",
        caption="Figure 2: CUDA OOM in Trainer → TaskCoordinator → graceful shutdown",
    )

    seq = c.sequence(
        actors=[
            {"name": "Trainer", "short": "T", "palette": "training"},
            {"name": "TaskCoordinator", "short": "TC", "palette": "infra"},
            {"name": "RolloutManager", "short": "RM", "palette": "rollout"},
            {"name": "DataCoordinator", "short": "DC", "palette": "data"},
            {"name": "MainRunner", "short": "MR", "palette": "highlight"},
        ],
        x=100,
        y=60,
        actor_spacing=165,
        msg_spacing=36,
    )

    # Note: CUDA OOM
    seq.note("T", "CUDA OOM occurs", side="right", width=110)

    # Failure report
    seq.message("T", "TC", label='report_failure("trainer", "CUDA OOM")')
    seq.message("TC", "TC", label="Set status = FAILED", self_msg=True)

    # Polling loop: RM
    seq.loop("Polling loop")
    seq.message("RM", "TC", label="should_stop()?")
    seq.message("TC", "RM", label="True", dashed=True)
    seq.end_loop()

    seq.note("RM", "Cancel pending rollouts", side="right", width=130)

    # Polling loop: DC
    seq.loop("Polling loop")
    seq.message("DC", "TC", label="should_stop()?")
    seq.message("TC", "DC", label="True", dashed=True)
    seq.end_loop()

    seq.note("DC", "Stop accepting samples", side="right", width=130)

    # Summary
    seq.message("MR", "TC", label="get_summary()")
    seq.message("TC", "MR", label="{status: failed, reason: CUDA OOM}", dashed=True)
    seq.note("MR", "Run cleanup procedures", side="right", width=130)

    # Render actors
    for part in seq.render_actors():
        c._elements.append(part)

    return c.render()


# ===================================================================
# advanced/failure_propagation  — Diagram 3
# Error propagation flow (left-to-right)
# ===================================================================


def render_failure_propagation_3() -> str:
    c = SvgCanvas(
        width=880,
        height=360,
        title="Error Propagation Pipeline",
        caption="Figure 3: Error origin → failure propagation → graceful shutdown",
    )

    # --- Region: Trigger ---
    r_trig = c.region(x=30, y=60, w=220, h=80, title="Error Origin", palette="error")
    b_err = r_trig.box("Trainer: CUDA OOM", bold=True)

    # --- Region: Propagation ---
    r_prop = c.region(x=290, y=60, w=250, h=120, title="Failure Propagation", palette="data")
    b_report = r_prop.box("report_failure to Coordinator")
    b_coord = r_prop.box("Coordinator status = FAILED")

    # --- Region: Shutdown ---
    r_shut = c.region(x=580, y=60, w=270, h=160, title="Graceful Shutdown", palette="infra")
    b_cancel = r_shut.box("RolloutManager: cancel rollouts")
    b_flush = r_shut.box("DataCoordinator: flush buffer")
    b_exit = r_shut.box("MainRunner: cleanup & exit")

    # --- Arrows ---
    c.arrow(b_err, b_report, label="error caught", src_side="right", dst_side="left")
    c.arrow(b_report, b_coord, label="set status", src_side="bottom", dst_side="top")
    c.arrow(b_coord, b_cancel, label="should_stop = True", src_side="right", dst_side="left")
    c.arrow(b_coord, b_flush, label="should_stop = True", src_side="right", dst_side="left", waypoints=[(565, 148)])
    c.arrow(b_cancel, b_exit, label="", src_side="bottom", dst_side="top")
    c.arrow(b_flush, b_exit, label="", src_side="bottom", dst_side="top")

    return c.render()


# ===================================================================
# advanced/performance_tuning  — Diagram 1
# Bottleneck diagnosis tree (top-bottom)
# ===================================================================


def render_performance_tuning_1() -> str:
    c = SvgCanvas(
        width=900,
        height=580,
        title="Bottleneck Diagnosis Decision Tree",
        caption="Figure 1: Check GPU utilization → identify bottleneck → apply tuning strategy",
    )

    # --- Region: Diagnosis ---
    r_diag = c.region(x=300, y=52, w=300, h=110, title="Bottleneck Diagnosis", palette="infra")
    b_check = r_diag.box("Check GPU Utilization", bold=True)

    # Three condition boxes
    b_train_idle = c.box(x=40, y=200, w=200, h=BOX_HEIGHT, label="Training GPUs idle?", palette="infra")
    b_roll_idle = c.box(x=350, y=200, w=200, h=BOX_HEIGHT, label="Rollout GPUs idle?", palette="infra")
    b_both = c.box(x=660, y=200, w=200, h=BOX_HEIGHT, label="Both GPUs busy?", palette="infra")

    # --- Region: Rollout Bottleneck ---
    r_rb = c.region(x=20, y=270, w=240, h=160, title="Rollout is Bottleneck", palette="rollout")
    b_conc = r_rb.box("Increase rollout concurrency")
    b_gpu = r_rb.box("Add more rollout GPUs")
    b_speed = r_rb.box("Speed up tool environments")

    # --- Region: Training Bottleneck ---
    r_tb = c.region(x=330, y=270, w=240, h=160, title="Training is Bottleneck", palette="training")
    b_micro = r_tb.box("Reduce micro-batch size")
    b_offload = r_tb.box("Enable param offloading")
    b_tgpu = r_tb.box("Add more training GPUs")

    # --- Region: Balanced ---
    r_bal = c.region(x=640, y=270, w=240, h=120, title="Pipeline Balanced", palette="data")
    b_async = r_bal.box("Fine-tune async_factor")
    b_mon = r_bal.box("Monitor for regression")

    # --- Arrows ---
    c.arrow(b_check, b_train_idle, label="gen_duration >> step_time", src_side="bottom", dst_side="top", waypoints=[(450, 180), (140, 180)])
    c.arrow(b_check, b_roll_idle, label="step_time >> gen_duration", src_side="bottom", dst_side="top")
    c.arrow(b_check, b_both, label="Both similar", src_side="bottom", dst_side="top", waypoints=[(450, 180), (760, 180)])

    c.arrow(b_train_idle, b_conc, label="", src_side="bottom", dst_side="top", waypoints=[(140, 260)])
    c.arrow(b_roll_idle, b_micro, label="", src_side="bottom", dst_side="top", waypoints=[(450, 260)])
    c.arrow(b_both, b_async, label="", src_side="bottom", dst_side="top", waypoints=[(760, 260)])

    return c.render()


# ===================================================================
# advanced/performance_tuning  — Diagram 2
# Sequence diagram: async pipeline (RM ↔ DC ↔ TG)
# ===================================================================


def render_performance_tuning_2() -> str:
    c = SvgCanvas(
        width=750,
        height=560,
        title="Async Training Pipeline (async_factor = 2)",
        caption="Figure 2: RolloutManager buffers batches ahead of TrainerGroup",
    )

    seq = c.sequence(
        actors=[
            {"name": "RolloutManager", "short": "RM", "palette": "rollout"},
            {"name": "DataCoordinator", "short": "DC", "palette": "data"},
            {"name": "TrainerGroup", "short": "TG", "palette": "training"},
        ],
        x=130,
        y=60,
        actor_spacing=220,
        msg_spacing=34,
    )

    seq.note("RM", "async_factor = 2", side="right", width=120)

    seq.message("RM", "DC", label="Scored batch 1")
    seq.message("RM", "DC", label="Scored batch 2")
    seq.note("RM", "Rollout continues...", side="right", width=120)
    seq.message("DC", "TG", label="Training batch 1")
    seq.message("TG", "TG", label="Forward + Backward", self_msg=True)
    seq.message("RM", "DC", label="Scored batch 3")
    seq.message("DC", "TG", label="Training batch 2")
    seq.message("TG", "TG", label="Forward + Backward", self_msg=True)
    seq.message("TG", "RM", label="Weight sync")
    seq.note("RM", "Uses updated weights", side="right", width=130)
    seq.message("RM", "DC", label="Scored batch 4")
    seq.message("DC", "TG", label="Training batch 3")
    seq.message("TG", "TG", label="Forward + Backward", self_msg=True)

    for part in seq.render_actors():
        c._elements.append(part)

    return c.render()


# ===================================================================
# faq/troubleshooting  — Diagram 1
# Troubleshooting decision tree (top-down)
# ===================================================================


def render_troubleshooting_1() -> str:
    c = SvgCanvas(
        width=900,
        height=820,
        title="Troubleshooting Decision Tree",
        caption="Figure 1: Identify error stage → diagnose → apply fix",
    )

    # Central diamond: Where does the error occur?
    d_start = c.diamond(x=390, y=56, size=56, label="Stage?", palette="highlight")

    # Four category labels
    # --- Column 1: Init ---
    r_init = c.region(x=20, y=140, w=200, h=200, title="Initialization", palette="infra")
    b_ray = r_init.box("Ray connection?")
    b_gpualloc = r_init.box("GPU allocation?")
    b_sglang = r_init.box("SGLang startup?")

    # Init solutions
    b_ray_fix = c.box(x=20, y=360, w=200, h=30, label="ray stop && ray start", palette="infra")
    b_gpu_fix = c.box(x=20, y=400, w=200, h=30, label="Reduce actor/rollout GPUs", palette="infra")
    b_sgl_fix = c.box(x=20, y=440, w=200, h=30, label="Reduce gpu_mem_utilization", palette="infra")

    # --- Column 2: Training ---
    r_train = c.region(x=240, y=140, w=200, h=200, title="Training Loop", palette="training")
    b_oom = r_train.box("OOM error?")
    b_noprog = r_train.box("No progress?")
    b_reward0 = r_train.box("Reward = 0?")

    # Training solutions
    b_oom_fix = c.box(x=240, y=360, w=200, h=30, label="Reduce micro_batch_size", palette="training")
    b_prog_fix = c.box(x=240, y=400, w=200, h=30, label="Check DataCoordinator logs", palette="training")
    b_rew_fix = c.box(x=240, y=440, w=200, h=30, label="Verify reward_fn & data", palette="training")

    # --- Column 3: Agentic ---
    r_agent = c.region(x=460, y=140, w=220, h=240, title="Multi-Turn / Agentic", palette="tools")
    b_tool_det = r_agent.box("Tool calls not detected?")
    b_aio = r_agent.box("AIO connection failed?")
    b_timeout = r_agent.box("Tool timeout?")
    b_short = r_agent.box("Trajectories too short?")

    # Agentic solutions
    b_tool_fix = c.box(x=460, y=400, w=220, h=30, label="Set tool_format: hermes", palette="tools")
    b_aio_fix = c.box(x=460, y=440, w=220, h=30, label="Check AIO Proxy health", palette="tools")
    b_to_fix = c.box(x=460, y=480, w=220, h=30, label="Scale AIO instances", palette="tools")
    b_short_fix = c.box(x=460, y=520, w=220, h=30, label="Increase max_env_turns", palette="tools")

    # --- Column 4: Checkpoint ---
    r_ckpt = c.region(x=700, y=140, w=180, h=160, title="Checkpoint", palette="data")
    b_resume = r_ckpt.box("Resume fails?")
    b_disk = r_ckpt.box("Disk full?")

    # Checkpoint solutions
    b_res_fix = c.box(x=700, y=360, w=180, h=30, label="Check resume_mode & path", palette="data")
    b_disk_fix = c.box(x=700, y=400, w=180, h=30, label="max_actor_ckpt_to_keep: 5", palette="data")

    # --- Arrows from central diamond to categories ---
    c.arrow(d_start, b_ray, label="Startup fails", src_side="left", dst_side="top", waypoints=[(120, 84)])
    c.arrow(d_start, b_oom, label="Training error", src_side="bottom", dst_side="top", waypoints=[(418, 130), (340, 130)])
    c.arrow(d_start, b_tool_det, label="Tool issues", src_side="right", dst_side="top", waypoints=[(570, 84)])
    c.arrow(d_start, b_resume, label="Checkpoint", src_side="right", dst_side="top", waypoints=[(790, 84)])

    # Issue -> Fix arrows
    c.arrow(b_ray, b_ray_fix, label="", src_side="bottom", dst_side="top", waypoints=[(120, 350)])
    c.arrow(b_gpualloc, b_gpu_fix, label="", src_side="bottom", dst_side="top", waypoints=[(120, 390)])
    c.arrow(b_sglang, b_sgl_fix, label="", src_side="bottom", dst_side="top", waypoints=[(120, 435)])
    c.arrow(b_oom, b_oom_fix, label="", src_side="bottom", dst_side="top", waypoints=[(340, 350)])
    c.arrow(b_noprog, b_prog_fix, label="", src_side="bottom", dst_side="top", waypoints=[(340, 390)])
    c.arrow(b_reward0, b_rew_fix, label="", src_side="bottom", dst_side="top", waypoints=[(340, 435)])
    c.arrow(b_tool_det, b_tool_fix, label="", src_side="bottom", dst_side="top", waypoints=[(570, 390)])
    c.arrow(b_aio, b_aio_fix, label="", src_side="bottom", dst_side="top", waypoints=[(570, 435)])
    c.arrow(b_timeout, b_to_fix, label="", src_side="bottom", dst_side="top", waypoints=[(570, 475)])
    c.arrow(b_short, b_short_fix, label="", src_side="bottom", dst_side="top", waypoints=[(570, 515)])
    c.arrow(b_resume, b_res_fix, label="", src_side="bottom", dst_side="top", waypoints=[(790, 350)])
    c.arrow(b_disk, b_disk_fix, label="", src_side="bottom", dst_side="top", waypoints=[(790, 390)])

    return c.render()


# ===================================================================
# reference/module_map  — Diagram 1
# Full module architecture map (top-down, multi-layer)
# ===================================================================


def render_module_map_1() -> str:
    c = SvgCanvas(
        width=900,
        height=880,
        title="Module Architecture Map",
        caption="Figure 1: Complete module dependency graph — entry point through foundation",
    )

    # ── Layer 0: Entry Point ──
    r_entry = c.region(x=340, y=52, w=220, h=68, title="Entry Point", palette="highlight")
    b_main = r_entry.box("async_train.py / MainRunner", bold=True)

    # ── Layer 1: Worker Layer ──
    r_work = c.region(x=50, y=155, w=800, h=80, title="Worker Layer", palette="rollout")
    b_tg = c.box(x=80, y=195, w=210, h=BOX_HEIGHT, label="worker/actor/ TrainerGroup", bold=True, palette="training")
    b_rm = c.box(x=330, y=195, w=230, h=BOX_HEIGHT, label="worker/rollout/ RolloutManager", bold=True, palette="rollout")
    b_val = c.box(x=600, y=195, w=230, h=BOX_HEIGHT, label="worker/validate/ ValidateMonitor", palette="infra")

    # ── Layer 2: Execution Layer ──
    r_exec = c.region(x=240, y=270, w=500, h=115, title="Execution Layer", palette="tools")
    b_nf = c.box(x=260, y=310, w=210, h=BOX_HEIGHT, label="agent_flow/ NaiveFlow", palette="tools")
    b_af = c.box(x=510, y=310, w=210, h=BOX_HEIGHT, label="agentflow/ AgentFlow Protocol", palette="tools")
    b_agex = c.box(x=380, y=350, w=220, h=BOX_HEIGHT, label="agent_executor/", palette="tools")

    # ── Layer 3: Engine Layer ──
    r_eng = c.region(x=50, y=420, w=540, h=115, title="Engine Layer", palette="infra")
    b_actor_eng = c.box(x=70, y=460, w=170, h=BOX_HEIGHT, label="engine/actor/", bold=True, palette="training")
    b_rollout_eng = c.box(x=260, y=460, w=170, h=BOX_HEIGHT, label="engine/rollout/ SGLang", bold=True, palette="rollout")
    b_sync = c.box(x=450, y=460, w=120, h=BOX_HEIGHT, label="param_sync/", palette="infra")

    # ── Layer 4: Data Layer ──
    r_data = c.region(x=620, y=420, w=250, h=115, title="Data Layer", palette="data")
    b_dc = c.box(x=640, y=460, w=210, h=BOX_HEIGHT, label="data_coordinator/", bold=True, palette="data")
    b_dl = c.box(x=640, y=500, w=210, h=BOX_HEIGHT, label="dataloader/", palette="data")

    # ── Layer 5: Environment Layer ──
    r_env = c.region(x=50, y=570, w=280, h=80, title="Environment Layer", palette="tools")
    b_toolenv = r_env.box("tool_env/ ToolEnv, ToolParser")

    # ── Layer 6: Algorithm Layer ──
    r_algo = c.region(x=370, y=570, w=230, h=80, title="Algorithm Layer", palette="training")
    b_algo = r_algo.box("algorithm/ advantage, loss, KL")

    # ── Layer 7: Foundation ──
    r_found = c.region(x=50, y=690, w=800, h=115, title="Foundation", palette="infra")
    b_params = c.box(x=80, y=730, w=200, h=BOX_HEIGHT, label="params/ Config dataclasses", palette="infra")
    b_models = c.box(x=330, y=730, w=200, h=BOX_HEIGHT, label="models/ Model loading", palette="infra")
    b_utils = c.box(x=580, y=730, w=250, h=BOX_HEIGHT, label="utils/ TaskCoordinator, metrics", palette="infra")

    # ── Arrows: Entry → Workers ──
    c.arrow(b_main, b_tg, label="orchestrates", src_side="bottom", dst_side="top", waypoints=[(450, 148), (185, 148)])
    c.arrow(b_main, b_rm, label="orchestrates", src_side="bottom", dst_side="top")
    c.arrow(b_main, b_val, label="orchestrates", src_side="bottom", dst_side="top", waypoints=[(450, 148), (715, 148)])

    # Workers → Execution
    c.arrow(b_rm, b_agex, label="dispatches to", src_side="bottom", dst_side="top")
    c.arrow(b_agex, b_nf, label="runs", src_side="left", dst_side="bottom")
    c.arrow(b_agex, b_af, label="runs", src_side="right", dst_side="bottom")

    # Workers → Engine
    c.arrow(b_tg, b_actor_eng, label="trains with", src_side="bottom", dst_side="top", waypoints=[(185, 440)])
    c.arrow(b_rm, b_rollout_eng, label="infers with", src_side="bottom", dst_side="top", waypoints=[(445, 440), (345, 440)])

    # Workers → Data
    c.arrow(b_tg, b_dc, label="fetches data", src_side="bottom", dst_side="top", waypoints=[(185, 410), (745, 410)])

    # Engine internal
    c.arrow(b_actor_eng, b_sync, label="syncs weights", src_side="right", dst_side="left")
    c.arrow(b_sync, b_rollout_eng, label="updates", src_side="left", dst_side="right")

    # Data internal
    c.arrow(b_dc, b_dl, label="loads from", src_side="bottom", dst_side="top")

    # Execution → Environment
    c.arrow(b_nf, b_toolenv, label="calls tools", src_side="bottom", dst_side="top", waypoints=[(365, 560), (190, 560)])
    c.arrow(b_af, b_toolenv, label="calls tools", src_side="bottom", dst_side="top", waypoints=[(615, 560), (190, 560)])

    # Workers → Algorithm
    c.arrow(b_tg, b_algo, label="computes loss", src_side="bottom", dst_side="top", waypoints=[(185, 560), (485, 560)])

    # Workers / Entry → Foundation
    c.arrow(b_main, b_params, label="reads config", src_side="bottom", dst_side="top", waypoints=[(450, 680), (180, 680)])
    c.arrow(b_tg, b_models, label="loads model", src_side="bottom", dst_side="top", waypoints=[(185, 680), (430, 680)])
    c.arrow(b_rm, b_models, label="loads model", src_side="bottom", dst_side="top", waypoints=[(445, 680), (430, 680)])
    c.arrow(b_main, b_utils, label="lifecycle mgmt", src_side="bottom", dst_side="top", waypoints=[(450, 680), (705, 680)])

    return c.render()


# ===================================================================
# REGISTRY — maps output filename to render function
# ===================================================================

REGISTRY = {
    "developer_guidecode_structure_1.svg": render_code_structure_1,
    "developer_guidecode_structure_2.svg": render_code_structure_2,
    "developer_guidecode_structure_3.svg": render_code_structure_3,
    "developer_guidecode_structure_4.svg": render_code_structure_4,
    "developer_guidecode_structure_5.svg": render_code_structure_5,
    "developer_guideadding_new_executor_or_flow_1.svg": render_adding_new_executor_1,
    "developer_guideadding_new_executor_or_flow_2.svg": render_adding_new_executor_2,
    "developer_guidecontributing_1.svg": render_contributing_1,
    "advancedfailure_propagation_1.svg": render_failure_propagation_1,
    "advancedfailure_propagation_2.svg": render_failure_propagation_2,
    "advancedfailure_propagation_3.svg": render_failure_propagation_3,
    "advancedperformance_tuning_1.svg": render_performance_tuning_1,
    "advancedperformance_tuning_2.svg": render_performance_tuning_2,
    "faqtroubleshooting_1.svg": render_troubleshooting_1,
    "referencemodule_map_1.svg": render_module_map_1,
}


# ===================================================================
# CLI entry point — render all or a specific diagram
# ===================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render batch 4 SVG diagrams")
    parser.add_argument("--output-dir", default="docs/mkdocs/docs/assets/images/diagrams", help="Directory to write SVGs")
    parser.add_argument("--only", default=None, help="Render only this filename key")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    items = REGISTRY.items()
    if args.only:
        items = [(k, v) for k, v in items if k == args.only]

    for fname, fn in items:
        svg = fn()
        dest = out / fname
        dest.write_text(svg)
        print(f"  ✓ {fname}  ({len(svg)} bytes)")

    print(f"\nDone — {len(list(items))} diagrams written to {out}/")
