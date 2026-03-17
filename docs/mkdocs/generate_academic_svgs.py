#!/usr/bin/env python3
"""Parse mermaid sources and generate academic-style SVGs with custom layout.

Reads /tmp/mermaid_diagrams.json, applies a custom layout algorithm,
and renders clean, hand-crafted-quality SVGs using svg_renderer.py.

Handles: flowchart (45), sequence (8), state (3), class (2), gantt (1).
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from svg_renderer import (
    ARROW_COLOR,
    BOX_BOLD_SIZE,
    BOX_GAP,
    BOX_HEIGHT,
    BOX_TEXT_SIZE,
    LABEL_SIZE,
    PALETTES,
    REGION_PADDING,
    REGION_TITLE_HEIGHT,
    REGION_TITLE_SIZE,
    BoxInfo,
    Point,
    Rect,
    Region,
    SvgCanvas,
    estimate_text_width,
)

DIAGRAMS_DIR = Path(__file__).parent / "docs" / "assets" / "images" / "diagrams"

# ---------------------------------------------------------------------------
# Mermaid parser (simplified, reuses patterns from mermaid_to_d2.py)
# ---------------------------------------------------------------------------


@dataclass
class MNode:
    id: str
    label: str
    shape: Optional[str] = None  # "diamond", "oval", "circle", "hexagon", None=rect
    container: Optional[str] = None


@dataclass
class MEdge:
    src: str
    dst: str
    label: str = ""
    style: str = ""  # "dashed", "thick", ""


@dataclass
class MSubgraph:
    id: str
    label: str
    parent: Optional[str] = None
    children: list[str] = field(default_factory=list)  # node ids


@dataclass
class MFlowchart:
    direction: str  # "TB", "LR", "BT", "RL"
    nodes: dict[str, MNode] = field(default_factory=dict)
    edges: list[MEdge] = field(default_factory=list)
    subgraphs: dict[str, MSubgraph] = field(default_factory=dict)
    sg_order: list[str] = field(default_factory=list)


def _clean(label: str) -> str:
    label = label.replace("<br/>", "\n").replace("<br>", "\n")
    label = re.sub(r"</?[a-zA-Z][^>]*>", "", label)
    if len(label) >= 2 and label[0] == '"' and label[-1] == '"':
        label = label[1:-1]
    # Strip mermaid trapezoid/shape markers: /"...", /".."/, ["..."], etc.
    label = re.sub(r'^[/\\]"', "", label)
    label = re.sub(r'"[/\\]$', "", label)
    return label.strip()


def _parse_node(token: str) -> tuple[str, str, Optional[str]]:
    """Parse a mermaid node token -> (id, label, shape)."""
    token = token.strip()
    if not token:
        return ("", "", None)
    patterns = [
        (r'^(\w+)\[\("(.*)"\)\]$', "cylinder"),
        (r'^(\w+)\(\("(.*)"\)\)$', "circle"),
        (r"^(\w+)\(\((.+)\)\)$", "circle"),
        (r'^(\w+)\{\{"(.*)"\}\}$', "hexagon"),
        (r"^(\w+)\{\{(.+)\}\}$", "hexagon"),
        (r'^(\w+)\{"(.*)"\}$', "diamond"),
        (r"^(\w+)\{(.+)\}$", "diamond"),
        (r'^(\w+)\("(.*)"\)$', "oval"),
        (r"^(\w+)\((.+)\)$", "oval"),
        (r'^(\w+)\[\["(.*)"\]\]$', None),  # subroutine [[" "]]
        (r"^(\w+)\[\[(.+)\]\]$", None),  # subroutine [[ ]]
        (r'^(\w+)\["(.*)"\]$', None),
        (r"^(\w+)\[(.+)\]$", None),
        (r"^(\w+)$", None),
    ]
    for pat, shape in patterns:
        m = re.match(pat, token, re.DOTALL)
        if m:
            nid = m.group(1)
            label = m.group(2) if m.lastindex >= 2 else nid
            return (nid, _clean(label), shape)
    return (token, token, None)


def parse_flowchart(source: str) -> MFlowchart:
    """Parse mermaid flowchart source into structured data."""
    lines = source.strip().split("\n")
    fc = MFlowchart(direction="TB")

    first = lines[0].strip()
    m = re.match(r"(?:graph|flowchart)\s+(TD|TB|LR|RL|BT)", first)
    if m:
        fc.direction = m.group(1) if m.group(1) != "TD" else "TB"

    sg_stack: list[str] = []

    def current_container():
        return sg_stack[-1] if sg_stack else None

    def register(nid, label, shape):
        if not nid:
            return
        if nid not in fc.nodes:
            fc.nodes[nid] = MNode(id=nid, label=label, shape=shape, container=current_container())
            # Add to parent subgraph's children
            if current_container() and current_container() in fc.subgraphs:
                fc.subgraphs[current_container()].children.append(nid)

    for raw in lines[1:]:
        line = raw.strip()
        if not line or line.startswith("%%"):
            continue

        # Subgraph
        if line.startswith("subgraph "):
            sg_m = re.match(r'subgraph\s+(\w+)\["(.*)"\]', line)
            if not sg_m:
                sg_m = re.match(r'subgraph\s+(\w+)\["(.*)"\]', line)
            if not sg_m:
                sg_m = re.match(r"subgraph\s+(\w+)\s*$", line)
            if not sg_m:
                # subgraph "Label" or subgraph Label
                sg_m2 = re.match(r'subgraph\s+"?([^"]+)"?\s*$', line)
                if sg_m2:
                    label = sg_m2.group(1).strip()
                    sg_id = re.sub(r"[^\w]", "_", label)
                    sg = MSubgraph(id=sg_id, label=_clean(label), parent=current_container())
                    fc.subgraphs[sg_id] = sg
                    fc.sg_order.append(sg_id)
                    sg_stack.append(sg_id)
                    continue
            if sg_m:
                sg_id = sg_m.group(1)
                sg_label = _clean(sg_m.group(2)) if sg_m.lastindex >= 2 else sg_id
                sg = MSubgraph(id=sg_id, label=sg_label, parent=current_container())
                fc.subgraphs[sg_id] = sg
                fc.sg_order.append(sg_id)
                sg_stack.append(sg_id)
            continue

        if line == "end":
            if sg_stack:
                sg_stack.pop()
            continue

        # Style / class directives - skip
        if line.startswith("style ") or line.startswith("classDef ") or line.startswith("class "):
            continue
        if line.startswith("linkStyle"):
            continue
        if line.startswith("direction "):
            continue

        # Edge line
        edge_pat = r"(<-->|-.->|==>|-->|---|<--)"
        if re.search(edge_pat, line):
            # Extract labels |"text"|
            labels = {}
            counter = [0]

            def repl(m):
                key = f"__L{counter[0]}__"
                counter[0] += 1
                labels[key] = m.group(2)
                return m.group(1) + " " + key

            processed = re.sub(r'(-->|-.->|==>|<-->|---)\|"?([^"|]*)"?\|', repl, line)

            parts = re.split(edge_pat, processed)
            i = 0
            while i + 2 < len(parts):
                src_tok = parts[i].strip()
                edge_op = parts[i + 1].strip()
                dst_tok = parts[i + 2].strip()

                label = ""
                for key, val in labels.items():
                    if key in dst_tok:
                        label = _clean(val)
                        dst_tok = dst_tok.replace(key, "").strip()
                        break
                    if key in src_tok:
                        label = _clean(val)
                        src_tok = src_tok.replace(key, "").strip()
                        break

                style = ""
                if edge_op == "-.->":
                    style = "dashed"
                elif edge_op == "==>":
                    style = "thick"

                # Handle & splits (mermaid parallel syntax: A & B --> C)
                # But don't split on & inside bracket-quoted labels like ["text & more"]
                def safe_amp_split(tok: str) -> list[str]:
                    """Split on ' & ' only when not inside [...] brackets."""
                    result = []
                    depth = 0
                    current = []
                    i = 0
                    while i < len(tok):
                        ch = tok[i]
                        if ch == "[":
                            depth += 1
                        elif ch == "]":
                            depth = max(0, depth - 1)
                        if depth == 0 and tok[i : i + 3] == " & ":
                            result.append("".join(current))
                            current = []
                            i += 3
                            continue
                        current.append(ch)
                        i += 1
                    result.append("".join(current))
                    return result

                for st in safe_amp_split(src_tok):
                    sid, sl, ss = _parse_node(st.strip())
                    register(sid, sl, ss)
                    for dt in safe_amp_split(dst_tok):
                        did, dl, ds = _parse_node(dt.strip())
                        register(did, dl, ds)
                        if sid and did:
                            fc.edges.append(MEdge(src=sid, dst=did, label=label, style=style))
                i += 2
        else:
            # Standalone node definition
            nid, nl, ns = _parse_node(line.rstrip(";"))
            if nid and nid not in ("graph", "flowchart"):
                register(nid, nl, ns)

    return fc


# ---------------------------------------------------------------------------
# Sequence diagram parser
# ---------------------------------------------------------------------------


@dataclass
class SeqActor:
    id: str
    label: str


@dataclass
class SeqMessage:
    src: str
    dst: str
    label: str
    dashed: bool = False
    self_msg: bool = False


@dataclass
class SeqNote:
    actors: list[str]
    text: str


@dataclass
class SeqBlock:
    """Loop, alt, opt, etc."""

    kind: str  # "loop", "alt", "opt"
    label: str
    messages: list  # SeqMessage, SeqNote, or SeqBlock


@dataclass
class MSequence:
    actors: list[SeqActor] = field(default_factory=list)
    events: list = field(default_factory=list)  # SeqMessage | SeqNote | SeqBlock


def parse_sequence(source: str) -> MSequence:
    """Parse mermaid sequenceDiagram source."""
    seq = MSequence()
    actor_map = {}
    lines = source.strip().split("\n")
    block_stack = []

    for raw in lines:
        line = raw.strip()
        if not line or line == "sequenceDiagram" or line.startswith("%%"):
            continue

        # Participant
        pm = re.match(r"participant\s+(\w+)\s+as\s+(.+)", line)
        if pm:
            a = SeqActor(id=pm.group(1), label=pm.group(2).strip())
            seq.actors.append(a)
            actor_map[a.id] = a
            continue

        # Note
        nm = re.match(r"Note\s+over\s+([^:]+):\s*(.+)", line)
        if nm:
            actors = [a.strip() for a in nm.group(1).split(",")]
            note = SeqNote(actors=actors, text=nm.group(2).strip())
            target = block_stack[-1].messages if block_stack else seq.events
            target.append(note)
            continue

        # Loop/alt/opt start
        bm = re.match(r"(loop|alt|opt|critical|break)\s*(.*)", line)
        if bm:
            block = SeqBlock(kind=bm.group(1), label=bm.group(2).strip(), messages=[])
            target = block_stack[-1].messages if block_stack else seq.events
            target.append(block)
            block_stack.append(block)
            continue

        # End
        if line == "end":
            if block_stack:
                block_stack.pop()
            continue

        # Message: A->>B: text or A-->>B: text
        mm = re.match(r"(\w+)\s*(->>|-->>|->|-->)\s*(\w+)\s*:\s*(.+)", line)
        if mm:
            dashed = "--" in mm.group(2)
            src, dst = mm.group(1), mm.group(3)
            self_msg = src == dst
            msg = SeqMessage(src=src, dst=dst, label=mm.group(4).strip(), dashed=dashed, self_msg=self_msg)
            target = block_stack[-1].messages if block_stack else seq.events
            target.append(msg)
            continue

    return seq


# ---------------------------------------------------------------------------
# State diagram parser
# ---------------------------------------------------------------------------


@dataclass
class MState:
    id: str
    label: str


@dataclass
class MTransition:
    src: str
    dst: str
    label: str = ""


@dataclass
class MStateDiagram:
    states: dict[str, MState] = field(default_factory=dict)
    transitions: list[MTransition] = field(default_factory=list)


def parse_state(source: str) -> MStateDiagram:
    """Parse mermaid stateDiagram-v2 source."""
    sd = MStateDiagram()
    lines = source.strip().split("\n")

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("%%") or line.startswith("stateDiagram"):
            continue
        if line.startswith("direction"):
            continue

        # State definition: state "Label" as ID
        sm = re.match(r'state\s+"([^"]+)"\s+as\s+(\w+)', line)
        if sm:
            sd.states[sm.group(2)] = MState(id=sm.group(2), label=sm.group(1))
            continue

        # Transition: A --> B : label
        tm = re.match(r"(\[?\*?\]?\w*)\s*-->\s*(\[?\*?\]?\w*)\s*(?::\s*(.+))?", line)
        if tm:
            src = tm.group(1).strip()
            dst = tm.group(2).strip()
            label = tm.group(3).strip() if tm.group(3) else ""
            # Register states
            if src == "[*]":
                src = "_start"
                sd.states.setdefault(src, MState(id="_start", label=""))
            elif src not in sd.states:
                sd.states[src] = MState(id=src, label=src)
            if dst == "[*]":
                dst = "_end"
                sd.states.setdefault(dst, MState(id="_end", label=""))
            elif dst not in sd.states:
                sd.states[dst] = MState(id=dst, label=dst)
            sd.transitions.append(MTransition(src=src, dst=dst, label=label))
            continue

    return sd


# ---------------------------------------------------------------------------
# Semantic palette assignment
# ---------------------------------------------------------------------------

_PALETTE_KEYWORDS = {
    "training": [
        "Training",
        "Train",
        "Critic",
        "Actor",
        "Advantage",
        "Update",
        "ForwardPass",
        "Algorithm",
        "TrainingStep",
        "TrainingBottleneck",
        "train_loop",
        "train_phase",
        "train_sys",
        "train_cluster",
    ],
    "rollout": [
        "Rollout",
        "Engine",
        "Executor",
        "Agentic",
        "NaiveFlow",
        "trajectory",
        "rollout_loop",
        "rollout_phase",
        "rollout_sys",
        "rollout_cluster",
        "RolloutEngine",
        "Execution",
        "Sampling",
        "RolloutBottleneck",
        "NaiveFlowStates",
    ],
    "data": [
        "Data",
        "buffer",
        "Loading",
        "Coordinator",
        "sample",
        "data_loop",
        "data_layer",
        "data_sys",
        "DataLayer",
        "buffer_phase",
        "Aggregation",
    ],
    "tools": ["Tools", "tool", "aio", "Environment", "Client", "Backends", "tool_layer"],
    "infra": [
        "MainRunner",
        "driver",
        "Init",
        "Cleanup",
        "Shutdown",
        "Sync",
        "Start",
        "init_sequence",
        "cleanup",
        "orchestrator",
        "coordinate",
        "monitor",
        "monitoring",
    ],
    "highlight": ["Reward", "Checkpoint", "Save", "Validation", "metrics", "predict", "Diagnosis", "resolve"],
    "error": ["Trigger", "Propagation", "Reporting", "trigger"],
}


def palette_for_subgraph(sg_label: str) -> str:
    """Determine the palette name based on subgraph label."""
    for palette_name, keywords in _PALETTE_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in sg_label.lower():
                return palette_name
    return "default"


# ---------------------------------------------------------------------------
# Flowchart layout engine
# ---------------------------------------------------------------------------


@dataclass
class NodePos:
    x: float
    y: float
    w: float
    h: float
    rank: int = 0


def layout_flowchart(fc: MFlowchart) -> tuple[dict[str, NodePos], float, float]:
    """Compute positions for all nodes. Returns (positions, canvas_w, canvas_h)."""
    if not fc.nodes:
        return {}, 200, 100

    horizontal = fc.direction in ("LR", "RL")

    # Compute node widths based on label text
    node_widths = {}
    for nid, node in fc.nodes.items():
        # Use first line of label for width calc
        first_line = node.label.split("\n")[0] if node.label else nid
        tw = estimate_text_width(first_line, BOX_TEXT_SIZE)
        node_widths[nid] = max(tw + 36, 110)

    node_h = BOX_HEIGHT
    h_gap = 40 if horizontal else 50  # gap between nodes in same rank
    v_gap = 55 if not horizontal else 45  # gap between ranks

    # Build adjacency
    children = defaultdict(list)
    parents = defaultdict(list)
    for e in fc.edges:
        if e.src in fc.nodes and e.dst in fc.nodes:
            children[e.src].append(e.dst)
            parents[e.dst].append(e.src)

    # Find roots (no parents)
    all_ids = list(fc.nodes.keys())
    roots = [n for n in all_ids if not parents[n]]
    if not roots:
        roots = [all_ids[0]]

    # Assign ranks via BFS
    ranks = {}
    queue = list(roots)
    for r in roots:
        ranks[r] = 0
    visited = set(roots)
    while queue:
        nid = queue.pop(0)
        for c in children[nid]:
            new_rank = ranks[nid] + 1
            if c not in ranks or new_rank > ranks[c]:
                ranks[c] = new_rank
            if c not in visited:
                visited.add(c)
                queue.append(c)

    # Assign unvisited nodes
    max_rank = max(ranks.values()) if ranks else 0
    for nid in all_ids:
        if nid not in ranks:
            max_rank += 1
            ranks[nid] = max_rank

    # Group by rank
    rank_groups = defaultdict(list)
    for nid, r in ranks.items():
        rank_groups[r].append(nid)

    # Order within rank: by first appearance in source, then by parent position
    for r in rank_groups:
        rank_groups[r].sort(key=lambda n: all_ids.index(n) if n in all_ids else 999)

    # Compute positions
    MAX_ROW_WIDTH = 950  # max width before wrapping to next sub-row
    MAX_COL_HEIGHT = 800  # max height before wrapping to next sub-column

    positions = {}
    if horizontal:
        # LR: ranks go left to right, nodes within rank go top to bottom
        # First pass: simple left-to-right layout
        sorted_ranks = sorted(rank_groups.keys())
        x_cursor = 40
        rank_x: dict[int, float] = {}  # rank -> x start
        rank_max_w: dict[int, float] = {}  # rank -> max node width
        for r in sorted_ranks:
            group = rank_groups[r]
            max_w = max(node_widths.get(n, 120) for n in group)
            rank_x[r] = x_cursor
            rank_max_w[r] = max_w
            x_cursor += max_w + v_gap

        total_lr_width = x_cursor

        # If too wide, wrap ranks into macro-rows
        if total_lr_width > MAX_ROW_WIDTH and len(sorted_ranks) > 4:
            # Split ranks into macro-rows that fit within MAX_ROW_WIDTH
            macro_rows: list[list[int]] = [[]]
            row_w = 0
            for r in sorted_ranks:
                rw = rank_max_w[r] + v_gap
                if row_w + rw > MAX_ROW_WIDTH and macro_rows[-1]:
                    macro_rows.append([])
                    row_w = 0
                macro_rows[-1].append(r)
                row_w += rw

            y_base = 40
            for mrow in macro_rows:
                x_cursor = 40
                mrow_max_y = y_base
                for r in mrow:
                    group = rank_groups[r]
                    y_cursor = y_base
                    for nid in group:
                        w = node_widths.get(nid, 120)
                        positions[nid] = NodePos(x=x_cursor, y=y_cursor, w=w, h=node_h, rank=r)
                        y_cursor += node_h + h_gap
                    mrow_max_y = max(mrow_max_y, y_cursor)
                    x_cursor += rank_max_w[r] + v_gap
                y_base = mrow_max_y + v_gap
        else:
            # Fits in one row, use standard LR layout
            for r in sorted_ranks:
                group = rank_groups[r]
                max_w = rank_max_w[r]
                sub_cols: list[list[str]] = [[]]
                col_h = 0
                for nid in group:
                    needed = node_h + (h_gap if col_h > 0 else 0)
                    if col_h + needed > MAX_COL_HEIGHT and sub_cols[-1]:
                        sub_cols.append([])
                        col_h = 0
                    sub_cols[-1].append(nid)
                    col_h += needed

                sub_x = rank_x[r]
                for col in sub_cols:
                    y_cursor = 40
                    for nid in col:
                        w = node_widths.get(nid, 120)
                        positions[nid] = NodePos(x=sub_x, y=y_cursor, w=w, h=node_h, rank=r)
                        y_cursor += node_h + h_gap
                    sub_x += max_w + h_gap
    else:
        # TB: ranks go top to bottom, nodes within rank go left to right
        y_cursor = 40
        for r in sorted(rank_groups.keys()):
            group = rank_groups[r]

            # Wrap into sub-rows if total width exceeds threshold
            sub_rows: list[list[str]] = [[]]
            row_w = 0
            for nid in group:
                w = node_widths.get(nid, 120)
                needed = w + (h_gap if row_w > 0 else 0)
                if row_w + needed > MAX_ROW_WIDTH and sub_rows[-1]:
                    sub_rows.append([])
                    row_w = 0
                sub_rows[-1].append(nid)
                row_w += w + h_gap

            for row in sub_rows:
                x_start = 40
                for nid in row:
                    w = node_widths.get(nid, 120)
                    positions[nid] = NodePos(x=x_start, y=y_cursor, w=w, h=node_h, rank=r)
                    x_start += w + h_gap
                y_cursor += node_h + v_gap

    # Compute canvas size
    if positions:
        max_x = max(p.x + p.w for p in positions.values()) + 40
        max_y = max(p.y + p.h for p in positions.values()) + 40
    else:
        max_x, max_y = 200, 100

    return positions, max_x, max_y


def _compute_sg_bounds(fc: MFlowchart, positions: dict[str, NodePos], sg_id: str) -> Optional[Rect]:
    """Compute bounding box for a subgraph's children."""
    sg = fc.subgraphs[sg_id]
    child_nodes = [n for n in sg.children if n in positions]
    if not child_nodes:
        return None

    min_x = min(positions[n].x for n in child_nodes) - REGION_PADDING
    min_y = min(positions[n].y for n in child_nodes) - REGION_TITLE_HEIGHT - REGION_PADDING
    max_x = max(positions[n].x + positions[n].w for n in child_nodes) + REGION_PADDING
    max_y = max(positions[n].y + positions[n].h for n in child_nodes) + REGION_PADDING

    return Rect(min_x, min_y, max_x - min_x, max_y - min_y)


# ---------------------------------------------------------------------------
# SVG generation: Flowchart
# ---------------------------------------------------------------------------


def render_flowchart(fc: MFlowchart, title: str, caption: str) -> str:
    """Generate academic-style SVG for a parsed flowchart."""
    positions, canvas_w, canvas_h = layout_flowchart(fc)

    # Compute subgraph bounds and adjust for regions
    sg_bounds = {}
    for sg_id in fc.sg_order:
        bounds = _compute_sg_bounds(fc, positions, sg_id)
        if bounds:
            sg_bounds[sg_id] = bounds

    # Adjust canvas for regions + title + caption
    extra_top = 50 if title else 10
    extra_bottom = 50 if caption else 20

    # Shift all positions down for title bar
    for nid, p in positions.items():
        p.y += extra_top

    # Recompute sg bounds after shift
    sg_bounds = {}
    for sg_id in fc.sg_order:
        bounds = _compute_sg_bounds(fc, positions, sg_id)
        if bounds:
            sg_bounds[sg_id] = bounds

    final_w = max(canvas_w + 40, 300)
    final_h = canvas_h + extra_top + extra_bottom

    # Also account for subgraph bounds that may extend beyond node positions
    for bounds in sg_bounds.values():
        final_w = max(final_w, bounds.x + bounds.w + 20)
        final_h = max(final_h, bounds.y + bounds.h + extra_bottom)

    c = SvgCanvas(width=final_w, height=final_h, title=title, caption=caption)

    # Draw subgraph regions
    for sg_id in fc.sg_order:
        if sg_id not in sg_bounds:
            continue
        sg = fc.subgraphs[sg_id]
        bounds = sg_bounds[sg_id]
        pal_name = palette_for_subgraph(sg.label)
        pal = PALETTES.get(pal_name, PALETTES["default"])

        c._elements.append(
            f'<rect x="{bounds.x}" y="{bounds.y}" width="{bounds.w}" height="{bounds.h}" '
            f'rx="2" fill="{pal["bg"]}" stroke="{pal["border"]}" stroke-width="1.2"/>'
        )
        c._elements.append(
            f'<text x="{bounds.cx}" y="{bounds.y + 15}" text-anchor="middle" '
            f'fill="{pal["title_color"]}" font-size="{REGION_TITLE_SIZE}" font-weight="bold">'
            f"{_esc_svg(sg.label)}</text>"
        )

    # Draw nodes
    box_map = {}
    for nid, node in fc.nodes.items():
        if nid not in positions:
            continue
        p = positions[nid]

        # Determine colors
        pal_name = "default"
        if node.container and node.container in fc.subgraphs:
            pal_name = palette_for_subgraph(fc.subgraphs[node.container].label)
        pal = PALETTES.get(pal_name, PALETTES["default"])

        # Use first line of label
        display_label = node.label.split("\n")[0] if node.label else nid

        if node.shape == "diamond":
            bi = c.diamond(p.x, p.y, max(p.w, p.h), display_label, pal_name)
        elif node.shape == "oval":
            # Render as rounded rect
            c._elements.append(
                f'<rect x="{p.x}" y="{p.y}" width="{p.w}" height="{p.h}" '
                f'rx="{p.h / 2}" fill="{pal["box_bg"]}" stroke="{pal["box_border"]}" '
                f'stroke-width="0.8"/>'
            )
            c._elements.append(
                f'<text x="{p.x + p.w / 2}" y="{p.y + p.h / 2 + 4}" text-anchor="middle" '
                f'fill="{pal["box_text"]}" font-size="{BOX_TEXT_SIZE}">{_esc_svg(display_label)}</text>'
            )
            bi = BoxInfo(rect=Rect(p.x, p.y, p.w, p.h), label=display_label)
        else:
            bi = c.box(p.x, p.y, p.w, p.h, display_label, palette=pal_name)

        box_map[nid] = bi

    # Draw edges
    horizontal = fc.direction in ("LR", "RL")
    for edge in fc.edges:
        if edge.src not in box_map or edge.dst not in box_map:
            continue
        src_bi = box_map[edge.src]
        dst_bi = box_map[edge.dst]
        src_p = positions[edge.src]
        dst_p = positions[edge.dst]

        # Determine arrow direction
        if horizontal:
            if dst_p.x > src_p.x:
                ss, ds = "right", "left"
            elif dst_p.x < src_p.x:
                ss, ds = "left", "right"
            elif dst_p.y > src_p.y:
                ss, ds = "bottom", "top"
            else:
                ss, ds = "top", "bottom"
        else:
            if dst_p.y > src_p.y:
                ss, ds = "bottom", "top"
            elif dst_p.y < src_p.y:
                ss, ds = "top", "bottom"
            elif dst_p.x > src_p.x:
                ss, ds = "right", "left"
            else:
                ss, ds = "left", "right"

        dashed = edge.style == "dashed"
        c.arrow(src_bi, dst_bi, label=edge.label, src_side=ss, dst_side=ds, dashed=dashed)

    return c.render()


# ---------------------------------------------------------------------------
# SVG generation: Sequence
# ---------------------------------------------------------------------------


def render_sequence(seq: MSequence, title: str, caption: str) -> str:
    """Generate academic-style SVG for a parsed sequence diagram."""
    # Estimate canvas size based on actor count and message count
    n_actors = len(seq.actors)
    actor_spacing = max(140, min(180, 900 // max(n_actors, 1)))

    # Count total events (including nested in blocks)
    def count_events(events):
        total = 0
        for e in events:
            if isinstance(e, SeqBlock):
                total += 1 + count_events(e.messages)
            else:
                total += 1
        return total

    n_events = count_events(seq.events)
    msg_spacing = 34

    canvas_w = max(n_actors * actor_spacing + 80, 400)
    canvas_h = max(n_events * msg_spacing + 150, 250)
    if title:
        canvas_h += 50
    if caption:
        canvas_h += 40

    c = SvgCanvas(width=canvas_w, height=canvas_h, title=title, caption=caption, framed=bool(title))

    y_start = 60 if title else 30

    # Assign palettes to actors based on name
    actor_dicts = []
    for a in seq.actors:
        pal = "default"
        for p_name, keywords in _PALETTE_KEYWORDS.items():
            for kw in keywords:
                if kw.lower() in a.label.lower() or kw.lower() in a.id.lower():
                    pal = p_name
                    break
            if pal != "default":
                break
        actor_dicts.append({"name": a.label, "short": a.id, "palette": pal})

    x_start = 60
    sd = c.sequence(actor_dicts, x=x_start, y=y_start, actor_spacing=actor_spacing, msg_spacing=msg_spacing)

    def render_events(events):
        for event in events:
            if isinstance(event, SeqMessage):
                sd.message(event.src, event.dst, event.label, dashed=event.dashed, self_msg=event.self_msg)
            elif isinstance(event, SeqNote):
                if event.actors:
                    sd.note(event.actors[0], event.text)
            elif isinstance(event, SeqBlock):
                sd.loop(f"{event.kind}: {event.label}")
                render_events(event.messages)
                sd.end_loop()

    render_events(seq.events)

    # Render actor boxes and lifelines
    actor_parts = sd.render_actors()
    for part in actor_parts:
        c._bg_elements.append(part)

    return c.render()


# ---------------------------------------------------------------------------
# SVG generation: State
# ---------------------------------------------------------------------------


def render_state(sd: MStateDiagram, title: str, caption: str) -> str:
    """Generate academic-style SVG for a parsed state diagram."""
    # Simple vertical layout for states
    states_list = [s for s in sd.states.values() if s.id not in ("_start", "_end")]
    n_states = len(states_list)

    state_w = 160
    state_h = 34
    v_gap = 50
    canvas_w = max(state_w + 120, 350)
    canvas_h = (n_states + 2) * (state_h + v_gap) + 100
    if title:
        canvas_h += 50
    if caption:
        canvas_h += 40

    c = SvgCanvas(width=canvas_w, height=canvas_h, title=title, caption=caption, framed=bool(title))

    st_diag = c.state_diagram()
    y_cursor = 70 if title else 30
    cx = canvas_w / 2

    # Start dot
    if "_start" in sd.states:
        st_diag.start_dot(cx, y_cursor)
        y_cursor += 40

    # States
    state_positions = {}
    for state in states_list:
        w = max(estimate_text_width(state.label, BOX_TEXT_SIZE) + 40, state_w)
        st_diag.state(state.id, state.label, cx - w / 2, y_cursor, w, state_h, "rollout")
        state_positions[state.id] = y_cursor
        y_cursor += state_h + v_gap

    # End dot
    if "_end" in sd.states:
        st_diag.end_dot(cx, y_cursor)
        y_cursor += 30

    # Transitions
    for t in sd.transitions:
        st_diag.transition(t.src, t.dst, t.label)

    return c.render()


# ---------------------------------------------------------------------------
# SVG generation: Class diagram (simplified)
# ---------------------------------------------------------------------------


def render_class(source: str, title: str, caption: str) -> str:
    """Generate academic-style SVG for a class diagram."""
    lines = source.strip().split("\n")

    classes = {}
    current_class = None
    relationships = []

    for raw in lines:
        line = raw.strip()
        if not line or line == "classDiagram" or line.startswith("%%"):
            continue

        # Class definition
        cm = re.match(r"class\s+(\w+)\s*\{?", line)
        if cm:
            current_class = cm.group(1)
            classes.setdefault(current_class, {"methods": [], "attrs": []})
            continue

        if line == "}":
            current_class = None
            continue

        # Member inside class block
        if current_class and current_class in classes:
            if "(" in line:
                classes[current_class]["methods"].append(line.strip().lstrip("+- "))
            else:
                classes[current_class]["attrs"].append(line.strip().lstrip("+- "))
            continue

        # Relationship
        rm = re.match(r"(\w+)\s*(--|<\|--|\.\.>|-->|--\*|--o)\s*(\w+)\s*(?::\s*(.+))?", line)
        if rm:
            relationships.append((rm.group(1), rm.group(3), rm.group(4) or ""))
            for cls_name in [rm.group(1), rm.group(3)]:
                classes.setdefault(cls_name, {"methods": [], "attrs": []})
            continue

    # Layout: simple grid
    n_classes = len(classes)
    cols = min(n_classes, 3)
    class_w = 200
    class_gap = 40
    canvas_w = cols * (class_w + class_gap) + 60
    y_start = 60 if title else 20

    c = SvgCanvas(width=max(canvas_w, 400), height=600, title=title, caption=caption, framed=bool(title))

    positions_map = {}
    for i, (cls_name, cls_data) in enumerate(classes.items()):
        col = i % cols
        row = i // cols
        x = 40 + col * (class_w + class_gap)
        y = y_start + row * 200

        n_members = len(cls_data["attrs"]) + len(cls_data["methods"])
        cls_h = 28 + n_members * 16 + 10
        pal = PALETTES["rollout"]

        # Class header
        c._elements.append(
            f'<rect x="{x}" y="{y}" width="{class_w}" height="{cls_h}" '
            f'rx="2" fill="{pal["bg"]}" stroke="{pal["border"]}" stroke-width="1.2"/>'
        )
        c._elements.append(
            f'<rect x="{x}" y="{y}" width="{class_w}" height="24" '
            f'rx="2" fill="{pal["border"]}" stroke="{pal["border"]}" stroke-width="1.2"/>'
        )
        c._elements.append(
            f'<text x="{x + class_w / 2}" y="{y + 16}" text-anchor="middle" '
            f'fill="#fff" font-size="{BOX_BOLD_SIZE}" font-weight="bold">{_esc_svg(cls_name)}</text>'
        )

        # Members
        my = y + 30
        for attr in cls_data["attrs"]:
            c._elements.append(f'<text x="{x + 10}" y="{my}" fill="#424242" font-size="{LABEL_SIZE}">' f"{_esc_svg(attr)}</text>")
            my += 18
        if cls_data["attrs"] and cls_data["methods"]:
            c._elements.append(
                f'<line x1="{x}" y1="{my - 4}" x2="{x + class_w}" y2="{my - 4}" ' f'stroke="{pal["box_border"]}" stroke-width="0.5"/>'
            )
            my += 4
        for method in cls_data["methods"]:
            c._elements.append(f'<text x="{x + 10}" y="{my}" fill="#424242" font-size="{LABEL_SIZE}">' f"{_esc_svg(method)}</text>")
            my += 18

        positions_map[cls_name] = Rect(x, y, class_w, cls_h)

    # Draw relationships
    for src, dst, label in relationships:
        if src in positions_map and dst in positions_map:
            p1 = positions_map[src].port("bottom")
            p2 = positions_map[dst].port("top")
            c.arrow_between(p1, p2, label=label)

    return c.render()


# ---------------------------------------------------------------------------
# SVG generation: Gantt
# ---------------------------------------------------------------------------


def render_gantt(source: str, title: str, caption: str) -> str:
    """Generate academic-style SVG for a gantt chart."""
    lines = source.strip().split("\n")

    gantt_title = ""
    date_format = ""
    sections = []
    current_section = None
    tasks = []

    for raw in lines:
        line = raw.strip()
        if not line or line == "gantt" or line.startswith("%%"):
            continue
        if line.startswith("title"):
            gantt_title = line.replace("title", "").strip()
            continue
        if line.startswith("dateFormat"):
            continue
        if line.startswith("axisFormat"):
            continue
        if line.startswith("section"):
            current_section = line.replace("section", "").strip()
            sections.append(current_section)
            continue
        # Task line: TaskName :status, id, start, duration
        parts = line.split(":")
        if len(parts) >= 2:
            task_name = parts[0].strip()
            tasks.append({"name": task_name, "section": current_section})

    # Simple layout
    canvas_w = 700
    row_h = 28
    section_h = 24
    y_start = 70 if title else 30
    canvas_h = y_start + len(tasks) * row_h + len(sections) * section_h + 60

    c = SvgCanvas(width=canvas_w, height=canvas_h, title=title or gantt_title, caption=caption, framed=bool(title or gantt_title))

    y = y_start
    bar_x = 180
    bar_w = canvas_w - bar_x - 60
    current_sec = None
    task_i = 0

    for task in tasks:
        if task["section"] != current_sec:
            current_sec = task["section"]
            c._elements.append(
                f'<rect x="30" y="{y}" width="{canvas_w - 80}" height="{section_h}" '
                f'rx="2" fill="#F5F5F5" stroke="#E0E0E0" stroke-width="0.8"/>'
            )
            c._elements.append(
                f'<text x="40" y="{y + 16}" fill="#616161" font-size="{BOX_TEXT_SIZE}" '
                f'font-weight="bold">{_esc_svg(current_sec or "")}</text>'
            )
            y += section_h + 4

        # Task bar
        segment_w = bar_w / max(len(tasks), 1)
        bx = bar_x + task_i * segment_w * 0.3
        bw = max(segment_w * 0.8, 40)
        pal = PALETTES["rollout"]
        c._elements.append(
            f'<text x="{bar_x - 10}" y="{y + 18}" text-anchor="end" '
            f'fill="#424242" font-size="{LABEL_SIZE}">{_esc_svg(task["name"])}</text>'
        )
        c._elements.append(
            f'<rect x="{bx}" y="{y + 4}" width="{bw}" height="{row_h - 8}" ' f'rx="3" fill="{pal["border"]}" stroke="none"/>'
        )
        y += row_h
        task_i += 1

    return c.render()


# ---------------------------------------------------------------------------
# Filename convention (matches old SVG engine)
# ---------------------------------------------------------------------------


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "_", text)
    return text[:60].rstrip("_")


def _esc_svg(text: str) -> str:
    import html as _html

    return _html.escape(text, quote=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    DIAGRAMS_DIR.mkdir(parents=True, exist_ok=True)

    src = Path("/tmp/mermaid_diagrams.json")
    if not src.exists():
        print(f"ERROR: {src} not found")
        sys.exit(1)

    with open(src) as f:
        all_diagrams = json.load(f)

    success = 0
    failed = 0

    for d in all_diagrams:
        rel = d["file"]
        idx = d["index"]
        dtype = d["type"]
        source = d["source"]

        page_slug = slugify(rel.replace("/", "").replace(".md", ""))
        svg_name = f"{page_slug}_{idx}.svg"
        svg_path = DIAGRAMS_DIR / svg_name

        # Build title/caption from file context
        page_title = rel.replace("/", " > ").replace(".md", "").replace("_", " ").title()
        title_text = f"{page_title} (Fig. {idx})"
        caption_text = f"Figure {idx}"

        print(f"  {rel} [{idx}] ({dtype}): {svg_name}", end=" ")

        try:
            if dtype == "flowchart":
                fc = parse_flowchart(source)
                svg_text = render_flowchart(fc, title=title_text, caption=caption_text)
            elif dtype == "sequence":
                seq = parse_sequence(source)
                svg_text = render_sequence(seq, title=title_text, caption=caption_text)
            elif dtype == "state":
                sd = parse_state(source)
                svg_text = render_state(sd, title=title_text, caption=caption_text)
            elif dtype == "class":
                svg_text = render_class(source, title=title_text, caption=caption_text)
            elif dtype == "gantt":
                svg_text = render_gantt(source, title=title_text, caption=caption_text)
            else:
                print(f"✗ Unknown type: {dtype}")
                failed += 1
                continue

            with open(svg_path, "w") as f:
                f.write(svg_text)

            size_kb = len(svg_text) / 1024
            print(f"✓ ({size_kb:.1f} KB)")
            success += 1

        except Exception as e:
            print(f"✗ {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"Generated: {success}  |  Failed: {failed}  |  Total: {len(all_diagrams)}")
    old_total = sum(f.stat().st_size for f in DIAGRAMS_DIR.glob("*.svg"))
    print(f"Total SVG size: {old_total / 1024:.0f} KB")


if __name__ == "__main__":
    main()
