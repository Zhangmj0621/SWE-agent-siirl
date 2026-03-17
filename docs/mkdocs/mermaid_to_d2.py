#!/usr/bin/env python3
"""Convert Mermaid diagram sources to D2 diagram language.

Supports 5 diagram types:
  1. Flowchart (graph/flowchart TD/TB/LR/RL/BT)
  2. Sequence (sequenceDiagram)
  3. State (stateDiagram-v2)
  4. Class (classDiagram)
  5. Gantt (gantt)

Usage:
    python mermaid_to_d2.py
    Reads /tmp/mermaid_diagrams.json, writes .d2 files to /tmp/d2_output/
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "_", text)
    return text[:60].rstrip("_")


def clean_label(label: str) -> str:
    """Replace <br/>, <br>, strip other HTML tags, trim quotes."""
    label = label.replace("<br/>", "\\n").replace("<br>", "\\n")
    label = re.sub(r"</?[a-zA-Z][^>]*>", "", label)
    # Strip surrounding quotes if present
    if len(label) >= 2 and label[0] == '"' and label[-1] == '"':
        label = label[1:-1]
    return label.strip()


def escape_d2_label(label: str) -> str:
    """Escape a label for D2 -- wrap in quotes if it contains special chars."""
    if not label:
        return '""'
    # If label contains characters that need quoting in D2
    needs_quoting = any(c in label for c in "{}()[]<>:;|#&@$!?=+/\\'\n") or "\\n" in label
    if needs_quoting:
        escaped = label.replace("\\", "\\\\").replace('"', '\\"')
        # Keep intentional \\n as actual escaped newline
        escaped = escaped.replace("\\\\n", "\\n")
        return f'"{escaped}"'
    return label


def d2_id(node_id: str) -> str:
    """Ensure a node ID is valid in D2."""
    return node_id


# ---------------------------------------------------------------------------
# Flowchart types
# ---------------------------------------------------------------------------


@dataclass
class FlowNode:
    node_id: str
    label: str
    shape: Optional[str] = None
    container: Optional[str] = None


@dataclass
class FlowEdge:
    src: str
    dst: str
    label: str = ""
    style: str = ""  # "dashed", "thick", "invisible", "bidirectional", "reverse", ""


@dataclass
class FlowSubgraph:
    sg_id: str
    label: str
    parent: Optional[str] = None
    direction: Optional[str] = None
    children_ids: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Flowchart converter
# ---------------------------------------------------------------------------

_DIRECTION_MAP = {
    "TB": "down",
    "TD": "down",
    "LR": "right",
    "RL": "left",
    "BT": "up",
}


def _parse_node_def(token: str) -> tuple[str, str, Optional[str]]:
    """Parse a mermaid node token like A["Text"], A{Text}, A(("Text")), etc.

    Returns (node_id, label, shape_or_None).
    """
    token = token.strip()
    if not token:
        return ("", "", None)

    # Try various bracket patterns (order matters -- longest first)
    patterns = [
        # A[("Text")] -- cylinder
        (r'^(\w+)\[\("(.*)"\)\]$', "cylinder"),
        # A(("Text")) -- circle
        (r'^(\w+)\(\("(.*)"\)\)$', "circle"),
        (r"^(\w+)\(\((.+)\)\)$", "circle"),
        # A[["Text"]] -- double bracket
        (r'^(\w+)\[\["(.*)"\]\]$', None),
        (r"^(\w+)\[\[(.+)\]\]$", None),
        # A[/"Text"/] -- parallelogram
        (r'^(\w+)\[/"(.*)"/\]$', "parallelogram"),
        (r"^(\w+)\[/(.+)/\]$", "parallelogram"),
        # A>"Text"] -- asymmetric
        (r'^(\w+)>"(.*)"\]$', None),
        # A{{"Text"}} -- hexagon
        (r'^(\w+)\{\{"(.*)"\}\}$', "hexagon"),
        (r"^(\w+)\{\{(.+)\}\}$", "hexagon"),
        # A{"Text"} -- diamond
        (r'^(\w+)\{"(.*)"\}$', "diamond"),
        (r"^(\w+)\{(.+)\}$", "diamond"),
        # A("Text") -- stadium/oval
        (r'^(\w+)\("(.*)"\)$', "oval"),
        (r"^(\w+)\((.+)\)$", "oval"),
        # A["Text"] -- rectangle (default)
        (r'^(\w+)\["(.*)"\]$', None),
        (r"^(\w+)\[(.+)\]$", None),
        # Plain id
        (r"^(\w+)$", None),
    ]

    for pat, shape in patterns:
        m = re.match(pat, token, re.DOTALL)
        if m:
            node_id = m.group(1)
            label = m.group(2) if m.lastindex >= 2 else node_id
            return (node_id, clean_label(label), shape)

    # Fallback: use the whole token as id
    return (token, token, None)


@dataclass
class _EdgeParseResult:
    """Result of parsing an edge line, including node definitions found."""

    edges: list[tuple[str, str, str, str]]  # (src_id, dst_id, style, label)
    node_defs: list[tuple[str, str, Optional[str]]]  # (node_id, label, shape)


def _split_edge_line(line: str) -> list[tuple[str, str, str, str]]:
    """Split a line that may contain chained edges into individual edges.

    Returns list of (src_token, dst_token, edge_style, label).
    Handles: A --> B --> C, A -->|"label"| B --> C,
             A & B --> C (parallel sources), A --> B & C (parallel targets).
    """
    result = _parse_edge_line_full(line)
    return result.edges


def _parse_edge_line_full(line: str) -> _EdgeParseResult:
    """Parse an edge line, returning both edges and node definitions."""
    edges: list[tuple[str, str, str, str]] = []
    node_defs: list[tuple[str, str, Optional[str]]] = []

    # Extract labels from edge operators first, keep the operator.
    label_map: dict[str, str] = {}
    counter = [0]

    def _replace_label(m):
        key = f"__LABEL{counter[0]}__"
        counter[0] += 1
        label_map[key] = m.group(2)
        return m.group(1) + " " + key

    processed = re.sub(
        r'(-->|-.->|==>|<-->|---)\|"?([^"|]*)"?\|',
        _replace_label,
        line,
    )

    # Edge operators (order: longest first for correct matching)
    edge_pattern = r"(<-->|-.->|==>|-->|---|<--)"

    # Split on edge operators
    parts = re.split(edge_pattern, processed)

    if len(parts) < 3:
        return _EdgeParseResult(edges=edges, node_defs=node_defs)

    i = 0
    while i + 2 < len(parts):
        src_token = parts[i].strip()
        edge_raw = parts[i + 1].strip()
        dst_token = parts[i + 2].strip()

        # Recover labels from placeholders
        label = ""
        for key, val in label_map.items():
            if key in dst_token:
                label = clean_label(val)
                dst_token = dst_token.replace(key, "").strip()
                break
            if key in src_token:
                label = clean_label(val)
                src_token = src_token.replace(key, "").strip()
                break

        # Determine style from edge operator
        style = ""
        if edge_raw == "-.->":
            style = "dashed"
        elif edge_raw == "==>":
            style = "thick"
        elif edge_raw == "<-->":
            style = "bidirectional"
        elif edge_raw == "---":
            style = "invisible"
        elif edge_raw == "<--":
            style = "reverse"

        # Handle & (parallel connections) in src and dst tokens
        src_tokens = _split_ampersand(src_token)
        dst_tokens = _split_ampersand(dst_token)

        for st in src_tokens:
            s_id, s_label, s_shape = _parse_node_def(st) if st else ("", "", None)
            if s_id:
                node_defs.append((s_id, s_label, s_shape))
            for dt in dst_tokens:
                d_id, d_label, d_shape = _parse_node_def(dt) if dt else ("", "", None)
                if d_id and i == 0:
                    # Only add dst node_defs once (not for every src in &)
                    pass
                if s_id and d_id:
                    edges.append((s_id, d_id, style, label))

        # Add dst node defs separately
        for dt in dst_tokens:
            d_id, d_label, d_shape = _parse_node_def(dt) if dt else ("", "", None)
            if d_id:
                node_defs.append((d_id, d_label, d_shape))

        i += 2

    return _EdgeParseResult(edges=edges, node_defs=node_defs)


def _split_ampersand(token: str) -> list[str]:
    """Split a token on ' & ' to handle mermaid parallel connection syntax.

    Only splits at ' & ' that appears outside of brackets ([], (), {}, "").
    For example, 'A & B' splits to ['A', 'B'], but
    'F1["Check env & logs"]' does NOT split.
    """
    if " & " not in token:
        return [token] if token else []

    # Check if the & is inside brackets by tracking nesting
    # Only split at positions where bracket depth is 0 and we're not inside quotes
    split_positions: list[int] = []
    depth = 0
    in_quotes = False
    i = 0
    while i < len(token):
        c = token[i]
        if c == '"' and (i == 0 or token[i - 1] != "\\"):
            in_quotes = not in_quotes
        elif not in_quotes:
            if c in "([{":
                depth += 1
            elif c in ")]}":
                depth = max(0, depth - 1)
            elif depth == 0 and token[i : i + 3] == " & ":
                split_positions.append(i)
        i += 1

    if not split_positions:
        return [token] if token else []

    # Split at the found positions
    parts: list[str] = []
    prev = 0
    for pos in split_positions:
        part = token[prev:pos].strip()
        if part:
            parts.append(part)
        prev = pos + 3  # skip " & "
    remaining = token[prev:].strip()
    if remaining:
        parts.append(remaining)

    if not parts:
        return [token] if token else []
    return parts


def convert_flowchart(source: str) -> str:
    """Convert a mermaid flowchart to D2."""
    lines = source.split("\n")
    output_lines: list[str] = []
    notes: list[str] = []

    # Parsed structures
    nodes: dict[str, FlowNode] = {}
    edges: list[FlowEdge] = []
    subgraphs: dict[str, FlowSubgraph] = {}
    subgraph_order: list[str] = []  # preserve insertion order
    subgraph_stack: list[str] = []  # stack of subgraph ids for nesting

    # Top-level direction
    top_direction = "down"

    # Parse the header line
    first_line = lines[0].strip()
    header_match = re.match(r"(?:graph|flowchart)\s+(TD|TB|LR|RL|BT)", first_line)
    if header_match:
        top_direction = _DIRECTION_MAP.get(header_match.group(1), "down")

    def _current_container() -> Optional[str]:
        return subgraph_stack[-1] if subgraph_stack else None

    def _register_node(
        node_id: str,
        label: str,
        shape: Optional[str],
        container: Optional[str] = None,
    ):
        if not node_id:
            return
        if node_id not in nodes:
            nodes[node_id] = FlowNode(
                node_id=node_id,
                label=label,
                shape=shape,
                container=container,
            )
        else:
            # Update label/shape if this is a richer definition
            if label and label != node_id and nodes[node_id].label == node_id:
                nodes[node_id].label = label
            if shape and not nodes[node_id].shape:
                nodes[node_id].shape = shape
            # Don't override container if already set
            if container and not nodes[node_id].container:
                nodes[node_id].container = container

    # Parse lines
    for raw_line in lines[1:]:
        line = raw_line.strip()
        if not line or line.startswith("%%"):
            continue

        # Subgraph start
        if line.startswith("subgraph "):
            sg_id = None
            sg_label = None

            # subgraph id["Label"]
            m = re.match(r'subgraph\s+(\w+)\["(.*)"\]', line)
            if m:
                sg_id, sg_label = m.group(1), clean_label(m.group(2))
            else:
                # subgraph id[Label]
                m = re.match(r"subgraph\s+(\w+)\[(.+)\]", line)
                if m:
                    sg_id, sg_label = m.group(1), clean_label(m.group(2))
                else:
                    # subgraph "Label"
                    m = re.match(r'subgraph\s+"(.+)"', line)
                    if m:
                        sg_label = clean_label(m.group(1))
                        sg_id = slugify(sg_label)[:30]
                    else:
                        # subgraph id
                        m = re.match(r"subgraph\s+(\w+)\s*$", line)
                        if m:
                            sg_id = m.group(1)
                            sg_label = sg_id

            if sg_id:
                parent = _current_container()
                subgraphs[sg_id] = FlowSubgraph(sg_id=sg_id, label=sg_label or sg_id, parent=parent)
                subgraph_order.append(sg_id)
                subgraph_stack.append(sg_id)
            continue

        if line == "end":
            if subgraph_stack:
                subgraph_stack.pop()
            continue

        # Direction inside subgraph
        dir_match = re.match(r"direction\s+(TB|TD|LR|RL|BT)", line)
        if dir_match:
            if subgraph_stack:
                sg_id = subgraph_stack[-1]
                subgraphs[sg_id].direction = _DIRECTION_MAP.get(dir_match.group(1), None)
            continue

        # Style/class/click lines -- skip
        if line.startswith("style ") or line.startswith("class ") or line.startswith("click "):
            continue

        container = _current_container()

        # Check if line contains an edge operator
        has_edge = bool(re.search(r"(<-->|-.->|==>|-->|---|<--)", line))

        if has_edge:
            parse_result = _parse_edge_line_full(line)
            for src_id, dst_id, style, label in parse_result.edges:
                edges.append(FlowEdge(src=src_id, dst=dst_id, label=label, style=style))
                # Register nodes from edge lines with current container context
                _register_node(src_id, src_id, None, container)
                _register_node(dst_id, dst_id, None, container)

            # Register full node definitions (with labels/shapes) from tokens
            for nid, nlabel, nshape in parse_result.node_defs:
                _register_node(nid, nlabel, nshape, container)
        else:
            # Pure node definition line
            node_id, label, shape = _parse_node_def(line)
            if node_id:
                _register_node(node_id, label, shape, container)
                if container and container in subgraphs:
                    if node_id not in subgraphs[container].children_ids:
                        subgraphs[container].children_ids.append(node_id)

    # Attach nodes to their subgraph children lists
    for nid, node in nodes.items():
        if node.container and node.container in subgraphs:
            sg = subgraphs[node.container]
            if nid not in sg.children_ids:
                sg.children_ids.append(nid)

    # Build the qualified-name lookup: node_id -> "container.node_id"
    def _qualified_name(node_id: str) -> str:
        """Build the full qualified path for a node, handling nesting."""
        if node_id in subgraphs:
            sg = subgraphs[node_id]
            if sg.parent:
                return f"{_qualified_name(sg.parent)}.{node_id}"
            return node_id
        if node_id in nodes and nodes[node_id].container:
            container = nodes[node_id].container
            return f"{_qualified_name(container)}.{node_id}"
        return node_id

    # ---- Generate D2 output ----
    output_lines.append(f"direction: {top_direction}")
    output_lines.append("")

    rendered_nodes: set[str] = set()

    def _render_node(node: FlowNode, indent: int = 0):
        prefix = "  " * indent
        label = node.label if node.label != node.node_id else node.node_id
        label_str = escape_d2_label(label)
        if node.shape:
            output_lines.append(f"{prefix}{node.node_id}: {label_str} {{")
            output_lines.append(f"{prefix}  shape: {node.shape}")
            output_lines.append(f"{prefix}}}")
        elif label == node.node_id:
            output_lines.append(f"{prefix}{node.node_id}")
        else:
            output_lines.append(f"{prefix}{node.node_id}: {label_str}")

    def _edge_attrs(edge: FlowEdge) -> str:
        """Build D2 edge attributes string."""
        style_parts: list[str] = []

        if edge.style == "dashed":
            style_parts.append("style.stroke-dash: 3")
        elif edge.style == "thick":
            style_parts.append("style.stroke-width: 3")
        elif edge.style == "invisible":
            style_parts.append("style.opacity: 0.3")

        label_str = escape_d2_label(edge.label) if edge.label else ""

        if style_parts and label_str:
            return f"{label_str} {{ {'; '.join(style_parts)} }}"
        elif style_parts:
            return "{ " + "; ".join(style_parts) + " }"
        elif label_str:
            return label_str
        return ""

    def _render_edge_local(edge: FlowEdge, indent: int = 0):
        """Render an edge using local (unqualified) names -- inside subgraphs."""
        prefix = "  " * indent
        edge_str = f"{prefix}{edge.src} -> {edge.dst}"
        attrs = _edge_attrs(edge)
        if attrs:
            edge_str += f": {attrs}"
        output_lines.append(edge_str)

    def _render_subgraph(sg_id: str, indent: int = 0):
        sg = subgraphs[sg_id]
        prefix = "  " * indent
        label_str = escape_d2_label(sg.label)
        output_lines.append(f"{prefix}{sg_id}: {label_str} {{")

        if sg.direction:
            output_lines.append(f"{prefix}  direction: {sg.direction}")

        # Render child subgraphs
        child_sgs = [sid for sid, s in subgraphs.items() if s.parent == sg_id]
        for child_sg_id in child_sgs:
            _render_subgraph(child_sg_id, indent + 1)

        # Render child nodes
        for nid in sg.children_ids:
            if nid in subgraphs:
                continue  # Already rendered as sub-subgraph
            if nid in rendered_nodes:
                continue
            rendered_nodes.add(nid)
            node = nodes.get(nid)
            if node:
                _render_node(node, indent + 1)

        # Render edges that are internal to this subgraph
        for edge in edges:
            src_container = nodes.get(edge.src, FlowNode("", "")).container
            dst_container = nodes.get(edge.dst, FlowNode("", "")).container
            if src_container == sg_id and dst_container == sg_id:
                _render_edge_local(edge, indent + 1)

        output_lines.append(f"{prefix}}}")
        output_lines.append("")

    # Render top-level subgraphs
    top_level_sgs = [sid for sid in subgraph_order if subgraphs[sid].parent is None]
    for sg_id in top_level_sgs:
        _render_subgraph(sg_id)

    # Render top-level nodes (not in any subgraph)
    for nid, node in nodes.items():
        if node.container is None and nid not in rendered_nodes and nid not in subgraphs:
            rendered_nodes.add(nid)
            _render_node(node)

    # Render edges that cross subgraph boundaries or are at top level
    output_lines.append("")
    rendered_edges: set[tuple] = set()
    for edge in edges:
        src_container = nodes.get(edge.src, FlowNode("", "")).container
        dst_container = nodes.get(edge.dst, FlowNode("", "")).container

        # Skip edges already rendered inside subgraphs
        if src_container and dst_container and src_container == dst_container:
            continue

        src_qual = _qualified_name(edge.src)
        dst_qual = _qualified_name(edge.dst)

        if edge.style == "bidirectional":
            edge_str = f"{src_qual} <-> {dst_qual}"
        elif edge.style == "reverse":
            edge_str = f"{dst_qual} -> {src_qual}"
        else:
            edge_str = f"{src_qual} -> {dst_qual}"

        attrs = _edge_attrs(edge)
        if attrs:
            edge_str += f": {attrs}"

        edge_key = (src_qual, dst_qual, edge.label, edge.style)
        if edge_key not in rendered_edges:
            rendered_edges.add(edge_key)
            output_lines.append(edge_str)

    for note in notes:
        output_lines.append(f"# CONVERSION NOTE: {note}")

    return "\n".join(output_lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Sequence diagram converter
# ---------------------------------------------------------------------------


def convert_sequence(source: str) -> str:
    """Convert a mermaid sequence diagram to D2."""
    lines = source.split("\n")
    output_lines: list[str] = ["shape: sequence_diagram"]
    notes: list[str] = []

    participants: dict[str, str] = {}  # id -> alias
    participant_order: list[str] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line == "sequenceDiagram":
            continue

        # Skip rect/end blocks (coloring)
        if line.startswith("rect ") or line == "end":
            continue

        # Participant
        m = re.match(r"participant\s+(\w+)\s+as\s+(.+)", line)
        if m:
            pid, alias = m.group(1), clean_label(m.group(2))
            participants[pid] = alias
            participant_order.append(pid)
            continue

        # Actor
        m = re.match(r"actor\s+(\w+)\s+as\s+(.+)", line)
        if m:
            pid, alias = m.group(1), clean_label(m.group(2))
            participants[pid] = alias
            participant_order.append(pid)
            continue

        # Note over ...
        if line.startswith("Note over") or line.startswith("Note "):
            note_text = re.sub(r"Note\s+(?:over|right of|left of)\s+\S+:?\s*", "", line)
            if note_text:
                notes.append(f"Note: {note_text}")
            continue

        # loop / alt / else / par blocks
        if line.startswith("loop "):
            notes.append(f"Loop: {line[5:]}")
            continue
        if line.startswith("alt "):
            notes.append(f"Alt: {line[4:]}")
            continue
        if line.startswith("else"):
            notes.append(f"Else: {line[5:] if len(line) > 5 else ''}")
            continue
        if line.startswith("par "):
            notes.append(f"Parallel: {line[4:]}")
            continue

        # Message: A->>B: text, A-->>B: text, etc.
        m = re.match(
            r"(\w+)\s*(->>|-->>|-\)|--\)|->|-->)\+?\s*(\w+)\s*:\s*(.*)",
            line,
        )
        if m:
            src, arrow, dst, msg = (
                m.group(1),
                m.group(2),
                m.group(3),
                clean_label(m.group(4)),
            )
            is_dashed = arrow.startswith("--")
            msg_str = escape_d2_label(msg) if msg else ""

            if is_dashed and msg_str:
                output_lines.append(f"{src} -> {dst}: {msg_str} {{")
                output_lines.append("  style.stroke-dash: 3")
                output_lines.append("}")
            elif is_dashed:
                output_lines.append(f"{src} -> {dst}: {{")
                output_lines.append("  style.stroke-dash: 3")
                output_lines.append("}")
            elif msg_str:
                output_lines.append(f"{src} -> {dst}: {msg_str}")
            else:
                output_lines.append(f"{src} -> {dst}")
            continue

    # Build final output: participants right after shape declaration
    result = [output_lines[0]]  # shape: sequence_diagram
    for pid in participant_order:
        alias = participants[pid]
        result.append(f"{pid}: {escape_d2_label(alias)}")
    result.extend(output_lines[1:])

    for note in notes:
        result.append(f"# CONVERSION NOTE: {note}")

    return "\n".join(result).rstrip() + "\n"


# ---------------------------------------------------------------------------
# State diagram converter
# ---------------------------------------------------------------------------


def convert_state(source: str) -> str:
    """Convert a mermaid stateDiagram-v2 to D2."""
    lines = source.split("\n")
    output_lines: list[str] = []
    notes: list[str] = []

    states: dict[str, str] = {}  # id -> label
    transitions: list[tuple[str, str, str]] = []  # (src, dst, label)
    state_block_lines: dict[str, list] = {}
    in_note = False
    note_lines: list[str] = []
    current_state_block: Optional[str] = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line == "stateDiagram-v2":
            continue

        # Multi-line note
        if in_note:
            if line == "end note":
                in_note = False
                note_text = " ".join(note_lines)
                notes.append(note_text)
                note_lines = []
            else:
                note_lines.append(line)
            continue

        # Note start
        if line.startswith("note "):
            in_note = True
            note_lines = []
            continue

        # State block
        m = re.match(r"state\s+(\w+)\s*\{", line)
        if m:
            current_state_block = m.group(1)
            state_block_lines[current_state_block] = []
            states[current_state_block] = current_state_block
            continue

        if line == "}" and current_state_block:
            current_state_block = None
            continue

        # Transition: [*] --> STATE: label or STATE --> STATE: label
        m = re.match(r"(\[\*\]|\w+)\s*-->\s*(\[\*\]|\w+)\s*(?::\s*(.+))?", line)
        if m:
            src, dst, label = m.group(1), m.group(2), m.group(3) or ""
            label = clean_label(label)

            if src == "[*]":
                src = "start"
                states["start"] = ""
            else:
                if src not in states:
                    states[src] = src
            if dst == "[*]":
                dst = "end_state"
                states["end_state"] = ""
            else:
                if dst not in states:
                    states[dst] = dst

            if current_state_block:
                state_block_lines[current_state_block].append((src, dst, label))
            else:
                transitions.append((src, dst, label))
            continue

        # Plain state declaration inside a block
        if current_state_block and line and not line.startswith("%%"):
            m2 = re.match(r"(\w+)\s*$", line)
            if m2:
                state_name = m2.group(1)
                states[state_name] = state_name

    # Generate D2 output

    # Special start/end states
    if "start" in states:
        output_lines.append('start: "" {')
        output_lines.append("  shape: circle")
        output_lines.append('  style.fill: "#000"')
        output_lines.append("}")
    if "end_state" in states:
        output_lines.append('end_state: "" {')
        output_lines.append("  shape: circle")
        output_lines.append('  style.fill: "#000"')
        output_lines.append("}")

    # Regular states
    for sid, label in states.items():
        if sid in ("start", "end_state"):
            continue

        if sid in state_block_lines:
            # Composite state
            output_lines.append(f"{sid}: {escape_d2_label(label)} {{")
            child_states: set[str] = set()
            for item in state_block_lines[sid]:
                if isinstance(item, tuple):
                    s, d, _ = item
                    child_states.add(s)
                    child_states.add(d)
            for cs in sorted(child_states):
                if cs not in ("start", "end_state"):
                    output_lines.append(f"  {cs}")
            for item in state_block_lines[sid]:
                if isinstance(item, tuple):
                    s, d, l = item
                    edge_line = f"  {s} -> {d}"
                    if l:
                        edge_line += f": {escape_d2_label(l)}"
                    output_lines.append(edge_line)
            output_lines.append("}")
        else:
            output_lines.append(f"{sid}: {escape_d2_label(label)}")

    output_lines.append("")

    # Transitions
    for src, dst, label in transitions:
        edge_line = f"{src} -> {dst}"
        if label:
            edge_line += f": {escape_d2_label(label)}"
        output_lines.append(edge_line)

    for note in notes:
        output_lines.append(f"# CONVERSION NOTE: {note}")

    return "\n".join(output_lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Class diagram converter
# ---------------------------------------------------------------------------


def convert_class(source: str) -> str:
    """Convert a mermaid classDiagram to D2."""
    lines = source.split("\n")
    output_lines: list[str] = []
    notes: list[str] = []

    classes: dict[str, dict] = {}
    class_order: list[str] = []
    relationships: list[tuple[str, str, str, str]] = []

    current_class: Optional[str] = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line == "classDiagram":
            continue

        # Class block start
        m = re.match(r"class\s+(\w+)(?:~\w+~)?\s*\{", line)
        if m:
            cls_name = m.group(1)
            current_class = cls_name
            if cls_name not in classes:
                classes[cls_name] = {"stereotype": None, "members": []}
                class_order.append(cls_name)
            continue

        # Class block end
        if line == "}" and current_class:
            current_class = None
            continue

        # Inside class block
        if current_class:
            # Stereotype
            m = re.match(r"<<(\w+)>>", line)
            if m:
                classes[current_class]["stereotype"] = m.group(1)
                continue

            # Member or method -- remove visibility markers
            member_line = re.sub(r"^[+\-#~]\s*", "", line)
            if member_line:
                member_line = member_line.rstrip("*").strip()
                classes[current_class]["members"].append(member_line)
            continue

        # Relationship lines
        rel_patterns = [
            (r"(\w+)\s+<\|--\s+(\w+)(?:\s*:\s*(.+))?", "inheritance"),
            (r"(\w+)\s+<\|\.\.?\s+(\w+)(?:\s*:\s*(.+))?", "realization"),
            (r"(\w+)\s+\*--\s+(\w+)(?:\s*:\s*(.+))?", "composition"),
            (r"(\w+)\s+o--\s+(\w+)(?:\s*:\s*(.+))?", "aggregation"),
            (r"(\w+)\s+-->\s+(\w+)(?:\s*:\s*(.+))?", "association"),
            (r"(\w+)\s+\.\.>\s+(\w+)(?:\s*:\s*(.+))?", "dependency"),
            (r"(\w+)\s+--\s+(\w+)(?:\s*:\s*(.+))?", "link"),
        ]

        matched = False
        for pat, rel_type in rel_patterns:
            m = re.match(pat, line)
            if m:
                src, dst = m.group(1), m.group(2)
                label = clean_label(m.group(3)) if m.group(3) else ""
                relationships.append((src, dst, rel_type, label))
                for name in (src, dst):
                    if name not in classes:
                        classes[name] = {"stereotype": None, "members": []}
                        class_order.append(name)
                matched = True
                break

        if not matched and line and not line.startswith("%%"):
            notes.append(f"Unhandled line: {line}")

    # Generate D2 output
    for cls_name in class_order:
        cls_info = classes[cls_name]
        output_lines.append(f"{cls_name}: {cls_name} {{")
        output_lines.append("  shape: class")
        if cls_info["stereotype"]:
            output_lines.append(f'  "<<{cls_info["stereotype"]}>>"')
        for member in cls_info["members"]:
            output_lines.append(f"  {member}")
        output_lines.append("}")
        output_lines.append("")

    # Relationships
    for src, dst, rel_type, label in relationships:
        if rel_type == "inheritance":
            header = f"{src} <-> {dst}"
            if label:
                header += f": {escape_d2_label(label)}"
            output_lines.append(f"{header} {{")
            output_lines.append("  source-arrowhead.shape: triangle")
            output_lines.append("  target-arrowhead.shape: none")
            output_lines.append("}")
        elif rel_type == "realization":
            header = f"{src} <-> {dst}"
            if label:
                header += f": {escape_d2_label(label)}"
            output_lines.append(f"{header} {{")
            output_lines.append("  source-arrowhead.shape: triangle")
            output_lines.append("  target-arrowhead.shape: none")
            output_lines.append("  style.stroke-dash: 3")
            output_lines.append("}")
        elif rel_type == "composition":
            header = f"{src} <-> {dst}"
            if label:
                header += f": {escape_d2_label(label)}"
            output_lines.append(f"{header} {{")
            output_lines.append("  source-arrowhead.shape: diamond")
            output_lines.append("  source-arrowhead.style.filled: true")
            output_lines.append("  target-arrowhead.shape: none")
            output_lines.append("}")
        elif rel_type == "aggregation":
            header = f"{src} <-> {dst}"
            if label:
                header += f": {escape_d2_label(label)}"
            output_lines.append(f"{header} {{")
            output_lines.append("  source-arrowhead.shape: diamond")
            output_lines.append("  target-arrowhead.shape: none")
            output_lines.append("}")
        elif rel_type == "dependency":
            edge_line = f"{src} -> {dst}"
            if label:
                edge_line += f": {escape_d2_label(label)}"
            output_lines.append(f"{edge_line} {{")
            output_lines.append("  style.stroke-dash: 3")
            output_lines.append("}")
        elif rel_type == "association":
            edge_line = f"{src} -> {dst}"
            if label:
                edge_line += f": {escape_d2_label(label)}"
            output_lines.append(edge_line)
        else:
            edge_line = f"{src} -- {dst}"
            if label:
                edge_line += f": {escape_d2_label(label)}"
            output_lines.append(edge_line)

    for note in notes:
        output_lines.append(f"# CONVERSION NOTE: {note}")

    return "\n".join(output_lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Gantt diagram converter
# ---------------------------------------------------------------------------


def convert_gantt(source: str) -> str:
    """Convert a mermaid gantt diagram to D2 (approximation).

    Gantt charts have no direct D2 equivalent, so we create a horizontal
    layout with containers for sections and nodes for tasks.
    """
    lines = source.split("\n")
    output_lines: list[str] = ["direction: right", ""]
    notes: list[str] = []

    title = ""
    current_section = None
    section_counter = 0
    sections: dict[str, list[str]] = {}
    section_ids: list[tuple[str, str]] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line == "gantt":
            continue

        # Title
        m = re.match(r"title\s+(.+)", line)
        if m:
            title = clean_label(m.group(1))
            continue

        # dateFormat, axisFormat -- skip
        if line.startswith("dateFormat") or line.startswith("axisFormat"):
            continue

        # Section
        m = re.match(r"section\s+(.+)", line)
        if m:
            section_counter += 1
            current_section = f"section{section_counter}"
            section_label = clean_label(m.group(1))
            sections[current_section] = []
            section_ids.append((current_section, section_label))
            continue

        # Task line
        if current_section and line:
            parts = line.split(":", 1)
            task_name = clean_label(parts[0].strip())
            if task_name:
                sections[current_section].append(task_name)

    # Generate D2 output
    if title:
        notes.append(f"Original title: {title}")

    task_counter = 0
    for sec_id, sec_label in section_ids:
        output_lines.append(f"{sec_id}: {escape_d2_label(sec_label)} {{")
        for task in sections.get(sec_id, []):
            task_counter += 1
            task_id = f"task{task_counter}"
            output_lines.append(f"  {task_id}: {escape_d2_label(task)}")
        output_lines.append("}")
        output_lines.append("")

    # Link sections sequentially
    for i in range(len(section_ids) - 1):
        src_id = section_ids[i][0]
        dst_id = section_ids[i + 1][0]
        output_lines.append(f"{src_id} -> {dst_id}: {{")
        output_lines.append("  style.opacity: 0.3")
        output_lines.append("}")

    for note in notes:
        output_lines.append(f"# CONVERSION NOTE: {note}")

    return "\n".join(output_lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Main conversion dispatcher
# ---------------------------------------------------------------------------


def convert_mermaid_to_d2(source: str, diagram_type: str) -> str:
    """Convert a mermaid diagram source string to D2 diagram language.

    Args:
        source: The mermaid diagram source code.
        diagram_type: One of "flowchart", "sequence", "state", "class", "gantt".

    Returns:
        The D2 diagram source code.
    """
    converters = {
        "flowchart": convert_flowchart,
        "sequence": convert_sequence,
        "state": convert_state,
        "class": convert_class,
        "gantt": convert_gantt,
    }

    converter = converters.get(diagram_type)
    if converter is None:
        return (
            f"# CONVERSION NOTE: Unknown diagram type '{diagram_type}'\n"
            f"# Original source:\n" + "\n".join(f"# {line}" for line in source.split("\n")) + "\n"
        )

    try:
        return converter(source)
    except Exception as e:
        # If conversion fails, output what we can with a note
        return (
            f"# CONVERSION NOTE: Partial conversion failed: {e}\n"
            f"# Original source:\n" + "\n".join(f"# {line}" for line in source.split("\n")) + "\n"
        )


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    input_path = Path("/tmp/mermaid_diagrams.json")
    output_dir = Path("/tmp/d2_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path) as f:
        diagrams = json.load(f)

    print(f"Loaded {len(diagrams)} diagrams from {input_path}")

    success_count = 0
    error_count = 0

    for entry in diagrams:
        file_stem = Path(entry["file"]).stem
        index = entry["index"]
        diagram_type = entry["type"]
        source = entry["source"]
        caption = entry.get("caption", "")

        slug = slugify(caption) if caption else f"{file_stem}_{index}"
        out_filename = f"{file_stem}_{index}_{slug}.d2"
        out_path = output_dir / out_filename

        try:
            d2_source = convert_mermaid_to_d2(source, diagram_type)

            header = f"# Source: {entry['file']} (diagram {index})\n"
            if caption:
                header += f"# Caption: {caption}\n"
            header += f"# Type: {diagram_type}\n\n"

            with open(out_path, "w") as f:
                f.write(header + d2_source)

            has_notes = "CONVERSION NOTE" in d2_source
            status = "OK (with notes)" if has_notes else "OK"
            print(f"  [{status}] {out_path.name}")
            success_count += 1

        except Exception as e:
            print(f"  [ERROR] {out_path.name}: {e}")
            with open(out_path, "w") as f:
                f.write(
                    f"# CONVERSION NOTE: Failed to convert: {e}\n"
                    f"# Source: {entry['file']} (diagram {index})\n"
                    f"# Type: {diagram_type}\n\n" + "\n".join(f"# {line}" for line in source.split("\n")) + "\n"
                )
            error_count += 1

    print(f"\nDone: {success_count} succeeded, {error_count} failed, " f"{len(diagrams)} total")
    print(f"Output directory: {output_dir}")
