#!/usr/bin/env python3
"""Academic-style SVG renderer for siirl-agentic documentation diagrams.

Provides high-level primitives (regions, boxes, arrows, sequence diagrams)
with a unified color palette inspired by SGLang-JAX architecture figures.

Usage:
    canvas = SvgCanvas(width=900, height=500, title="My Diagram", caption="Figure 1: ...")
    r = canvas.region(x=50, y=80, w=250, h=300, title="Training (GPU)", palette="training")
    r.box("TrainerGroup", bold=True)
    r.box("Actor Model (Megatron)")
    r.box("Ref Model")
    canvas.arrow(r1_box, r2_box, label="training batch")
    svg_text = canvas.render()
"""

from __future__ import annotations

import html
import textwrap
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Academic Color Palette
# ---------------------------------------------------------------------------

PALETTES = {
    "training": {
        "bg": "#E8F5E9",
        "border": "#81C784",
        "title_color": "#2E7D32",
        "box_bg": "#FFFFFF",
        "box_border": "#A5D6A7",
        "box_text": "#424242",
    },
    "rollout": {
        "bg": "#E3F2FD",
        "border": "#64B5F6",
        "title_color": "#1565C0",
        "box_bg": "#FFFFFF",
        "box_border": "#90CAF9",
        "box_text": "#424242",
    },
    "data": {
        "bg": "#FFF8E1",
        "border": "#FFD54F",
        "title_color": "#F57F17",
        "box_bg": "#FFFFFF",
        "box_border": "#FFE082",
        "box_text": "#424242",
    },
    "tools": {
        "bg": "#F3E5F5",
        "border": "#CE93D8",
        "title_color": "#7B1FA2",
        "box_bg": "#FFFFFF",
        "box_border": "#E1BEE7",
        "box_text": "#424242",
    },
    "infra": {
        "bg": "#F5F5F5",
        "border": "#BDBDBD",
        "title_color": "#616161",
        "box_bg": "#FFFFFF",
        "box_border": "#E0E0E0",
        "box_text": "#424242",
    },
    "default": {
        "bg": "#FAFAFA",
        "border": "#E0E0E0",
        "title_color": "#616161",
        "box_bg": "#FFFFFF",
        "box_border": "#E0E0E0",
        "box_text": "#424242",
    },
    "highlight": {
        "bg": "#FFF3E0",
        "border": "#FFB74D",
        "title_color": "#E65100",
        "box_bg": "#FFFFFF",
        "box_border": "#FFCC80",
        "box_text": "#424242",
    },
    "error": {
        "bg": "#FFEBEE",
        "border": "#EF9A9A",
        "title_color": "#C62828",
        "box_bg": "#FFFFFF",
        "box_border": "#EF9A9A",
        "box_text": "#424242",
    },
}

# Arrow and line colors
ARROW_COLOR = "#78909C"
ARROW_DASH_COLOR = "#90A4AE"
CONNECTOR_COLOR = "#BDBDBD"
TITLE_BG = "#F8F9FA"
FRAME_COLOR = "#333333"

# Typography
FONT_FAMILY = "Helvetica, 'Helvetica Neue', Arial, sans-serif"
TITLE_SIZE = 14
REGION_TITLE_SIZE = 13
BOX_TEXT_SIZE = 12
BOX_BOLD_SIZE = 12.5
LABEL_SIZE = 10.5
CAPTION_SIZE = 12

# Spacing
BOX_HEIGHT = 30
BOX_GAP = 10
BOX_PADDING_X = 18
REGION_TITLE_HEIGHT = 26
REGION_PADDING = 16
REGION_GAP = 22


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Point:
    x: float
    y: float


@dataclass
class Rect:
    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    @property
    def top(self) -> float:
        return self.y

    @property
    def bottom(self) -> float:
        return self.y + self.h

    @property
    def left(self) -> float:
        return self.x

    @property
    def right(self) -> float:
        return self.x + self.w

    def port(self, side: str) -> Point:
        """Get connection point on a side: top, bottom, left, right."""
        if side == "top":
            return Point(self.cx, self.top)
        elif side == "bottom":
            return Point(self.cx, self.bottom)
        elif side == "left":
            return Point(self.left, self.cy)
        elif side == "right":
            return Point(self.right, self.cy)
        raise ValueError(f"Unknown side: {side}")


@dataclass
class BoxInfo:
    """Returned when a box is drawn, for use in arrow connections."""

    rect: Rect
    label: str
    region: Optional["Region"] = None

    def port(self, side: str) -> Point:
        return self.rect.port(side)


# ---------------------------------------------------------------------------
# Region: colored container with auto-stacking boxes
# ---------------------------------------------------------------------------


class Region:
    def __init__(self, canvas: "SvgCanvas", x: float, y: float, w: float, h: float, title: str, palette_name: str = "default"):
        self.canvas = canvas
        self.rect = Rect(x, y, w, h)
        self.title = title
        self.palette = PALETTES.get(palette_name, PALETTES["default"])
        self._next_y = y + REGION_TITLE_HEIGHT + REGION_PADDING
        self._boxes: list[BoxInfo] = []
        self._inner_x = x + REGION_PADDING
        self._inner_w = w - 2 * REGION_PADDING

    def box(self, label: str, bold: bool = False, palette_override: Optional[dict] = None) -> BoxInfo:
        """Add a box inside this region, auto-positioned vertically."""
        p = palette_override or self.palette
        bx = self._inner_x
        by = self._next_y
        bw = self._inner_w
        bh = BOX_HEIGHT
        self._next_y = by + bh + BOX_GAP
        bi = BoxInfo(rect=Rect(bx, by, bw, bh), label=label, region=self)
        self._boxes.append(bi)
        # Record drawing command
        border = self.palette["box_border"] if not bold else self.palette["border"]
        self.canvas._elements.append(_svg_box(bx, by, bw, bh, label, bold=bold, fill=p["box_bg"], stroke=border, text_color=p["box_text"]))
        return bi

    def note(self, text: str, size: float = 9, color: Optional[str] = None):
        """Add a small italic note below the last box."""
        c = color or self.palette["border"]
        ny = self._next_y - BOX_GAP + 4
        self.canvas._elements.append(
            f'<text x="{self.rect.cx}" y="{ny}" text-anchor="middle" '
            f'fill="{c}" font-size="{size}" font-style="italic">{_esc(text)}</text>'
        )
        self._next_y = ny + 14

    def render_bg(self) -> str:
        """Render the region background and title."""
        r = self.rect
        p = self.palette
        lines = [
            f'<rect x="{r.x}" y="{r.y}" width="{r.w}" height="{r.h}" '
            f'rx="2" fill="{p["bg"]}" stroke="{p["border"]}" stroke-width="1.2"/>',
            f'<text x="{r.cx}" y="{r.y + 16}" text-anchor="middle" '
            f'fill="{p["title_color"]}" font-size="{REGION_TITLE_SIZE}" '
            f'font-weight="bold">{_esc(self.title)}</text>',
        ]
        return "\n  ".join(lines)


# ---------------------------------------------------------------------------
# Sequence Diagram helpers
# ---------------------------------------------------------------------------


@dataclass
class Actor:
    name: str
    short: str
    x: float
    palette_name: str = "default"


class SequenceDiagram:
    """Helper for building sequence diagrams."""

    def __init__(
        self, canvas: "SvgCanvas", actors: list[dict], x: float = 50, y: float = 60, actor_spacing: float = 150, msg_spacing: float = 36
    ):
        self.canvas = canvas
        self.x = x
        self.y = y
        self.actor_spacing = actor_spacing
        self.msg_spacing = msg_spacing
        self._msg_y = y + 50  # below actor boxes
        self._actors: dict[str, Actor] = {}

        for i, a in enumerate(actors):
            ax = x + i * actor_spacing
            name = a.get("name", a.get("short", f"Actor{i}"))
            short = a.get("short", name)
            pal = a.get("palette", "default")
            self._actors[short] = Actor(name=name, short=short, x=ax, palette_name=pal)

    @property
    def current_y(self) -> float:
        return self._msg_y

    def message(self, src: str, dst: str, label: str, dashed: bool = False, self_msg: bool = False):
        """Draw a message arrow between two actors."""
        a1 = self._actors[src]
        a2 = self._actors[dst]
        y = self._msg_y
        self._msg_y += self.msg_spacing

        if self_msg or src == dst:
            # Self-message: loop back
            self.canvas._elements.append(
                f'<path d="M{a1.x},{y} L{a1.x + 30},{y} L{a1.x + 30},{y + 18} L{a1.x},{y + 18}" '
                f'fill="none" stroke="{ARROW_COLOR}" stroke-width="1.2" '
                f'{"stroke-dasharray=&quot;5,3&quot; " if dashed else ""}'
                f'marker-end="url(#seq-arr)"/>'
            )
            self.canvas._elements.append(
                f'<text x="{a1.x + 34}" y="{y + 12}" fill="{ARROW_COLOR}" ' f'font-size="{LABEL_SIZE}">{_esc(label)}</text>'
            )
            self._msg_y += 10
            return

        dash = 'stroke-dasharray="5,3" ' if dashed else ""
        self.canvas._elements.append(
            f'<line x1="{a1.x}" y1="{y}" x2="{a2.x}" y2="{y}" '
            f'stroke="{ARROW_COLOR}" stroke-width="1.2" {dash}'
            f'marker-end="url(#seq-arr)"/>'
        )
        tx = (a1.x + a2.x) / 2
        self.canvas._elements.append(
            f'<text x="{tx}" y="{y - 5}" text-anchor="middle" fill="#607D8B" ' f'font-size="{LABEL_SIZE}">{_esc(label)}</text>'
        )

    def note(self, actor: str, text: str, side: str = "right", width: float = 120):
        """Draw a note next to an actor."""
        a = self._actors[actor]
        y = self._msg_y
        nx = a.x + 20 if side == "right" else a.x - width - 20
        lines = textwrap.wrap(text, width=20)
        nh = max(len(lines) * 14 + 10, 28)
        self.canvas._elements.append(
            f'<rect x="{nx}" y="{y - 4}" width="{width}" height="{nh}" ' f'rx="2" fill="#FFFDE7" stroke="#FDD835" stroke-width="0.8"/>'
        )
        for i, line in enumerate(lines):
            self.canvas._elements.append(
                f'<text x="{nx + 8}" y="{y + 10 + i * 14}" fill="#6D4C00" ' f'font-size="{LABEL_SIZE}">{_esc(line)}</text>'
            )
        self._msg_y += nh + 8

    def loop(self, label: str, count: int = 1):
        """Mark the start of a loop block. Returns the y position."""
        y = self._msg_y - 5
        self.canvas._loop_starts.append((y, label))
        return y

    def end_loop(self):
        """End the current loop block."""
        if self.canvas._loop_starts:
            start_y, label = self.canvas._loop_starts.pop()
            end_y = self._msg_y
            actors = list(self._actors.values())
            lx = actors[0].x - 30
            lw = actors[-1].x - actors[0].x + 60
            self.canvas._bg_elements.append(
                f'<rect x="{lx}" y="{start_y}" width="{lw}" height="{end_y - start_y + 10}" '
                f'rx="2" fill="none" stroke="#B0BEC5" stroke-width="1" stroke-dasharray="6,3"/>'
            )
            self.canvas._bg_elements.append(
                f'<rect x="{lx}" y="{start_y}" width="{len(label) * 6.5 + 16}" height="16" '
                f'rx="2" fill="#ECEFF1" stroke="#B0BEC5" stroke-width="0.8"/>'
            )
            self.canvas._bg_elements.append(
                f'<text x="{lx + 8}" y="{start_y + 12}" fill="#546E7A" ' f'font-size="{LABEL_SIZE}" font-weight="600">{_esc(label)}</text>'
            )

    def render_actors(self) -> list[str]:
        """Render actor boxes and lifelines."""
        parts = []
        max_y = self._msg_y + 20
        for a in self._actors.values():
            pal = PALETTES.get(a.palette_name, PALETTES["default"])
            bw = max(len(a.name) * 7 + 20, 80)
            bx = a.x - bw / 2
            by = self.y
            # Lifeline
            parts.append(
                f'<line x1="{a.x}" y1="{by + 28}" x2="{a.x}" y2="{max_y}" ' f'stroke="#CFD8DC" stroke-width="1" stroke-dasharray="4,3"/>'
            )
            # Actor box
            parts.append(
                f'<rect x="{bx}" y="{by}" width="{bw}" height="28" '
                f'rx="2" fill="{pal["bg"]}" stroke="{pal["border"]}" stroke-width="1.2"/>'
            )
            parts.append(
                f'<text x="{a.x}" y="{by + 18}" text-anchor="middle" '
                f'fill="{pal["title_color"]}" font-size="{BOX_TEXT_SIZE}" font-weight="600">'
                f"{_esc(a.name)}</text>"
            )
        return parts


# ---------------------------------------------------------------------------
# State Diagram helpers
# ---------------------------------------------------------------------------


@dataclass
class StateBox:
    rect: Rect
    label: str


class StateDiagram:
    """Helper for building state diagrams."""

    def __init__(self, canvas: "SvgCanvas"):
        self.canvas = canvas
        self._states: dict[str, StateBox] = {}

    def state(self, name: str, label: str, x: float, y: float, w: float = 140, h: float = 32, palette_name: str = "default") -> StateBox:
        pal = PALETTES.get(palette_name, PALETTES["default"])
        r = Rect(x, y, w, h)
        sb = StateBox(rect=r, label=label)
        self._states[name] = sb
        self.canvas._elements.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
            f'rx="{h / 2}" fill="{pal["bg"]}" stroke="{pal["border"]}" stroke-width="1.2"/>'
        )
        self.canvas._elements.append(
            f'<text x="{r.cx}" y="{r.cy + 4}" text-anchor="middle" '
            f'fill="{pal["title_color"]}" font-size="{BOX_TEXT_SIZE}" font-weight="600">'
            f"{_esc(label)}</text>"
        )
        return sb

    def start_dot(self, x: float, y: float, name: str = "_start") -> StateBox:
        r = Rect(x - 6, y - 6, 12, 12)
        sb = StateBox(rect=r, label="")
        self._states[name] = sb
        self.canvas._elements.append(f'<circle cx="{x}" cy="{y}" r="6" fill="#424242"/>')
        return sb

    def end_dot(self, x: float, y: float, name: str = "_end") -> StateBox:
        r = Rect(x - 8, y - 8, 16, 16)
        sb = StateBox(rect=r, label="")
        self._states[name] = sb
        self.canvas._elements.append(f'<circle cx="{x}" cy="{y}" r="8" fill="none" stroke="#424242" stroke-width="1.5"/>')
        self.canvas._elements.append(f'<circle cx="{x}" cy="{y}" r="5" fill="#424242"/>')
        return sb

    def transition(
        self,
        src: str,
        dst: str,
        label: str = "",
        src_side: str = "bottom",
        dst_side: str = "top",
        waypoints: Optional[list[tuple[float, float]]] = None,
    ):
        s = self._states[src]
        d = self._states[dst]
        p1 = s.rect.port(src_side)
        p2 = d.rect.port(dst_side)
        self.canvas.arrow_between(p1, p2, label=label, waypoints=waypoints)


# ---------------------------------------------------------------------------
# SvgCanvas: main document builder
# ---------------------------------------------------------------------------


class SvgCanvas:
    def __init__(self, width: float, height: float, title: str = "", caption: str = "", framed: bool = True):
        self.width = width
        self.height = height
        self.title = title
        self.caption = caption
        self.framed = framed
        self._regions: list[Region] = []
        self._bg_elements: list[str] = []  # behind main elements
        self._elements: list[str] = []
        self._loop_starts: list[tuple[float, str]] = []

    def region(self, x: float, y: float, w: float, h: float, title: str, palette: str = "default") -> Region:
        r = Region(self, x, y, w, h, title, palette)
        self._regions.append(r)
        return r

    def box(
        self, x: float, y: float, w: float, h: float, label: str, bold: bool = False, palette: str = "default", shape: str = "rect"
    ) -> BoxInfo:
        """Draw a standalone box (not inside a region)."""
        p = PALETTES.get(palette, PALETTES["default"])
        bi = BoxInfo(rect=Rect(x, y, w, h), label=label)
        border = p["border"] if bold else p["box_border"]
        self._elements.append(
            _svg_box(x, y, w, h, label, bold=bold, fill=p["box_bg"], stroke=border, text_color=p["box_text"], shape=shape)
        )
        return bi

    def diamond(self, x: float, y: float, size: float, label: str, palette: str = "default") -> BoxInfo:
        """Draw a decision diamond."""
        p = PALETTES.get(palette, PALETTES["default"])
        cx, cy = x + size / 2, y + size / 2
        pts = f"{cx},{y} {x + size},{cy} {cx},{y + size} {x},{cy}"
        self._elements.append(f'<polygon points="{pts}" fill="{p["box_bg"]}" ' f'stroke="{p["border"]}" stroke-width="1"/>')
        self._elements.append(
            f'<text x="{cx}" y="{cy + 4}" text-anchor="middle" ' f'fill="{p["box_text"]}" font-size="{LABEL_SIZE}">{_esc(label)}</text>'
        )
        return BoxInfo(rect=Rect(x, y, size, size), label=label)

    def arrow(
        self,
        src: BoxInfo,
        dst: BoxInfo,
        label: str = "",
        src_side: str = "right",
        dst_side: str = "left",
        dashed: bool = False,
        color: Optional[str] = None,
        waypoints: Optional[list[tuple[float, float]]] = None,
    ):
        """Draw an arrow between two boxes."""
        p1 = src.port(src_side)
        p2 = dst.port(dst_side)
        self.arrow_between(p1, p2, label=label, dashed=dashed, color=color, waypoints=waypoints)

    def arrow_between(
        self,
        p1: Point,
        p2: Point,
        label: str = "",
        dashed: bool = False,
        color: Optional[str] = None,
        waypoints: Optional[list[tuple[float, float]]] = None,
    ):
        """Draw an arrow between two points, optionally via waypoints."""
        c = color or (ARROW_DASH_COLOR if dashed else ARROW_COLOR)
        dash = 'stroke-dasharray="6,3" ' if dashed else ""
        marker_id = "arr-dash" if dashed else "arr"

        if waypoints:
            path_d = f"M{p1.x},{p1.y}"
            for wx, wy in waypoints:
                path_d += f" L{wx},{wy}"
            path_d += f" L{p2.x},{p2.y}"
            self._elements.append(
                f'<path d="{path_d}" fill="none" stroke="{c}" ' f'stroke-width="1.2" {dash}marker-end="url(#{marker_id})"/>'
            )
        else:
            self._elements.append(
                f'<line x1="{p1.x}" y1="{p1.y}" x2="{p2.x}" y2="{p2.y}" '
                f'stroke="{c}" stroke-width="1.2" {dash}'
                f'marker-end="url(#{marker_id})"/>'
            )

        if label:
            if waypoints:
                mid = waypoints[len(waypoints) // 2]
                tx, ty = mid[0], mid[1] - 6
            else:
                tx = (p1.x + p2.x) / 2
                ty = (p1.y + p2.y) / 2 - 6
            self._elements.append(
                f'<text x="{tx}" y="{ty}" text-anchor="middle" '
                f'fill="#607D8B" font-size="{LABEL_SIZE}" font-style="italic">'
                f"{_esc(label)}</text>"
            )

    def step_number(self, x: float, y: float, num: int, color: str = ARROW_COLOR):
        """Draw a small circled step number."""
        self._elements.append(f'<circle cx="{x}" cy="{y}" r="8" fill="#fff" stroke="{color}" stroke-width="0.8"/>')
        self._elements.append(
            f'<text x="{x}" y="{y + 3.5}" text-anchor="middle" fill="{color}" ' f'font-size="8.5" font-weight="bold">{num}</text>'
        )

    def text(
        self,
        x: float,
        y: float,
        text: str,
        size: float = 10,
        color: str = "#424242",
        weight: str = "normal",
        anchor: str = "middle",
        style: str = "normal",
    ):
        """Draw arbitrary text."""
        self._elements.append(
            f'<text x="{x}" y="{y}" text-anchor="{anchor}" fill="{color}" '
            f'font-size="{size}" font-weight="{weight}" font-style="{style}">'
            f"{_esc(text)}</text>"
        )

    def dashed_connector(self, x1: float, y1: float, x2: float, y2: float):
        """Draw a light dashed connector line (no arrowhead)."""
        self._elements.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" ' f'stroke="{CONNECTOR_COLOR}" stroke-width="1" stroke-dasharray="4,3"/>'
        )

    def horizontal_divider(self, x: float, y: float, w: float, label: str = ""):
        """Draw a horizontal divider line with optional label."""
        self._elements.append(f'<line x1="{x}" y1="{y}" x2="{x + w}" y2="{y}" ' f'stroke="#E0E0E0" stroke-width="0.8"/>')
        if label:
            self._elements.append(
                f'<text x="{x + w / 2}" y="{y - 4}" text-anchor="middle" ' f'fill="#9E9E9E" font-size="{LABEL_SIZE}">{_esc(label)}</text>'
            )

    def sequence(
        self, actors: list[dict], x: float = 50, y: float = 60, actor_spacing: float = 150, msg_spacing: float = 36
    ) -> SequenceDiagram:
        """Create a sequence diagram helper."""
        return SequenceDiagram(self, actors, x, y, actor_spacing, msg_spacing)

    def state_diagram(self) -> StateDiagram:
        """Create a state diagram helper."""
        return StateDiagram(self)

    def render(self) -> str:
        """Produce the final SVG string."""
        parts = []
        parts.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="100%" '
            f'viewBox="0 0 {self.width} {self.height}" '
            f'font-family="{FONT_FAMILY}" '
            f'text-rendering="optimizeLegibility">'
        )

        # Defs: arrow markers
        parts.append("  <defs>")
        parts.append('    <marker id="arr" markerWidth="8" markerHeight="6" ' 'refX="7" refY="3" orient="auto">')
        parts.append(f'      <polygon points="0 0.8, 7 3, 0 5.2" fill="{ARROW_COLOR}"/>')
        parts.append("    </marker>")
        parts.append('    <marker id="arr-dash" markerWidth="8" markerHeight="6" ' 'refX="7" refY="3" orient="auto">')
        parts.append(f'      <polygon points="0 0.8, 7 3, 0 5.2" fill="{ARROW_DASH_COLOR}"/>')
        parts.append("    </marker>")
        parts.append('    <marker id="seq-arr" markerWidth="8" markerHeight="6" ' 'refX="7" refY="3" orient="auto">')
        parts.append(f'      <polygon points="0 0.8, 7 3, 0 5.2" fill="{ARROW_COLOR}"/>')
        parts.append("    </marker>")
        parts.append("  </defs>")

        # Background
        parts.append(f'  <rect width="{self.width}" height="{self.height}" fill="#fff"/>')

        # Frame
        frame_h = self.height - 60 if self.caption else self.height - 20
        if self.framed:
            parts.append(
                f'  <rect x="20" y="10" width="{self.width - 40}" height="{frame_h}" '
                f'rx="0" fill="none" stroke="{FRAME_COLOR}" stroke-width="1.5"/>'
            )

        # Title bar
        if self.title and self.framed:
            parts.append(
                f'  <rect x="20" y="10" width="{self.width - 40}" height="32" '
                f'rx="0" fill="{TITLE_BG}" stroke="{FRAME_COLOR}" stroke-width="1.5"/>'
            )
            parts.append(
                f'  <text x="{self.width / 2}" y="31" text-anchor="middle" '
                f'fill="#222" font-size="{TITLE_SIZE}" font-weight="bold" '
                f'letter-spacing="0.3">{_esc(self.title)}</text>'
            )

        # Background elements (loop boxes, etc.)
        for el in self._bg_elements:
            parts.append(f"  {el}")

        # Region backgrounds
        for region in self._regions:
            parts.append(f"  {region.render_bg()}")

        # Main elements
        for el in self._elements:
            parts.append(f"  {el}")

        # Caption
        if self.caption:
            parts.append(
                f'  <text x="{self.width / 2}" y="{self.height - 15}" '
                f'text-anchor="middle" fill="#555" font-size="{CAPTION_SIZE}">'
                f"{_esc(self.caption)}</text>"
            )

        parts.append("</svg>")
        return "\n".join(parts)

    def save(self, path: str):
        """Write the SVG to a file."""
        with open(path, "w") as f:
            f.write(self.render())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _esc(text: str) -> str:
    """Escape text for SVG XML."""
    return html.escape(text, quote=True)


def _svg_box(
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    bold: bool = False,
    fill: str = "#fff",
    stroke: str = "#E0E0E0",
    text_color: str = "#424242",
    shape: str = "rect",
) -> str:
    """Generate SVG for a single labeled box."""
    fs = BOX_BOLD_SIZE if bold else BOX_TEXT_SIZE
    fw = "600" if bold else "normal"
    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" ' f'rx="1" fill="{fill}" stroke="{stroke}" stroke-width="0.8"/>',
        f'<text x="{x + w / 2}" y="{y + h / 2 + 4}" text-anchor="middle" '
        f'fill="{text_color}" font-size="{fs}" font-weight="{fw}">{_esc(label)}</text>',
    ]
    return "\n  ".join(parts)


def estimate_text_width(text: str, font_size: float = 12) -> float:
    """Rough estimate of text width in px."""
    return len(text) * font_size * 0.62
