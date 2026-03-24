# Academic SVG Diagram Engine — Design Document

## Goal

Replace all 59 mermaid-rendered diagrams with hand-crafted-quality academic SVGs featuring hierarchical nesting, precise layout, and dual-theme high contrast.

## Quality Benchmark

- SGLang-JAX V1/V2 architecture diagrams (reference images)
- Existing `architecture.svg` on homepage

## Architecture

```
mermaid source (git history) → parser → structured definition → layout engine → SVG renderer → postprocessor
```

### Component Library

| Component | Description | SVG Elements |
|-----------|-------------|--------------|
| `Plane` | Full-width labeled region (e.g., "Control Plane") | rect + rotated side label |
| `Group` | Titled nested container (e.g., "Frontend (CPU)") | rect + title bar |
| `Box` | Leaf component (e.g., "Tokenizer") | rounded rect + centered text |
| `Arrow` | Labeled connection | path + marker + label rect |
| `Legend` | Color key | colored squares + text |
| `Annotation` | Explanatory text block | text lines |

### Layout Templates

| Template | Count | Strategy |
|----------|-------|----------|
| `hierarchical` | ~25 | Top-down layers, groups contain boxes |
| `pipeline` | ~20 | Linear flow (LR or TB), stages as groups |
| `sequence` | 8 | Participants horizontal, messages vertical |
| `state` | 3 | State nodes + transitions + nested groups |
| `class_diagram` | 3 | Class boxes + relationship lines |

### Dual-Theme Color System (High Contrast)

Each named color has light + dark variants. Contrast ratios verified ≥ 4.5:1.

```python
THEMES = {
    'blue': {
        'light': {'fill': '#dbeafe', 'stroke': '#2563eb', 'text': '#1e40af', 'group_fill': '#eff6ff', 'group_stroke': '#60a5fa'},
        'dark':  {'fill': '#1e3a5f', 'stroke': '#60a5fa', 'text': '#bfdbfe', 'group_fill': '#0c2340', 'group_stroke': '#3b82f6'},
    },
    'green': {
        'light': {'fill': '#d1fae5', 'stroke': '#059669', 'text': '#065f46', 'group_fill': '#ecfdf5', 'group_stroke': '#34d399'},
        'dark':  {'fill': '#064e3b', 'stroke': '#34d399', 'text': '#a7f3d0', 'group_fill': '#022c22', 'group_stroke': '#10b981'},
    },
    'amber': {
        'light': {'fill': '#fef3c7', 'stroke': '#d97706', 'text': '#92400e', 'group_fill': '#fffbeb', 'group_stroke': '#fbbf24'},
        'dark':  {'fill': '#78350f', 'stroke': '#fbbf24', 'text': '#fde68a', 'group_fill': '#451a03', 'group_stroke': '#f59e0b'},
    },
    'indigo': {
        'light': {'fill': '#e0e7ff', 'stroke': '#4f46e5', 'text': '#3730a3', 'group_fill': '#eef2ff', 'group_stroke': '#818cf8'},
        'dark':  {'fill': '#312e81', 'stroke': '#818cf8', 'text': '#c7d2fe', 'group_fill': '#1e1b4b', 'group_stroke': '#6366f1'},
    },
    'teal': {
        'light': {'fill': '#ccfbf1', 'stroke': '#0d9488', 'text': '#115e59', 'group_fill': '#f0fdfa', 'group_stroke': '#2dd4bf'},
        'dark':  {'fill': '#134e4a', 'stroke': '#2dd4bf', 'text': '#99f6e4', 'group_fill': '#042f2e', 'group_stroke': '#14b8a6'},
    },
    'rose': {
        'light': {'fill': '#ffe4e6', 'stroke': '#e11d48', 'text': '#9f1239', 'group_fill': '#fff1f2', 'group_stroke': '#fb7185'},
        'dark':  {'fill': '#881337', 'stroke': '#fb7185', 'text': '#fecdd3', 'group_fill': '#4c0519', 'group_stroke': '#f43f5e'},
    },
    'purple': {
        'light': {'fill': '#f3e8ff', 'stroke': '#7c3aed', 'text': '#5b21b6', 'group_fill': '#faf5ff', 'group_stroke': '#a78bfa'},
        'dark':  {'fill': '#4c1d95', 'stroke': '#a78bfa', 'text': '#ddd6fe', 'group_fill': '#2e1065', 'group_stroke': '#8b5cf6'},
    },
    'gray': {
        'light': {'fill': '#f1f5f9', 'stroke': '#64748b', 'text': '#334155', 'group_fill': '#f8fafc', 'group_stroke': '#94a3b8'},
        'dark':  {'fill': '#334155', 'stroke': '#94a3b8', 'text': '#cbd5e1', 'group_fill': '#1e293b', 'group_stroke': '#64748b'},
    },
}
```

### Layout Calculation

Bottom-up sizing:
```
Box.size     = measure_text(label) + padding(12, 8)
Group.size   = arrange_children(boxes, direction) + title_bar(28) + padding(16)
Plane.size   = arrange_children(groups, direction) + side_label(36) + padding(20)
Canvas.size  = sum(planes) + legend(80) + margin(40)
```

Arrow routing: orthogonal paths with quadratic Bezier corners (radius 8px).

### Dark Mode Implementation

Per-color CSS classes instead of blanket overrides:
```css
/* Light (default) */
.color-blue .box { fill: #dbeafe; stroke: #2563eb; }
.color-blue .box-text { fill: #1e40af; }

/* Dark */
@media (prefers-color-scheme: dark) {
  .color-blue .box { fill: #1e3a5f; stroke: #60a5fa; }
  .color-blue .box-text { fill: #bfdbfe; }
}
```

### Pipeline

1. Parse mermaid source → extract nodes, edges, subgraphs
2. Auto-detect template type (hierarchical/pipeline/sequence/state/class)
3. Auto-assign colors to groups (cycling through palette)
4. Layout engine calculates all coordinates
5. SVG renderer produces clean SVG with embedded dual-theme CSS
6. No inline styles on `<svg>` element (learned from previous issues)

### File Structure

```
docs/mkdocs/
  svg_engine/
    __init__.py
    components.py    # Box, Group, Plane, Arrow, Legend
    layout.py        # Layout calculation per template
    renderer.py      # SVG string generation
    themes.py        # Dual-theme color system
    parser.py        # Mermaid source → structured definition
  generate_all.py    # Main pipeline: parse → layout → render → save
```
