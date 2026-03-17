#!/usr/bin/env python3
"""Generate all SVG diagrams from mermaid source via D2.

Pipeline: mermaid_diagrams.json → mermaid_to_d2 → .d2 files → d2 CLI (ELK) → .svg

Uses the same filename convention as the old SVG engine so markdown references
don't need updating.
"""

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from mermaid_to_d2 import convert_mermaid_to_d2

DIAGRAMS_DIR = Path(__file__).parent / "docs" / "assets" / "images" / "diagrams"

# D2 rendering options
D2_LAYOUT = "elk"
D2_THEME = "0"  # Neutral Default (white background)
D2_PAD = "30"


def slugify(text: str) -> str:
    """Match the old SVG engine's filename convention."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "_", text)
    return text[:60].rstrip("_")


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

        # Use same naming as old engine
        page_slug = slugify(rel.replace("/", "").replace(".md", ""))
        svg_name = f"{page_slug}_{idx}.svg"
        svg_path = DIAGRAMS_DIR / svg_name

        print(f"  {rel} [{idx}] ({dtype}): {svg_name}", end=" ")

        try:
            # Convert mermaid → D2
            d2_source = convert_mermaid_to_d2(source, dtype)

            # Write temp .d2 file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".d2", delete=False) as tmp:
                tmp.write(d2_source)
                tmp_path = tmp.name

            # Render with d2 CLI
            result = subprocess.run(
                [
                    "d2",
                    "--layout",
                    D2_LAYOUT,
                    "--theme",
                    D2_THEME,
                    "--pad",
                    D2_PAD,
                    tmp_path,
                    str(svg_path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)

            if result.returncode != 0:
                print(f"✗ D2 error: {result.stderr.strip()[:200]}")
                failed += 1
            else:
                print("✓")
                success += 1

        except Exception as e:
            print(f"✗ {e}")
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"Generated: {success}  |  Failed: {failed}  |  Total: {len(all_diagrams)}")
    print(f"SVGs in output dir: {len(list(DIAGRAMS_DIR.glob('*.svg')))}")


if __name__ == "__main__":
    main()
