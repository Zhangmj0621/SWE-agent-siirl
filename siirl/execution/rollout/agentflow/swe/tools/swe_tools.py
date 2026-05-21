#!/usr/bin/env python3
"""
Minimal SWE-agent-like tools implemented natively for siirl-agentic.

Goals:
- Provide SWE-agent-compatible command names (open/goto/scroll_up/scroll_down/search_dir/find_file/submit/create/edit).
- Maintain a small state file at /root/state.json with keys: open_file, first_line, working_dir.
- Format observations with the SWE-agent style footer:
  (Open file: ...)
  (Current directory: ...)
  bash-$

This is intentionally lightweight and avoids depending on SWE-agent's tool bundles.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict

WINDOW = int(os.environ.get("SWE_TOOL_WINDOW", "100"))
MAX_OUTPUT = int(os.environ.get("SWE_TOOL_MAX_OUTPUT", "50000"))

STATE_PATH = Path(os.environ.get("SWE_TOOL_STATE_PATH", "/root/state.json"))
UNDO_DIR = Path(os.environ.get("SWE_TOOL_UNDO_DIR", "/root/.swe-tool-undo"))


def _load_state():
    # type: () -> Dict[str, str]
    if not STATE_PATH.exists():
        return {"open_file": "n/a", "first_line": "0", "working_dir": os.getcwd()}
    try:
        raw = STATE_PATH.read_text(encoding="utf-8", errors="replace").strip()
        if not raw:
            return {"open_file": "n/a", "first_line": "0", "working_dir": os.getcwd()}
        state = json.loads(raw)
        if not isinstance(state, dict):
            raise ValueError("state is not a dict")
        state.setdefault("open_file", "n/a")
        state.setdefault("first_line", "0")
        state.setdefault("working_dir", os.getcwd())
        # Ensure all values are strings for upstream dict[str, str] compatibility
        state = {k: str(v) for k, v in state.items()}
        return state
    except Exception:
        return {"open_file": "n/a", "first_line": "0", "working_dir": os.getcwd()}


def _save_state(state):
    # type: (dict) -> None
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")


def _undo_key(path):
    safe = str(path).replace("/", "__").replace("\\", "__")
    return UNDO_DIR / f"{safe}.bak"


def _save_undo(path, content):
    UNDO_DIR.mkdir(parents=True, exist_ok=True)
    _undo_key(path).write_text(content, encoding="utf-8")


def _load_undo(path):
    key = _undo_key(path)
    if not key.exists():
        return None
    return key.read_text(encoding="utf-8", errors="replace")


def _resolve_path(p, state):
    path = Path(p)
    if path.is_absolute():
        return path
    wd = Path(state.get("working_dir") or os.getcwd())
    return (wd / path).resolve()


def _clip(s):
    if len(s) <= MAX_OUTPUT:
        return s
    return s[:MAX_OUTPUT] + "\n<response clipped>"


def _format_obs(content, state):
    open_file = state.get("open_file", "n/a")
    working_dir = state.get("working_dir", os.getcwd())
    parts = [
        content.rstrip(),
        f"(Open file: {open_file})",
        f"(Current directory: {working_dir})",
        "bash-$",
    ]
    return "\n".join(parts)


def _numbered_window(lines, first_line):
    end = min(first_line + WINDOW, len(lines))
    window = lines[first_line:end]
    return "\n".join(f"{i + first_line + 1:6d}  {ln}" for i, ln in enumerate(window))


def cmd_open(args):
    state = _load_state()
    path = _resolve_path(args.path, state)
    if not path.exists() or not path.is_file():
        return _format_obs(f"Error: File {args.path} not found", state)
    content = path.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()
    first_line = 0
    if args.line_number is not None and 1 <= args.line_number <= len(lines):
        first_line = max(0, args.line_number - 1)
    state["open_file"] = str(path)
    state["first_line"] = str(first_line)
    state["working_dir"] = str(Path.cwd())
    _save_state(state)
    out = _numbered_window(lines, first_line)
    return _format_obs(_clip(out), state)


def cmd_goto(args):
    state = _load_state()
    open_file = state.get("open_file", "n/a")
    if open_file == "n/a":
        return _format_obs("No file open. Use the open command first.", state)
    path = Path(open_file)
    if not path.exists():
        return _format_obs(f"Error: File {open_file} not found", state)
    content = path.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()
    if args.line_number < 1 or args.line_number > len(lines):
        return _format_obs(f"Error: line must be between 1 and {len(lines)}", state)
    first_line = max(0, args.line_number - 1)
    state["first_line"] = str(first_line)
    state["working_dir"] = str(Path.cwd())
    _save_state(state)
    out = _numbered_window(lines, first_line)
    return _format_obs(_clip(out), state)


def cmd_scroll(args, direction):
    state = _load_state()
    open_file = state.get("open_file", "n/a")
    if open_file == "n/a":
        return _format_obs("No file open. Use the open command first.", state)
    path = Path(open_file)
    if not path.exists():
        return _format_obs(f"Error: File {open_file} not found", state)
    content = path.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()
    first_line = int(state.get("first_line") or 0)
    if direction == "up":
        first_line = max(0, first_line - WINDOW)
    else:
        first_line = min(first_line + WINDOW, max(0, len(lines) - WINDOW))
    state["first_line"] = str(first_line)
    state["working_dir"] = str(Path.cwd())
    _save_state(state)
    out = _numbered_window(lines, first_line)
    return _format_obs(_clip(out), state)


def _run_capture(cmd, cwd=None):
    try:
        p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
        return p.returncode, p.stdout or ""
    except FileNotFoundError:
        return 127, f"Error: command not found: {cmd[0]}"


def cmd_search_dir(args):
    state = _load_state()
    base = _resolve_path(args.dir, state) if args.dir else Path(state.get("working_dir") or os.getcwd())
    if not base.exists():
        return _format_obs(f"Error: dir not found: {str(base)}", state)
    # Prefer rg; fallback to grep -R.
    rc, out = _run_capture(["rg", "-n", "--no-heading", args.search_term, str(base)])
    if rc == 127:
        rc, out = _run_capture(["grep", "-RIn", args.search_term, str(base)])
    if not out.strip():
        out = "No matches found."
    state["working_dir"] = str(Path.cwd())
    _save_state(state)
    return _format_obs(_clip(out.strip()), state)


def cmd_search_file(args):
    state = _load_state()
    target = args.file or state.get("open_file", "n/a")
    if target == "n/a":
        return _format_obs("No file open. Specify a file or open one first.", state)
    path = _resolve_path(target, state)
    if not path.exists() or not path.is_file():
        return _format_obs(f"Error: file not found: {target}", state)
    # Simple line-based search with numbering.
    content = path.read_text(encoding="utf-8", errors="replace")
    pat = re.compile(re.escape(args.search_term))
    matches = []
    for i, ln in enumerate(content.splitlines(), start=1):
        if pat.search(ln):
            matches.append(f"{path}:{i}:{ln}")
            if len(matches) >= 200:
                break
    out = "\n".join(matches) if matches else "No matches found."
    state["working_dir"] = str(Path.cwd())
    _save_state(state)
    return _format_obs(_clip(out), state)


def cmd_find_file(args):
    state = _load_state()
    base = _resolve_path(args.dir, state) if args.dir else Path(state.get("working_dir") or os.getcwd())
    if not base.exists():
        return _format_obs(f"Error: dir not found: {str(base)}", state)
    name = args.file_name
    results = []
    for p in base.rglob(name):
        results.append(str(p))
        if len(results) >= 100:
            break
    out = "\n".join(results) if results else "No matching files found."
    state["working_dir"] = str(Path.cwd())
    _save_state(state)
    return _format_obs(_clip(out), state)


def cmd_create(args):
    state = _load_state()
    path = _resolve_path(args.filename, state)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("", encoding="utf-8")
    state["open_file"] = str(path)
    state["first_line"] = "0"
    state["working_dir"] = str(Path.cwd())
    _save_state(state)
    return _format_obs("Created file.", state)


def cmd_edit(args):
    state = _load_state()
    open_file = state.get("open_file", "n/a")
    if open_file == "n/a":
        return _format_obs("No file open. Use the open command first.", state)
    path = Path(open_file)
    if not path.exists():
        return _format_obs(f"Error: File {open_file} not found", state)
    content = path.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()
    n = len(lines)
    if args.start_line < 1 or args.end_line < args.start_line or args.end_line > n:
        return _format_obs(f"Error: Invalid line range. File has {n} lines.", state)
    _save_undo(path, content)
    before = lines[: args.start_line - 1]
    after = lines[args.end_line :]
    repl_lines = args.replacement_text.rstrip("\n").split("\n")
    new_lines = before + repl_lines + after
    new_content = "\n".join(new_lines) + ("\n" if content.endswith("\n") else "")
    path.write_text(new_content, encoding="utf-8")
    state["working_dir"] = str(Path.cwd())
    _save_state(state)
    return _format_obs("Edit applied.", state)


def cmd_insert(args):
    state = _load_state()
    open_file = state.get("open_file", "n/a")
    if open_file == "n/a":
        return _format_obs("No file open. Use the open command first.", state)
    path = Path(open_file)
    if not path.exists():
        return _format_obs(f"Error: File {open_file} not found", state)
    content = path.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()
    line = args.line if args.line is not None else len(lines)
    if line < 0 or line > len(lines):
        return _format_obs("Error: line out of range", state)
    _save_undo(path, content)
    insert_lines = args.text.split("\n")
    new_lines = lines[:line] + insert_lines + lines[line:]
    new_content = "\n".join(new_lines) + ("\n" if content.endswith("\n") or args.text.endswith("\n") else "")
    path.write_text(new_content, encoding="utf-8")
    state["working_dir"] = str(Path.cwd())
    _save_state(state)
    return _format_obs("Insert applied.", state)


def cmd_replace_in_window(args):
    state = _load_state()
    open_file = state.get("open_file", "n/a")
    if open_file == "n/a":
        return _format_obs("No file open. Use the open command first.", state)
    path = Path(open_file)
    if not path.exists():
        return _format_obs(f"Error: File {open_file} not found", state)
    content = path.read_text(encoding="utf-8", errors="replace")
    lines = content.splitlines()
    first_line = int(state.get("first_line") or 0)
    end = min(first_line + WINDOW, len(lines))
    window_text = "\n".join(lines[first_line:end])
    if args.search not in window_text:
        return _format_obs(
            "Error: search text not found in the currently displayed lines.",
            state,
        )
    _save_undo(path, content)
    if args.replace_all:
        new_window_text = window_text.replace(args.search, args.replace)
    else:
        new_window_text = window_text.replace(args.search, args.replace, 1)
    new_window_lines = new_window_text.split("\n")
    new_lines = lines[:first_line] + new_window_lines + lines[end:]
    new_content = "\n".join(new_lines) + ("\n" if content.endswith("\n") else "")
    path.write_text(new_content, encoding="utf-8")
    state["working_dir"] = str(Path.cwd())
    _save_state(state)
    return _format_obs("Edit applied.", state)


def cmd_str_replace_editor(args):
    state = _load_state()
    path = _resolve_path(args.path, state)

    if args.command == "view":
        if not path.exists():
            return _format_obs(f"Error: {args.path} not found", state)
        if path.is_dir():
            entries = []
            base_depth = len(path.parts)
            for p in sorted(path.rglob("*")):
                if any(part.startswith(".") for part in p.relative_to(path).parts):
                    continue
                depth = len(p.parts) - base_depth
                if depth > 2:
                    continue
                entries.append(str(p))
                if len(entries) >= 200:
                    break
            out = "\n".join(entries) if entries else "(empty)"
            return _format_obs(_clip(out), state)

        content = path.read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines()
        n_lines = len(lines)
        if args.view_range:
            start = max(0, args.view_range[0] - 1)
            end = n_lines if len(args.view_range) < 2 or args.view_range[1] < 0 else min(args.view_range[1], n_lines)
        else:
            start, end = 0, n_lines
        numbered = "\n".join(f"{i + start + 1:6d}  {ln}" for i, ln in enumerate(lines[start:end]))
        return _format_obs(_clip(numbered), state)

    if args.command == "create":
        if path.exists():
            return _format_obs(f"Error: File already exists: {args.path}", state)
        if args.file_text is None:
            return _format_obs("Error: file_text required for create", state)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(args.file_text, encoding="utf-8")
        state["open_file"] = str(path)
        state["first_line"] = "0"
        state["working_dir"] = str(Path.cwd())
        _save_state(state)
        return _format_obs("File created.", state)

    if args.command == "str_replace":
        if not path.exists() or not path.is_file():
            return _format_obs(f"Error: {args.path} not found", state)
        if args.old_str is None:
            return _format_obs("Error: old_str required for str_replace", state)
        content = path.read_text(encoding="utf-8", errors="replace")
        matches = content.count(args.old_str)
        if matches == 0:
            return _format_obs(
                "Error: old_str not found in file. Match must be exact including whitespace.",
                state,
            )
        if matches > 1:
            return _format_obs(
                "Error: old_str is not unique in file. Please include more context to make it unique.",
                state,
            )
        _save_undo(path, content)
        path.write_text(content.replace(args.old_str, args.new_str or "", 1), encoding="utf-8")
        return _format_obs("Replacement applied.", state)

    if args.command == "insert":
        if not path.exists() or not path.is_file():
            return _format_obs(f"Error: {args.path} not found", state)
        if args.new_str is None or args.insert_line is None:
            return _format_obs("Error: new_str and insert_line required for insert", state)
        content = path.read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines()
        if args.insert_line < 0 or args.insert_line > len(lines):
            return _format_obs("Error: insert_line out of range", state)
        _save_undo(path, content)
        insert_lines = args.new_str.split("\n")
        new_lines = lines[: args.insert_line] + insert_lines + lines[args.insert_line :]
        new_content = "\n".join(new_lines) + ("\n" if content.endswith("\n") or args.new_str.endswith("\n") else "")
        path.write_text(new_content, encoding="utf-8")
        return _format_obs("Insert applied.", state)

    if args.command == "undo_edit":
        backup = _load_undo(path)
        if backup is None:
            return _format_obs("Error: no previous edit to undo for this file", state)
        path.write_text(backup, encoding="utf-8")
        return _format_obs("Undo applied.", state)

    return _format_obs(f"Error: Unknown command {args.command}", state)


def cmd_submit(args):
    state = _load_state()
    wd = state.get("working_dir") or os.getcwd()
    # Prefer binary-safe diff similar to many SWE runs.
    cmd = [
        "bash",
        "-lc",
        f'cd "{wd}" && git add -N . >/dev/null 2>&1 || true; git -c core.fileMode=false diff --binary --no-color',
    ]
    rc, out = _run_capture(cmd)
    if rc != 0 and not out.strip():
        out = "No changes to submit."
    state["working_dir"] = str(Path.cwd())
    _save_state(state)
    return _format_obs(_clip(out.strip() or "No changes to submit."), state)


def _build_parser():
    p = argparse.ArgumentParser(prog="swe_tool")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("open")
    sp.add_argument("path")
    sp.add_argument("line_number", nargs="?", type=int)
    sp.set_defaults(_fn=cmd_open)

    sp = sub.add_parser("goto")
    sp.add_argument("line_number", type=int)
    sp.set_defaults(_fn=cmd_goto)

    sp = sub.add_parser("scroll_up")
    sp.set_defaults(_fn=lambda a: cmd_scroll(a, "up"))

    sp = sub.add_parser("scroll_down")
    sp.set_defaults(_fn=lambda a: cmd_scroll(a, "down"))

    sp = sub.add_parser("search_dir")
    sp.add_argument("search_term")
    sp.add_argument("dir", nargs="?")
    sp.set_defaults(_fn=cmd_search_dir)

    sp = sub.add_parser("search_file")
    sp.add_argument("search_term")
    sp.add_argument("file", nargs="?")
    sp.set_defaults(_fn=cmd_search_file)

    sp = sub.add_parser("find_file")
    sp.add_argument("file_name")
    sp.add_argument("dir", nargs="?")
    sp.set_defaults(_fn=cmd_find_file)

    sp = sub.add_parser("create")
    sp.add_argument("filename")
    sp.set_defaults(_fn=cmd_create)

    sp = sub.add_parser("edit")
    sp.add_argument("start_line", type=int)
    sp.add_argument("end_line", type=int)
    sp.add_argument("replacement_text")
    sp.set_defaults(_fn=cmd_edit)

    sp = sub.add_parser("insert")
    sp.add_argument("text")
    sp.add_argument("line", nargs="?", type=int)
    sp.set_defaults(_fn=cmd_insert)

    sp = sub.add_parser("replace_in_window")
    sp.add_argument("search")
    sp.add_argument("replace")
    sp.add_argument("replace_all", nargs="?", default=False, type=lambda x: str(x).lower() in ("1", "true", "yes"))
    sp.set_defaults(_fn=cmd_replace_in_window)

    sp = sub.add_parser("str_replace_editor")
    sp.add_argument("command")
    sp.add_argument("path")
    sp.add_argument("--file_text")
    sp.add_argument("--old_str")
    sp.add_argument("--new_str")
    sp.add_argument("--insert_line", type=int)
    sp.add_argument("--view_range", nargs="*", type=int)
    sp.set_defaults(_fn=cmd_str_replace_editor)

    sp = sub.add_parser("submit")
    sp.set_defaults(_fn=cmd_submit)

    return p


def main(argv):
    p = _build_parser()
    args = p.parse_args(argv)
    out = args._fn(args)
    sys.stdout.write(out + ("\n" if not out.endswith("\n") else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

