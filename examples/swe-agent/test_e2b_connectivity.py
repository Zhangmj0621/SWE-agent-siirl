#!/usr/bin/env python3
"""
Minimal E2B connectivity test — no RL, no model, no Ray.
Just verifies: can we create a sandbox, run commands, and clean up?

Usage:
    export E2B_API_KEY=<your-key>
    # For private deployment:
    # export E2B_API_URL=https://your-private-e2b-endpoint
    python examples/swe-agent/test_e2b_connectivity.py
"""

import os
import sys
import time

def main():
    api_key = os.environ.get("E2B_API_KEY")
    if not api_key:
        print("ERROR: E2B_API_KEY not set", file=sys.stderr)
        return 1

    try:
        from e2b import Sandbox
    except ImportError:
        print("ERROR: pip install e2b", file=sys.stderr)
        return 1

    template = "sweb-eval-x86-64-astropy-1776-astropy-12907"
    print(f"[1/5] Creating sandbox from template: {template}")
    t0 = time.time()

    try:
        sandbox = Sandbox(template=template, timeout=300)
    except Exception as e:
        print(f"FAILED to create sandbox: {e}", file=sys.stderr)
        return 1

    print(f"      Sandbox created in {time.time() - t0:.1f}s, id={sandbox.sandbox_id}")

    print("[2/5] Running: echo HELLO_E2B")
    result = sandbox.commands.run("echo HELLO_E2B", timeout=30)
    print(f"      stdout: {result.stdout.strip()}")
    assert "HELLO_E2B" in result.stdout, "Echo test failed"

    print("[3/5] Checking /testbed exists")
    result = sandbox.commands.run("ls /testbed", timeout=30)
    if result.exit_code == 0:
        print(f"      /testbed contents (first 5 lines):")
        for line in result.stdout.strip().split("\n")[:5]:
            print(f"        {line}")
    else:
        print(f"      /testbed does NOT exist (exit_code={result.exit_code})")

    print("[4/5] Checking git repo status")
    result = sandbox.commands.run("cd /testbed && git rev-parse --short HEAD && git status -sb | head -5", timeout=30)
    print(f"      {result.stdout.strip()}")

    print("[5/5] Cleanup")
    sandbox.kill()
    print(f"      Sandbox killed. Total time: {time.time() - t0:.1f}s")

    print("\n✅ E2B connectivity test PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
