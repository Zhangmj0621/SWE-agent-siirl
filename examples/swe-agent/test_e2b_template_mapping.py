"""
Standalone test: verify E2B template mapping and sandbox creation.
No RL framework, no data loading — just tests that the template name
resolves correctly and the sandbox starts.

Usage:
    export E2B_API_KEY=<your-key>
    # If private deployment:
    # export E2B_API_URL=https://your-private-e2b.example.com
    python examples/swe-agent/test_e2b_template_mapping.py
"""

import asyncio
import os
import sys


def resolve_template(image_name: str, strip_prefix: str = "", replace_suffix: dict = None) -> str:
    """Reproduce _resolve_template logic from e2b.py with our new config options."""
    # Step 1: auto-alias (same as _e2b_template_alias_for_docker_image)
    alias = (
        image_name.replace("/", "-")
        .replace(".", "-")
        .replace("_", "-")
        .replace(":", "-")
        .lower()
    )
    # Step 2: strip prefix
    if strip_prefix and alias.startswith(strip_prefix):
        alias = alias[len(strip_prefix):]
    # Step 3: replace suffix
    if replace_suffix:
        for old_suffix, new_suffix in replace_suffix.items():
            if alias.endswith(old_suffix):
                alias = alias[: -len(old_suffix)] + new_suffix
                break
    # Step 4: truncate
    if len(alias) > 64:
        alias = alias[:64]
    return alias


async def test_sandbox(template_name: str):
    """Try to create and use an E2B sandbox with the given template."""
    from e2b import AsyncSandbox

    api_key = os.environ.get("E2B_API_KEY")
    api_url = os.environ.get("E2B_API_URL")
    if not api_key:
        print("ERROR: E2B_API_KEY not set")
        sys.exit(1)

    kwargs = {"template": template_name, "api_key": api_key, "timeout": 300}
    if api_url:
        kwargs["api_url"] = api_url

    print(f"Creating sandbox with template: {template_name}")
    print(f"  API URL: {api_url or '(default)'}")

    try:
        sandbox = await AsyncSandbox.beta_create(**kwargs)
    except TypeError:
        # Older SDK versions may not support some kwargs
        for key in ("force_http", "auto_pause", "allow_internet_access"):
            kwargs.pop(key, None)
        sandbox = await AsyncSandbox.beta_create(**kwargs)

    print(f"Sandbox created! ID: {sandbox.sandbox_id}")

    # Run a quick smoke test
    result = await sandbox.commands.run("cat /etc/os-release | head -3", timeout=30, user="root")
    print(f"OS info:\n{result.stdout}")

    result = await sandbox.commands.run("ls /testbed 2>/dev/null && echo 'testbed exists' || echo 'no testbed'", timeout=30, user="root")
    print(f"Testbed check: {result.stdout.strip()}")

    result = await sandbox.commands.run("cd /testbed && git log --oneline -1 2>/dev/null || echo 'no git repo'", timeout=30, user="root")
    print(f"Git status: {result.stdout.strip()}")

    await sandbox.kill()
    print("Sandbox killed. Test PASSED.")


def main():
    # --- Config (same as sweagent_config_e2b_swebench_verified.yaml) ---
    strip_prefix = "swebench-"
    replace_suffix = {"-latest": "-2c4g"}

    # --- Test instance: sympy__sympy-23262 ---
    instance_id = "sympy__sympy-23262"

    # Simulate what make_test_spec produces as instance_image_key
    iid_lower = instance_id.lower()
    docker_compatible = iid_lower.replace("__", "_1776_")
    image_key = f"swebench/sweb.eval.x86_64.{docker_compatible}:latest"

    print("=" * 60)
    print("E2B Template Mapping Test")
    print("=" * 60)
    print(f"Instance ID:     {instance_id}")
    print(f"Docker image key: {image_key}")

    # Resolve template
    template = resolve_template(image_key, strip_prefix=strip_prefix, replace_suffix=replace_suffix)
    print(f"Resolved template: {template}")

    # Verify it matches expected
    expected = "sweb-eval-x86-64-sympy-1776-sympy-23262-2c4g"
    if template == expected:
        print(f"Mapping CORRECT (matches expected: {expected})")
    else:
        print(f"Mapping MISMATCH!")
        print(f"  Expected: {expected}")
        print(f"  Got:      {template}")
        sys.exit(1)

    print()
    print("--- Now testing sandbox creation ---")
    asyncio.run(test_sandbox(template))


if __name__ == "__main__":
    main()
