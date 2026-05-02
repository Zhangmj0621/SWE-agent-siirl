"""Test script to debug K8sBashSession initialization issues."""

import json
import logging
import os
import subprocess
import sys
import time
import uuid

# Setup logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class K8sSessionTester:
    """Test K8s bash session initialization step by step."""

    def __init__(
        self,
        namespace: str,
        image: str = "python:3.11",
        pod_name_prefix: str = "test-k8s-session",
        container: str = "main",
        kubeconfig: str = None,
        context: str = None,
    ):
        self.namespace = namespace
        self.image = image
        self.pod_name_prefix = pod_name_prefix
        self.container = container
        self.kubeconfig = kubeconfig or os.getenv("KUBECONFIG")
        self.context = context
        self.pod_name = None

    def _run_kubectl(self, *args, timeout: int = 30) -> tuple[bytes, bytes, int]:
        """Run kubectl command and return stdout, stderr, returncode."""
        cmd = ["kubectl"]
        if self.kubeconfig:
            cmd.extend(["--kubeconfig", self.kubeconfig])
        if self.context:
            cmd.extend(["--context", self.context])
        cmd.extend(args)

        logger.info(f"Running: {' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = proc.communicate(timeout=timeout)
        return stdout, stderr, proc.returncode

    def create_pod(self, timeout: float = 180.0) -> bool:
        """Create a test pod."""
        logger.info("=" * 60)
        logger.info("CREATING TEST POD")
        logger.info("=" * 60)

        # Generate unique pod name
        suffix = uuid.uuid4().hex[:8]
        self.pod_name = f"{self.pod_name_prefix}-{suffix}"
        logger.info(f"Pod name: {self.pod_name}")

        # Build pod spec
        pod_spec = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": self.pod_name,
                "namespace": self.namespace,
            },
            "spec": {
                "containers": [
                    {
                        "name": self.container,
                        "image": self.image,
                        "command": ["sleep", "infinity"],
                        "imagePullPolicy": "IfNotPresent",
                    }
                ],
                "restartPolicy": "Never",
            },
        }

        # Create pod
        logger.info(f"Creating pod with image: {self.image}")
        stdout, stderr, returncode = self._run_kubectl("apply", "-f", "-", input=json.dumps(pod_spec).encode())

        if returncode != 0:
            logger.error("❌ Failed to create pod")
            logger.error(f"stderr: {stderr.decode()}")
            return False

        logger.info("✅ Pod creation command succeeded")

        # Wait for pod to be ready
        logger.info(f"Waiting for pod to be ready (timeout: {timeout}s)")
        start_time = time.monotonic()

        while True:
            elapsed = time.monotonic() - start_time
            if elapsed > timeout:
                logger.error("❌ Timeout waiting for pod to be ready")
                self.delete_pod()
                return False

            stdout, stderr, returncode = self._run_kubectl("get", "pod", "-n", self.namespace, self.pod_name, "-o", "json")

            if returncode != 0:
                logger.error("❌ Failed to get pod status")
                self.delete_pod()
                return False

            pod_info = json.loads(stdout.decode())
            phase = pod_info.get("status", {}).get("phase", "")

            if phase == "Running":
                container_statuses = pod_info.get("status", {}).get("containerStatuses", [])
                if container_statuses and container_statuses[0].get("ready", False):
                    logger.info("✅ Pod is ready")
                    return True

            elif phase in ["Failed", "Succeeded"]:
                logger.error(f"❌ Pod entered phase {phase}")
                self.delete_pod()
                return False

            logger.info(f"Pod status: {phase}, waiting...")
            time.sleep(2)

    def delete_pod(self) -> bool:
        """Delete the test pod."""
        logger.info("=" * 60)
        logger.info("DELETING TEST POD")
        logger.info("=" * 60)

        if self.pod_name is None:
            logger.info("No pod to delete")
            return True

        stdout, stderr, returncode = self._run_kubectl("delete", "pod", "-n", self.namespace, self.pod_name, "--wait=false")

        if returncode != 0:
            logger.warning("⚠️  Failed to delete pod (may have been deleted already)")
            logger.warning(f"stderr: {stderr.decode()}")
            return True  # Not critical

        logger.info("✅ Pod deleted")
        self.pod_name = None
        return True

    def test_1_pod_exists(self) -> bool:
        """Test 1: Check if pod exists and is running."""
        logger.info("=" * 60)
        logger.info("TEST 1: Check if pod exists")
        logger.info("=" * 60)

        if self.pod_name is None:
            logger.error("❌ No pod name set")
            return False

        stdout, stderr, returncode = self._run_kubectl("get", "pod", "-n", self.namespace, self.pod_name, "-o", "json")

        if returncode != 0:
            logger.error(f"❌ Pod {self.pod_name} does not exist")
            logger.error(f"stderr: {stderr.decode()}")
            return False

        pod_info = json.loads(stdout.decode())
        phase = pod_info.get("status", {}).get("phase", "")
        logger.info(f"✅ Pod exists, phase: {phase}")

        if phase != "Running":
            logger.error(f"❌ Pod is not running (phase: {phase})")
            return False

        logger.info("✅ Pod is running")
        return True

    def test_2_container_ready(self) -> bool:
        """Test 2: Check if container is ready."""
        logger.info("=" * 60)
        logger.info("TEST 2: Check if container is ready")
        logger.info("=" * 60)

        stdout, stderr, returncode = self._run_kubectl("get", "pod", "-n", self.namespace, self.pod_name, "-o", "json")

        if returncode != 0:
            logger.error("❌ Failed to get pod info")
            return False

        pod_info = json.loads(stdout.decode())
        container_statuses = pod_info.get("status", {}).get("containerStatuses", [])

        for cs in container_statuses:
            if cs.get("name") == self.container:
                ready = cs.get("ready", False)
                state = cs.get("state", {})
                logger.info(f"Container ready: {ready}, state: {state}")
                if ready:
                    logger.info("✅ Container is ready")
                    return True
                else:
                    logger.error("❌ Container is not ready")
                    return False

        logger.error(f"❌ Container {self.container} not found")
        return False

    def test_3_exec_simple_command(self) -> bool:
        """Test 3: Test simple kubectl exec command."""
        logger.info("=" * 60)
        logger.info("TEST 3: Test simple kubectl exec")
        logger.info("=" * 60)

        stdout, stderr, returncode = self._run_kubectl(
            "exec", "-n", self.namespace, self.pod_name, "-c", self.container, "--", "echo", "hello"
        )

        if returncode != 0:
            logger.error("❌ kubectl exec failed")
            logger.error(f"stderr: {stderr.decode()}")
            return False

        output = stdout.decode().strip()
        if output == "hello":
            logger.info("✅ Simple exec works")
            return True
        else:
            logger.error(f"❌ Unexpected output: {output}")
            return False

    def test_4_bash_exists(self) -> bool:
        """Test 4: Check if bash exists in container."""
        logger.info("=" * 60)
        logger.info("TEST 4: Check if bash exists")
        logger.info("=" * 60)

        stdout, stderr, returncode = self._run_kubectl(
            "exec", "-n", self.namespace, self.pod_name, "-c", self.container, "--", "which", "bash"
        )

        if returncode != 0:
            logger.error("❌ bash not found")
            logger.info("ℹ️  Trying sh instead...")
            return self.test_4b_sh_exists()

        bash_path = stdout.decode().strip()
        logger.info(f"✅ bash found at: {bash_path}")
        return True

    def test_4b_sh_exists(self) -> bool:
        """Test 4b: Check if sh exists (fallback)."""
        stdout, stderr, returncode = self._run_kubectl(
            "exec", "-n", self.namespace, self.pod_name, "-c", self.container, "--", "which", "sh"
        )

        if returncode != 0:
            logger.error("❌ sh not found either")
            return False

        logger.info(f"✅ sh found at: {stdout.decode().strip()}")
        logger.info("⚠️  Only sh is available, bash is missing")
        return False

    def test_5_bash_exec(self) -> bool:
        """Test 5: Test bash -c command execution."""
        logger.info("=" * 60)
        logger.info("TEST 5: Test bash -c command")
        logger.info("=" * 60)

        stdout, stderr, returncode = self._run_kubectl(
            "exec", "-n", self.namespace, self.pod_name, "-c", self.container, "--", "bash", "-c", "echo 'test'"
        )

        if returncode != 0:
            logger.error("❌ bash -c failed")
            logger.error(f"stderr: {stderr.decode()}")
            return False

        output = stdout.decode().strip()
        if output == "test":
            logger.info("✅ bash -c works")
            return True
        else:
            logger.error(f"❌ Unexpected output: {output}")
            return False

    def test_6_interactive_bash(self) -> bool:
        """Test 6: Test interactive bash with pexpect."""
        logger.info("=" * 60)
        logger.info("TEST 6: Test interactive bash with pexpect")
        logger.info("=" * 60)

        try:
            import pexpect
        except ImportError:
            logger.error("❌ pexpect not installed")
            logger.info("Install with: pip install pexpect")
            return False

        # Build the kubectl exec command
        cmd_parts = ["kubectl"]
        if self.kubeconfig:
            cmd_parts.extend(["--kubeconfig", self.kubeconfig])
        if self.context:
            cmd_parts.extend(["--context", self.context])
        cmd_parts.extend(["exec", "-i", "-n", self.namespace, self.pod_name, "-c", self.container, "--", "bash", "-i"])
        cmd = " ".join(cmd_parts)

        logger.info(f"Starting: {cmd}")
        try:
            process = pexpect.spawn(
                cmd,
                encoding="utf-8",
                codec_errors="backslashreplace",
                echo=False,
                timeout=10,
            )
            logger.info("✅ pexpect.spawn() succeeded")
        except Exception as e:
            logger.error(f"❌ pexpect.spawn() failed: {e}")
            return False

        # Wait a bit for bash to start
        time.sleep(0.5)

        # Check if process is alive
        if not process.isalive():
            logger.error("❌ Process died immediately")
            logger.error(f"Process output: {process.before}")
            return False

        logger.info("✅ Process is alive")

        # Try to send a simple command
        logger.info("Sending: echo test123")
        process.sendline("echo test123")

        try:
            # Wait for output (just wait a bit)
            time.sleep(1)
            output = process.before
            logger.info(f"Process output: {output}")
            if "test123" in (output or ""):
                logger.info("✅ Command executed successfully")
            else:
                logger.warning("⚠️  Output doesn't contain expected text")
        except pexpect.TIMEOUT:
            logger.error("❌ Timeout waiting for output")
            return False

        # Close the process
        process.close()
        return True

    def test_7_ps1_test(self) -> bool:
        """Test 7: Test PS1 prompt setting."""
        logger.info("=" * 60)
        logger.info("TEST 7: Test PS1 prompt setting")
        logger.info("=" * 60)

        try:
            import pexpect
        except ImportError:
            return False

        cmd_parts = ["kubectl"]
        if self.kubeconfig:
            cmd_parts.extend(["--kubeconfig", self.kubeconfig])
        if self.context:
            cmd_parts.extend(["--context", self.context])
        cmd_parts.extend(["exec", "-i", "-n", self.namespace, self.pod_name, "-c", self.container, "--", "bash", "-i"])
        cmd = " ".join(cmd_parts)

        PS1 = "SHELLPS1PREFIX"

        process = pexpect.spawn(
            cmd,
            encoding="utf-8",
            codec_errors="backslashreplace",
            echo=False,
            timeout=10,
        )

        time.sleep(0.5)

        if not process.isalive():
            logger.error("❌ Process died")
            return False

        # Set PS1
        init_cmd = f"export PS1='{PS1}'"
        logger.info(f"Sending: {init_cmd}")
        process.sendline(init_cmd)

        try:
            process.expect(PS1, timeout=5)
            logger.info("✅ PS1 prompt set successfully")
            process.close()
            return True
        except pexpect.TIMEOUT:
            logger.error("❌ Timeout waiting for PS1 prompt")
            logger.error(f"Process.before: {process.before}")
            logger.error(f"Process.isalive(): {process.isalive()}")
            logger.error(f"Process.exitstatus: {process.exitstatus}")

            # Try to read remaining output
            if process.isalive():
                try:
                    remaining = process.read_nonblocking(size=2000, timeout=0.5)
                    logger.error(f"Remaining output: {remaining}")
                except Exception:
                    pass

            process.close()
            return False

    def run_all_tests(self):
        """Run all tests in sequence."""
        tests = [
            self.test_1_pod_exists,
            self.test_2_container_ready,
            self.test_3_exec_simple_command,
            self.test_4_bash_exists,
            self.test_5_bash_exec,
            self.test_6_interactive_bash,
            self.test_7_ps1_test,
        ]

        results = []
        for i, test in enumerate(tests, 1):
            try:
                result = test()
                results.append(result)
                if not result:
                    logger.info(f"\n❌ Test {i} failed, stopping")
                    break
                logger.info(f"\n✅ Test {i} passed\n")
            except Exception as e:
                logger.error(f"\n❌ Test {i} raised exception: {e}\n")
                import traceback

                traceback.print_exc()
                results.append(False)
                break

        logger.info("=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        for i, result in enumerate(results, 1):
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"Test {i}: {status}")

        return all(results)


def main():
    """Main entry point."""
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python test_k8s_session.py <namespace> [options]")
        print("\nOptions:")
        print("  --image <image>       Container image (default: python:3.11)")
        print("  --pod-prefix <prefix>  Pod name prefix (default: test-k8s-session)")
        print("  --container <name>    Container name (default: main)")
        print("  --kubeconfig <path>   Path to kubeconfig")
        print("  --context <name>      Kubernetes context")
        print("\nExample:")
        print("  python test_k8s_session.py default")
        print("  python test_k8s_session.py default --image ubuntu:22.04")
        sys.exit(1)

    namespace = sys.argv[1]

    # Parse optional arguments
    image = "python:3.11"
    pod_prefix = "test-k8s-session"
    container = "main"
    kubeconfig = None
    context = None

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--image" and i + 1 < len(sys.argv):
            image = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--pod-prefix" and i + 1 < len(sys.argv):
            pod_prefix = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--container" and i + 1 < len(sys.argv):
            container = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--kubeconfig" and i + 1 < len(sys.argv):
            kubeconfig = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--context" and i + 1 < len(sys.argv):
            context = sys.argv[i + 1]
            i += 2
        else:
            print(f"Unknown argument: {sys.argv[i]}")
            sys.exit(1)

    logger.info("Test configuration:")
    logger.info(f"  Namespace: {namespace}")
    logger.info(f"  Image: {image}")
    logger.info(f"  Pod prefix: {pod_prefix}")
    logger.info(f"  Container: {container}")
    logger.info(f"  Kubeconfig: {kubeconfig or os.getenv('KUBECONFIG', 'default')}")
    logger.info(f"  Context: {context or 'default'}")

    # Create tester
    tester = K8sSessionTester(
        namespace=namespace,
        image=image,
        pod_name_prefix=pod_prefix,
        container=container,
        kubeconfig=kubeconfig,
        context=context,
    )

    # Create pod
    logger.info("\n%s", "=" * 60)
    logger.info("STEP 1: Create test pod")
    logger.info("%s\n", "=" * 60)

    if not tester.create_pod():
        logger.error("Failed to create pod, exiting")
        sys.exit(1)

    # Run tests
    logger.info("\n%s", "=" * 60)
    logger.info("STEP 2: Run tests")
    logger.info("%s\n", "=" * 60)

    try:
        success = tester.run_all_tests()
    finally:
        # Always cleanup
        logger.info("\n%s", "=" * 60)
        logger.info("STEP 3: Cleanup")
        logger.info("%s\n", "=" * 60)
        tester.delete_pod()

    if success:
        logger.info("\n✅ All tests passed!")
    else:
        logger.info("\n❌ Some tests failed")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
