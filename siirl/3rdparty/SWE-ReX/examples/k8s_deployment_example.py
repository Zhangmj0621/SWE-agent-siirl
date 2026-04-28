"""
Example of using K8sDeployment with SWE-ReX.

This example demonstrates how to create and use a Kubernetes-based deployment
for running SWE-ReX remotely in a Kubernetes pod.

Requirements:
- kubectl must be installed and configured
- Access to a Kubernetes cluster
- The specified container image must be available to the cluster
"""

import asyncio

from swerex.deployment.config import K8sDeploymentConfig
from swerex.deployment.k8s import K8sDeployment
from swerex.runtime.abstract import Command


async def main():
    # Create a K8s deployment configuration
    config = K8sDeploymentConfig(
        image="python:3.11",  # Container image to use
        namespace="default",  # Kubernetes namespace
        port=None,  # Auto-assign a free port for port forwarding
        resource_requests={  # Resource requests for the pod
            "memory": "2Gi",
            "cpu": "1",
        },
        resource_limits={  # Resource limits for the pod
            "memory": "4Gi",
            "cpu": "2",
        },
    )

    # Create deployment from config
    deployment = K8sDeployment.from_config(config)

    try:
        # Start the deployment (creates pod and sets up port forwarding)
        print("Starting K8s deployment...")
        await deployment.start()
        print(f"Deployment started! Pod name: {deployment.pod_name}")

        # Get the runtime
        runtime = deployment.runtime

        # Execute a simple command
        print("\nExecuting command: echo 'Hello from K8s!'")
        result = await runtime.execute(Command(command="echo 'Hello from K8s!'"))
        print(f"Output: {result.output}")
        print(f"Exit code: {result.exit_code}")

        # Create a bash session for interactive commands
        print("\nCreating bash session...")
        session_id = await runtime.create_session()
        print(f"Session created: {session_id}")

        # Run commands in the session
        print("\nRunning commands in session...")
        result = await runtime.run_in_session(session_id, "pwd")
        print(f"Current directory: {result.output.strip()}")

        result = await runtime.run_in_session(session_id, "python --version")
        print(f"Python version: {result.output.strip()}")

        # Close the session
        await runtime.close_session(session_id)
        print("\nSession closed")

    finally:
        # Always clean up
        print("\nStopping deployment...")
        await deployment.stop()
        print("Deployment stopped")


if __name__ == "__main__":
    asyncio.run(main())