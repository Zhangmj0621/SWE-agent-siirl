# Kubernetes Deployment

The K8s deployment allows you to run SWE-ReX in Kubernetes pods with port-forwarding for local access.

## Basic Usage

```python
from swerex.deployment import get_deployment
from swerex.deployment.config import K8sDeploymentConfig

# Create configuration
config = K8sDeploymentConfig(
    image="python:3.11",
    namespace="default",
)

# Get deployment
deployment = get_deployment(config)

# Start the deployment
await deployment.start()

# Use the runtime
runtime = deployment.runtime
result = await runtime.execute("echo 'Hello from Kubernetes!'")
print(result.output)

# Clean up
await deployment.stop()
```

## Configuration Options

### Basic Settings

- `image` (str): Container image to use (default: `"python:3.11"`)
- `namespace` (str): Kubernetes namespace (default: `"default"`)
- `port` (int | None): Local port for port-forwarding. If None, a free port is automatically found
- `startup_timeout` (float): Time to wait for runtime to start in seconds (default: `180.0`)

### Resource Management

- `resource_requests` (dict): Resource requests for the pod
  - Default: `{"memory": "4Gi", "cpu": "2"}`
- `resource_limits` (dict): Resource limits for the pod
  - Default: `{"memory": "4Gi", "cpu": "2"}`

### Advanced Settings

- `exec_shell` (list[str]): Shell to use for running commands (default: `["/bin/sh", "-c"]`)

## Example with Custom Resources

```python
config = K8sDeploymentConfig(
    image="python:3.11",
    namespace="swerex-ns",
    resource_requests={"memory": "2Gi", "cpu": "1"},
    resource_limits={"memory": "8Gi", "cpu": "4"},
)

deployment = get_deployment(config)
await deployment.start()
# ... use deployment ...
await deployment.stop()
```

## How It Works

1. **Pod Creation**: Creates a Kubernetes pod with the specified image and resources
2. **Port Forwarding**: SetsNow let me create the documentation file:

<write_to_file>
<path>SWE-ReX/docs/k8s_deployment.md</path>
<content>
# Kubernetes Deployment

The K8s deployment allows you to run SWE-ReX in Kubernetes pods with port-forwarding for local access.

## Basic Usage

```python
from swerex.deployment import get_deployment
from swerex.deployment.config import K8sDeploymentConfig

# Create configuration
config = K8sDeploymentConfig(
    image="python:3.11",
    namespace="default",
)

# Get deployment
deployment = get_deployment(config)

# Start the deployment
await deployment.start()

# Use the runtime
runtime = deployment.runtime
result = await runtime.execute("echo 'Hello from Kubernetes!'")
print(result.output)

# Clean up
await deployment.stop()
```

## Configuration Options

### Basic Settings

- `image` (str): Container image to use (default: `"python:3.11"`)
- `namespace` (str): Kubernetes namespace (default: `"default"`)
- `port` (int | None): Local port for port-forwarding. If None, a free port is automatically found
- `startup_timeout` (float): Time to wait for runtime to start in seconds (default: `180.0`)

### Resource Management

- `resource_requests` (dict): Resource requests for the pod
  - Default: `{"memory": "4Gi", "cpu": "2"}`
- `resource_limits` (dict): Resource limits for the pod
  - Default: `{"memory": "4Gi", "cpu": "2"}`

### Advanced Settings

- `exec_shell` (list[str]): Shell to use for running commands (default: `["/bin/sh", "-c"]`)

## Example with Custom Resources

```python
config = K8sDeploymentConfig(
    image="python:3.11",
    namespace="swerex-ns",
    resource_requests={"memory": "2Gi", "cpu": "1"},
    resource_limits={"memory": "8Gi", "cpu": "4"},
)

deployment = get_deployment(config)
await deployment.start()
# ... use deployment ...
await deployment.stop()
```

## How It Works

1. **Pod Creation**: Creates a Kubernetes pod with the specified image and resources
2. **Port Forwarding**: Sets up `kubectl port-forward` from a local port to the pod's port 8000
3. **Runtime Connection**: Connects to the runtime via the forwarded port
4. **Cleanup**: On stop, terminates port forwarding and deletes the pod

## Prerequisites

- `kubectl` must be installed and configured
- You must have access to a Kubernetes cluster
- The specified namespace must exist
- You must have permissions to create/delete pods and setup port-forwarding

## Comparison with Docker Deployment

| Feature | Docker | Kubernetes |
|---------|--------|------------|
| Infrastructure | Local container runtime | Kubernetes cluster |
| Scaling | Limited to local resources | Can scale across cluster |
| Port Management | Direct port binding | Port forwarding |
| Resource Limits | Docker resource constraints | K8s resource quotas |
| Networking | Docker networks | K8s networking |

## Troubleshooting

### Pod doesn't start

Check pod status:
```bash
kubectl get pod <pod-name> -n <namespace>
kubectl describe pod <pod-name> -n <namespace>
kubectl logs <pod-name> -n <namespace>
```

### Port forwarding fails

- Ensure kubectl can connect to the cluster
- Check if the port is already in use
- Verify pod is running and ready

### Resource quota exceeded

Reduce resource requests/limits in config:
```python
config = K8sDeploymentConfig(
    resource_requests={"memory": "1Gi", "cpu": "0.5"},
    resource_limits={"memory": "2Gi", "cpu": "1"},
)