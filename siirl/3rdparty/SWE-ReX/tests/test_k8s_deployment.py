import pytest

from swerex.deployment.config import K8sDeploymentConfig
from swerex.deployment.k8s import K8sDeployment


def test_k8s_deployment_config():
    """Test K8sDeploymentConfig creation and validation."""
    config = K8sDeploymentConfig(
        image="python:3.11",
        namespace="test-namespace",
        port=8080,
    )
    
    assert config.image == "python:3.11"
    assert config.namespace == "test-namespace"
    assert config.port == 8080
    assert config.type == "k8s"
    assert config.startup_timeout == 180.0
    assert config.resource_requests == {"memory": "4Gi", "cpu": "2"}
    assert config.resource_limits == {"memory": "4Gi", "cpu": "2"}
    assert config.exec_shell == ["/bin/sh", "-c"]


def test_k8s_deployment_config_defaults():
    """Test K8sDeploymentConfig with default values."""
    config = K8sDeploymentConfig(image="python:3.11")
    
    assert config.namespace == "default"
    assert config.port is None
    assert config.type == "k8s"


def test_k8s_deployment_config_custom_resources():
    """Test K8sDeploymentConfig with custom resources."""
    config = K8sDeploymentConfig(
        image="python:3.11",
        resource_requests={"memory": "2Gi", "cpu": "1"},
        resource_limits={"memory": "8Gi", "cpu": "4"},
    )
    
    assert config.resource_requests == {"memory": "2Gi", "cpu": "1"}
    assert config.resource_limits == {"memory": "8Gi", "cpu": "4"}


def test_k8s_deployment_from_config():
    """Test creating K8sDeployment from config."""
    config = K8sDeploymentConfig(
        image="python:3.11",
        namespace="test",
    )
    
    deployment = K8sDeployment.from_config(config)
    assert deployment is not None
    assert deployment.pod_name is None  # Not started yet


def test_k8s_deployment_get_deployment():
    """Test get_deployment method."""
    config = K8sDeploymentConfig(image="python:3.11")
    deployment = config.get_deployment()
    
    assert isinstance(deployment, K8sDeployment)


def test_k8s_deployment_config_extra_forbid():
    """Test that extra fields are forbidden."""
    with pytest.raises(Exception):  # Will raise ValidationError
        K8sDeploymentConfig(
            image="python:3.11",
            invalid_field="should_fail"
        )


def test_k8s_deployment_pod_name_generation():
    """Test that pod name is generated correctly."""
    deployment = K8sDeployment(image="python:3.11")
    pod_name = deployment._get_pod_name()
    
    # Should be lowercase, alphanumeric with hyphens, max 63 chars
    assert pod_name.startswith("swerex-")
    assert len(pod_name) <= 63
    assert pod_name.islower() or '-' in pod_name
    assert not pod_name.endswith('-')
    assert not pod_name.endswith('.')