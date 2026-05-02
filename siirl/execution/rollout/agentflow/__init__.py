from types import MethodType

from .base import AgentFlow, Model, ModelResponse, Sample
from .utils import import_any

__all__ = ["Model", "AgentFlow", "Sample", "ModelResponse", "load_agentflow"]

BUILTIN_FLOW = {"swe": ".swe:agentflow"}


def load_agentflow(config: dict) -> AgentFlow:
    """使用配置加载 TaskFlow

    Args:
        config (dict): 整合在 RL 框架的配置管理中（例如配置文件 `scaffold` 键的值）

    Returns:
        TaskFlow: 一个 TaskFlow 实例

    Note:
        Model 不在这里传入，而是在 preprocess(sample, model) 时传入。


    示例 Config structure:
    {
        "name": "SimpleAgentFlow",  # name or path; must be specified
        "preprocess_fn": "module.path:function_name",  # optional override
        "generate_fn": "module.path:function_name",  # optional override
        "reward_fn": "example_evaluator:parse_and_eval_reward",  # injected evaluator
        "python_path": ["/root/math/custom_agents"],  # optional import paths
    }
    """
    python_path = config.get("python_path")
    agent_class = import_any(config["name"], builtin=BUILTIN_FLOW, path=python_path)

    preprocess_fn = import_any(config.get("preprocess_fn"), path=python_path)
    generate_fn = import_any(config.get("generate_fn"), path=python_path)
    reward_fn = import_any(config.get("reward_fn"), path=python_path)

    # Instantiate agent without model (model will be provided per sample)
    agent: AgentFlow = agent_class(config)

    # compose it
    if preprocess_fn is not None:
        agent.preprocess = MethodType(preprocess_fn, agent)
    if generate_fn is not None:
        agent.generate = MethodType(generate_fn, agent)
    if reward_fn is not None:
        agent.reward = MethodType(reward_fn, agent)
    return agent
