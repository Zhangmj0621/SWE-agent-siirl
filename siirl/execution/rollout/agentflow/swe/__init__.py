from ..base import Model, AgentFlow
from ..utils import import_any
from .base import SWESample, SWEAgentMeta
from .environment import ContainerEnvBuilder
from .agent import AgentBuilder
from .runtime import RuntimeBuilder

BUILTIN_PROVIDERS = {
    "agent": {"minisweagent": ".swe.agent.minisweagent:MiniSWEAgentBuilder"},
    "environment": {
        "k8s": ".swe.environment.k8s:K8sEnvBuilder",
        "k8rs": ".swe.environment.k8rs:K8rsEnvBuilder",
    },
    "runtime": {"swefactory": ".swe.runtime.swefactory:SWEFactoryBuiler"},
}


def agentflow(config: dict, model: Model) -> AgentFlow:
    """使用配置加载 TaskFlow

    Args:
        config (dict): 整合在 RL 框架的配置管理中（例如配置文件 `scaffold` 键的值）
        model (Model): Language Model （例如 SGLang router）

    Returns:
        TaskFlow: 一个 TaskFlow 实例


    示例 Config structure:
    {
        "name": "swe.agentflow",  # name or path; must be specified
        "agent": {"name":"swe.agent.minisweagent:MiniSWEAgentBuilder"} # agent config
        "environment": {"name":"swe.environment.k8s:K8sEnvBuilder"} # agent config
        "runtime": {"name":"swe.runtime.swefactory:function_name"} # agent config
        "python_path": ["/root/math/custom_agents"],  # optional import paths
    }
    """
    python_path = config.get("python_path")

    instances = []
    for name in ("agent", "environment", "runtime"):
        try:
            subconf = config[name]
            builtin = BUILTIN_PROVIDERS[name]
            cls = import_any(subconf["name"], builtin=builtin, path=python_path)
            if cls is None:
                raise ImportError(f"`{subconf['name']}` not found")
            instance = cls(subconf)
            instances.append(instance)
        except Exception as e:
            raise ImportError(f"Fail to initialize SWE agentflow {name} builder", e)

    return SWEAgentFlow(*instances, model=model)


class SWEAgentFlow(AgentFlow):
    def __init__(
        self,
        agent: AgentBuilder,
        env: ContainerEnvBuilder,
        runtime: RuntimeBuilder,
        model: Model,
    ):
        self.agent = agent
        self.env = env
        self.runtime = runtime
        self.model = model

    def preprocess(self, sample: dict) -> SWESample:
        """把数据集的一条数据 (dict) 处理成 Sample 对象；可能会 raise exception

        Args:
            sample (dict): 数据集的一条数据

        Returns:
            Sample: rollout 并 evaluate 的算例
        """
        data = self.runtime.parse_sampledata(sample)
        meta = SWEAgentMeta(data)
        s = SWESample(meta, self.model)
        runtime = self.runtime.build(s)
        agent = self.agent.build(s)
        meta.runtime = runtime
        meta.agent = agent
        return s

    async def generate(self, sample: SWESample):
        """运行 scaffold rollout / math solution generation

        Args:
            sample (Sample): 本 TaskFlow 兼容的算例
        """
        m = sample.m
        async with await self.env.start(m.data.container_args) as env:
            await m.runtime.bootstrap(env)
            await m.agent.run(env)
            await m.runtime.diff(env)

    async def reward(self, sample: SWESample):
        """运行 evaluate / verification

        Args:
            sample (Sample): 本 TaskFlow 兼容的算例
        """
        m = sample.m
        async with await self.env.start(m.data.container_args) as env:
            await m.runtime.eval(env)
