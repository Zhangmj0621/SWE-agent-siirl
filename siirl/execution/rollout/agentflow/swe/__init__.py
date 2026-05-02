import asyncio
from concurrent.futures import ThreadPoolExecutor

from loguru import logger

from siirl.execution.rollout.utils import EnvCreateError

from ..base import AgentFlow, Model
from ..utils import import_any
from .agent import AgentBuilder
from .base import SWEAgentMeta, SWESample
from .environment import ContainerEnvBuilder
from .runtime import RuntimeBuilder


def _should_eval(exit_status: str | None, is_validate: bool) -> tuple[bool, "SWESample.Status | None"]:
    """按 exit_status 判断是否需要走 reward/eval，并给出应设的 Sample.Status。

    对齐 Agentic_RL (SLIME)：只有 ABORTED 才会跳过 eval；TRUNCATED 依然会跑 eval，
    只是在下游训练时通过 status 将其排除出 loss。

    - train:
        - "submitted"                               → 跑 eval，不覆盖 status（由 reward 置 COMPLETED/FAILED）
        - "submitted (exit_context)" / "exit_context" → 跑 eval，status 置为 TRUNCATED
        - 其它                                       → 跳过 eval，status 置为 ABORTED
    - validate:
        - 以 "submitted" 开头                        → 跑 eval，不覆盖 status
        - 其它                                       → 跳过 eval，status 置为 ABORTED

    Returns:
        (do_eval, status_override)
    """
    if is_validate:
        if exit_status and exit_status.startswith("submitted"):
            return True, None
        return False, SWESample.Status.ABORTED

    # train
    if exit_status == "submitted":
        return True, None
    if exit_status in ("submitted (exit_context)", "exit_context"):
        return True, SWESample.Status.TRUNCATED
    return False, SWESample.Status.ABORTED


# Use original SWE-ReX/SWE-agent implementations via adapters
BUILTIN_PROVIDERS = {
    "agent": {
        "minisweagent": ".swe.agent.minisweagent:MiniSWEAgentBuilder",
        # 推荐：使用 RLTokenAgent (有完整的 token tracking 支持)
        "sweagent": ".swe.agent.sii_sweagent:RLTokenAgentBuilder",
        # 实验性：使用原始 SWE-agent + 适配器 (缺少 token tracking，仅用于非 RL 场景)
        "sweagent_original": ".swe.agent.swe_adapter:SWEAgentAdapterBuilder",
        # 旧名称（兼容）
        "sweagent_legacy": ".swe.agent.sii_sweagent:RLTokenAgentBuilder",
        # 使用 DefaultAgent + LiteLLMModel (通过 sglang API，用于 validate 阶段)
        "litellm_agent": ".swe.agent.sii_sweagent:LiteLLMAgentBuilder",
    },
    "environment": {
        "docker": ".swe.environment.docker:DockerEnvBuilder",
        # Use original SWE-ReX K8sDeployment via adapter (recommended)
        "k8s": ".swe.environment.k8s_adapter:K8sEnvAdapterBuilder",
        # Legacy K8sEnvBuilder (custom implementation, 1895 lines)
        "k8s_legacy": ".swe.environment.k8s:K8sEnvBuilder",
        "kr8s": ".swe.environment.kr8s:Kr8sEnvBuilder",
    },
    "runtime": {
        "swefactory": ".swe.runtime.swefactory:SWEFactoryBuiler",
        "swebench": ".swe.runtime.swebench:SWEBenchBuiler",
        "swebench_sii": ".swe.runtime.swebench_sii:SWEBenchBuiler",
        "swebench_agent": ".swe.runtime.swebench_k8s_eval:SWEBenchK8sEvalBuilder",
    },
}


def agentflow(config: dict) -> AgentFlow:
    """使用配置加载 TaskFlow

    Args:
        config (dict): 整合在 RL 框架的配置管理中（例如配置文件 `scaffold` 键的值）

    Returns:
        TaskFlow: 一个 TaskFlow 实例

    Note:
        Model 不在这里传入，而是在 preprocess(sample, model) 时传入。


    示例 Config structure:
    {
        "name": "swe.agentflow",  # name or path; must be specified
        "agent": {"name":"sweagent"},  # train 时使用的 agent
        "validate_agent": {"name":"litellm_agent"},  # validate 时使用的 agent（可选）
        "environment": {"name":"k8s"},  # environment config
        "runtime": {"name":"swebench_agent"},  # runtime config
        "thread_pool": {"max_workers": 128},  # optional thread pool config
        "python_path": ["/root/math/custom_agents"],  # optional import paths
    }
    """
    python_path = config.get("python_path")

    instances = []
    # 检查是否是 validate-only 模式（有 validate_agent 但没有 agent）
    is_validate_only = "validate_agent" in config and "agent" not in config

    # 在 validate-only 模式下，跳过加载 agent
    # 注意：顺序必须匹配 SWEAgentFlow.__init__ 的参数顺序: env, runtime, agent
    for name in ("environment", "runtime", "agent"):
        # 如果是 validate-only 模式且当前是 agent，跳过
        if name == "agent" and is_validate_only:
            continue
        try:
            subconf = config[name]
            builtin = BUILTIN_PROVIDERS[name]
            cls = import_any(subconf["name"], builtin=builtin, path=python_path)
            if cls is None:
                raise ImportError(f"`{subconf['name']}` not found")
            instance = cls(subconf)
            instances.append(instance)
        except Exception as e:
            raise ImportError(f"Fail to initialize SWE agentflow {name} builder: {e}") from e

    # 新增：加载 validate_agent（如果配置了）
    validate_agent = None
    if "validate_agent" in config:
        try:
            # 合并配置：validate_agent 可以复用 agent 的配置
            validate_agent_config = config["validate_agent"].copy()

            # 如果 validate_agent 没有配置 model，从 agent.model 复制
            if "model" not in validate_agent_config and "model" in config.get("agent", {}):
                validate_agent_config["model"] = config["agent"]["model"].copy()
            elif "model" in validate_agent_config and "model" in config.get("agent", {}):
                # 如果都有，合并配置（validate_agent 优先）
                merged_model = config["agent"]["model"].copy()
                merged_model.update(validate_agent_config["model"])
                validate_agent_config["model"] = merged_model

            # 过滤掉 GenericAPIModelConfig 不支持的字段
            # GenericAPIModelConfig 不支持这些字段（来自 SGLangModelConfig）
            if "model" in validate_agent_config:
                unsupported_fields = {"max_workers", "tool_parser"}
                for field in unsupported_fields:
                    validate_agent_config["model"].pop(field, None)

            # 如果 validate_agent 没有配置 templates，从 agent 复制
            if "templates" not in validate_agent_config and "templates" in config.get("agent", {}):
                validate_agent_config["templates"] = config["agent"]["templates"]

            # 如果 validate_agent 没有配置 tools，从 agent 复制
            if "tools" not in validate_agent_config and "tools" in config.get("agent", {}):
                validate_agent_config["tools"] = config["agent"]["tools"]

            # 如果 validate_agent 没有配置 max_requeries，从 agent 复制
            if "max_requeries" not in validate_agent_config and "max_requeries" in config.get("agent", {}):
                validate_agent_config["max_requeries"] = config["agent"]["max_requeries"]

            subconf = validate_agent_config
            builtin = BUILTIN_PROVIDERS["agent"]
            cls = import_any(subconf["name"], builtin=builtin, path=python_path)
            if cls is None:
                raise ImportError(f"`{subconf['name']}` not found")
            validate_agent = cls(subconf)
            logger.info(f"[agentflow] Loaded validate_agent: {subconf['name']}")
        except Exception as e:
            raise ImportError(f"Fail to initialize SWE agentflow validate_agent builder: {e}") from e

    # Create thread pool if configured
    thread_pool = None
    thread_pool_config = config.get("thread_pool")
    if thread_pool_config:
        max_workers = thread_pool_config.get("max_workers", 128)
        thread_pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="sweagentflow_")
        logger.info(f"[agentflow] Created thread pool with max_workers={max_workers}")

    return SWEAgentFlow(*instances, validate_agent=validate_agent, thread_pool=thread_pool)


class SWEAgentFlow(AgentFlow):
    def __init__(
        self,
        env: ContainerEnvBuilder,
        runtime: RuntimeBuilder,
        agent: AgentBuilder = None,  # 可选，validate 模式下可以为 None
        validate_agent: AgentBuilder = None,
        thread_pool: ThreadPoolExecutor = None,
    ):
        # 如果 agent 为 None，使用 validate_agent 作为主要的 agent
        self.agent = validate_agent if agent is None else agent
        self.validate_agent = validate_agent
        self.env = env
        self.runtime = runtime
        # Use provided thread pool or create a new one
        self._thread_pool = thread_pool
        self._owns_thread_pool = thread_pool is None

    def _preprocess_sync(self, sample: dict, model: Model, is_validate: bool = False) -> SWESample:
        """Synchronous wrapper for preprocess() - runs in thread pool.

        Args:
            sample: 数据集样本
            model: 模型实例
            is_validate: 是否为 validate 阶段
        """
        data = self.runtime.parse_sampledata(sample)
        meta = SWEAgentMeta(data)
        s = SWESample(meta, model)
        runtime = self.runtime.build(s)

        # 根据是否 validate 选择不同的 agent
        agent_builder = self.validate_agent if is_validate and self.validate_agent else self.agent
        agent = agent_builder.build(s)

        meta.runtime = runtime
        meta.agent = agent
        # 存下 is_validate，_reward_sync 依此决定 exit_status 过滤规则
        meta.rollout.is_validate = is_validate

        # Store weight_version (training step) from sample dict for eval to use
        if "weight_version" in sample:
            meta.weight_version = sample["weight_version"]
            logger.debug(f"[SWEAgentFlow._preprocess_sync] Set weight_version={sample['weight_version']} for sample")

        # Debug: Verify that agent was set correctly
        logger.debug(f"[SWEAgentFlow._preprocess_sync] is_validate={is_validate}, agent type: {type(meta.agent)}")
        return s

    async def preprocess(self, sample: dict, model: Model, is_validate: bool = False) -> SWESample:
        """把数据集的一条数据 (dict) 处理成 Sample 对象；可能会 raise exception

        Args:
            sample (dict): 数据集的一条数据
            model (Model): 模型实例（每个样本独立）
            is_validate (bool): 是否为 validate 阶段

        Returns:
            Sample: rollout 并 evaluate 的算例
        """
        loop = asyncio.get_running_loop()
        # Create thread pool if needed
        if self._thread_pool is None:
            self._thread_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sweagentflow_")
            self._owns_thread_pool = True
        # Run in thread pool
        return await loop.run_in_executor(self._thread_pool, self._preprocess_sync, sample, model, is_validate)

    def _generate_sync(self, sample: SWESample):
        """Synchronous wrapper for generate() - runs in thread pool."""
        m = sample.m
        logger.warning("[SWEAgentFlow._generate_sync] Running agent.run")
        logger.debug(f"[SWEAgentFlow._generate_sync] m.agent type: {type(m.agent)}, m.agent: {m.agent}")
        logger.debug(f"[SWEAgentFlow._generate_sync] m.agent.run: {getattr(m.agent, 'run', 'NO_RUN_ATTR')}")
        env = None
        try:
            # Use sync start method
            # 传递完整的 runtime_meta，包含 deployment_config
            env = self.env.start_sync(m.data.container_args, m.data.runtime_meta)
        except Exception as start_exc:
            logger.error(f"[AgentFlow._generate_sync] Failed to start environment: {start_exc}", exc_info=True)
            raise EnvCreateError(f"env start failed: {start_exc}") from start_exc

        try:
            # Use sync methods
            m.runtime.bootstrap_sync(env)
            m.agent.run(env)
            m.runtime.diff_sync(env)
        finally:
            # Cleanup - use sync version since we're in sync context
            if env is not None:
                if hasattr(env, "cleanup_sync"):
                    env.cleanup_sync()
                elif hasattr(env, "cleanup"):
                    # Fallback: try to call async cleanup (won't work but logs the call)
                    logger.warning("[AgentFlow._generate_sync] Calling async cleanup() in sync context, this may not work")
                    asyncio.create_task(env.cleanup())

        # 按 exit_status 决定是否让该样本进入 eval 阶段（与 Agentic_RL 对齐）
        # 注意：TRUNCATED 样本依然会跑 eval（仅预置 status=TRUNCATED，训练时由上层按 status 排除）。
        do_eval, status_override = _should_eval(m.rollout.exit_status, m.rollout.is_validate)
        if status_override is not None:
            sample.status = status_override
            tag = "Pre-set status" if do_eval else "Skip eval"
            sample.errors.append(f"{tag}: exit_status={m.rollout.exit_status!r}, is_validate={m.rollout.is_validate}")
            logger.info(
                f"[AgentFlow._generate_sync] {tag} (status={status_override.value}, do_eval={do_eval}, "
                f"exit_status={m.rollout.exit_status!r}, is_validate={m.rollout.is_validate})"
            )

    def _reward_sync(self, sample: SWESample):
        """Synchronous wrapper for reward() - runs in thread pool."""
        m = sample.m

        # 与 _generate_sync 对齐的 exit_status 过滤：不合格样本跳过 eval，但设置 reward=0
        # 这样样本仍然进入 databuffer，保证 GRPO batch size 一致性
        do_eval, _ = _should_eval(m.rollout.exit_status, m.rollout.is_validate)
        if not do_eval:
            logger.info(
                f"[AgentFlow._reward_sync] Skipping eval, setting reward=0 (exit_status={m.rollout.exit_status!r}, "
                f"is_validate={m.rollout.is_validate}, sample.status={sample.status.value})"
            )
            sample.reward = 0.0
            return

        # Check if runtime needs external env (e.g., run_instance_k8s_for_rl creates its own pod)
        needs_external_env = getattr(m.runtime, "needs_external_env", True)
        logger.debug(f"[AgentFlow._reward_sync] needs_external_env={needs_external_env}")

        env = None
        if needs_external_env:
            try:
                # Use sync start method
                # 传递完整的 runtime_meta，包含 deployment_config
                # 标记为 eval pod，跳过 reset（eval pod 应该保持镜像中的原始状态）
                runtime_meta = m.data.runtime_meta
                logger.debug(f"[AgentFlow._reward_sync] runtime_meta type: {type(runtime_meta)}")
                # runtime_meta 可能是 SBSample 对象或字典，都需要添加 _is_eval_pod 标志
                if isinstance(runtime_meta, dict):
                    runtime_meta = {**runtime_meta, "_is_eval_pod": True}
                    logger.debug("[AgentFlow._reward_sync] Set _is_eval_pod=True in dict")
                else:
                    # runtime_meta 是 SBSample 对象（dataclass），使用 getattr 添加标志
                    # 由于 SBSample 是 frozen dataclass，我们不能直接修改属性
                    # 所以在对象上设置一个非正式属性（绕过 frozen 检查）
                    object.__setattr__(runtime_meta, "_is_eval_pod", True)
                    logger.debug("[AgentFlow._reward_sync] Set _is_eval_pod=True on SBSample object")
                env = self.env.start_sync(m.data.container_args, runtime_meta)
            except Exception as start_exc:
                logger.error(f"[AgentFlow._reward_sync] Failed to start environment: {start_exc}", exc_info=True)
                raise EnvCreateError(f"env start failed: {start_exc}") from start_exc
        else:
            logger.info("[AgentFlow._reward_sync] Skipping external env creation (runtime handles its own pod)")

        try:
            # Use sync methods
            if env is not None:
                m.runtime.bootstrap_sync(env)
            m.runtime.eval_sync(env)
        finally:
            # Cleanup - use sync version since we're in sync context
            if env is not None:
                if hasattr(env, "cleanup_sync"):
                    env.cleanup_sync()
                elif hasattr(env, "cleanup"):
                    # Fallback: try to call async cleanup (won't work but logs the call)
                    logger.warning("[AgentFlow._reward_sync] Calling async cleanup() in sync context, this may not work")
                    asyncio.create_task(env.cleanup())

    async def generate(self, sample: SWESample):
        """运行 scaffold rollout / math solution generation

        Args:
            sample (Sample): 本 TaskFlow 兼容的算例
        """
        loop = asyncio.get_running_loop()
        # Create thread pool if needed
        if self._thread_pool is None:
            self._thread_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sweagentflow_")
            self._owns_thread_pool = True
        # Run in thread pool
        await loop.run_in_executor(self._thread_pool, self._generate_sync, sample)

    async def reward(self, sample: SWESample):
        """运行 evaluate / verification

        Args:
            sample (Sample): 本 TaskFlow 兼容的算例
        """
        loop = asyncio.get_running_loop()
        # Use same thread pool
        await loop.run_in_executor(self._thread_pool, self._reward_sync, sample)

    def close(self):
        """清理资源，关闭线程池（如果由我们创建）"""
        if self._owns_thread_pool and self._thread_pool is not None:
            logger.info("[SWEAgentFlow] Shutting down thread pool")
            self._thread_pool.shutdown(wait=True)
            self._thread_pool = None

    def __del__(self):
        """析构时确保线程池被关闭"""
        if self._owns_thread_pool and self._thread_pool is not None:
            import contextlib

            with contextlib.suppress(Exception):
                self._thread_pool.shutdown(wait=False)
