import asyncio
import contextlib
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Coroutine

from loguru import logger

from siirl.execution.rollout.utils import EnvCreateError, SglangGenerationAborted

from ..base import AgentFlow, Model
from ..utils import import_any
from .agent import AgentBuilder
from .base import SWEAgentMeta, SWESample
from .environment import ContainerEnvBuilder
from .runtime import RuntimeBuilder


# ---------------------------------------------------------------------------
# Flow thread pool (Plan D — pod-ops isolation)
#
# Under Ray, ``RolloutWorker`` shares a single event loop with Sglang, MetricWorker,
# data coordinator RPC, etc.  If any of those co-residents stalls the loop, every
# ``await`` here — including ``asyncio.wait_for`` timers inside swerex's
# ``kubectl`` subprocess calls — stalls along with it.  That's what caused the
# "4 pods stuck for 8 min, then all timeout at once" failure we saw.
#
# To restore the isolation the old ``_generate_sync`` pool used to provide, we
# run ``generate`` / ``reward`` bodies on a dedicated worker thread in a brand
# new event loop.  The rollout main loop merely awaits the worker-thread future;
# any stall on the main loop no longer poisons the kubectl / httpx timers used
# by SWEEnv.
#
# Concurrency:
#   - ``SWE_FLOW_THREADS`` env var controls pool size; default mirrors what the
#     old ``thread_pool.max_workers`` config expected (256).
#   - The pool is lazily created on first use and shared across flow instances.
# ---------------------------------------------------------------------------

_FLOW_EXECUTOR: ThreadPoolExecutor | None = None
_FLOW_EXECUTOR_LOCK = threading.Lock()


def _get_flow_executor() -> ThreadPoolExecutor:
    """Lazy-initialise the per-process flow thread pool."""
    global _FLOW_EXECUTOR
    if _FLOW_EXECUTOR is None:
        with _FLOW_EXECUTOR_LOCK:
            if _FLOW_EXECUTOR is None:
                max_workers = int(os.getenv("SWE_FLOW_THREADS", "256"))
                _FLOW_EXECUTOR = ThreadPoolExecutor(
                    max_workers=max_workers,
                    thread_name_prefix="sweagentflow",
                )
                logger.info(f"[SWEAgentFlow] flow executor created, max_workers={max_workers}")
    return _FLOW_EXECUTOR


def _run_async_in_thread(coro_fn: Callable[..., Coroutine[Any, Any, Any]], *args: Any, **kwargs: Any) -> Any:
    """Worker-thread entrypoint: build a fresh event loop, run ``coro_fn(*args)`` on it, close.

    Constructing the coroutine inside the worker keeps it bound to the loop
    that will actually drive it — avoids the "coroutine attached to a different
    loop" / cross-loop httpx failures you'd otherwise get when SWEEnv's HTTP
    client is created on one loop and used on another.

    Stale httpx client purge: ``GlobalAsyncHTTPClient`` caches its
    ``httpx.AsyncClient`` in a ``threading.local()`` slot that is NEVER
    cleaned up. Because our executor reuses threads across rollouts and each
    rollout gets a brand-new event loop that we close here, a cached client
    from a previous call is bound to an already-closed loop; the next call
    on the same thread then raises ``RuntimeError('Event loop is closed')``
    the instant it hits SGLang (see test12 at ~20:52). So: drop the cache
    on entry, drop+``aclose()`` on exit.
    """
    # Lazy import to avoid pulling http_utils at module import time.
    from siirl.utils.net_utils.http_utils import _thread_local as _http_tl

    if hasattr(_http_tl, "client"):
        # Client bound to a previous loop (now closed). aclose() would need
        # that dead loop — just drop the reference; httpx sockets close on GC.
        del _http_tl.client

    thread_loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(thread_loop)
        return thread_loop.run_until_complete(coro_fn(*args, **kwargs))
    finally:
        if hasattr(_http_tl, "client"):
            try:
                thread_loop.run_until_complete(_http_tl.client.aclose())
            except Exception:
                pass
            del _http_tl.client
        try:
            thread_loop.close()
        except Exception:  # pragma: no cover - defensive cleanup
            pass
        asyncio.set_event_loop(None)


def _inject_resume_handle(runtime_meta: Any, handle: dict) -> Any:
    """Return a runtime_meta variant carrying ``_resume_handle=handle``.

    ``K8sEnvAdapterBuilder`` / ``E2BEnvBuilder`` read ``_resume_handle`` from
    runtime_meta and reattach (same pod / ``Sandbox.connect``) instead of
    provisioning a fresh environment. This helper preserves the shape of
    runtime_meta (dict vs dataclass) and avoids mutating the caller's object.
    """
    if runtime_meta is None:
        return {"_resume_handle": handle}
    if isinstance(runtime_meta, dict):
        return {**runtime_meta, "_resume_handle": handle}
    # Non-frozen dataclass (e.g. SBSample): attach attribute in-place.
    try:
        object.__setattr__(runtime_meta, "_resume_handle", handle)
    except Exception:
        logger.warning(
            "[SWEAgentFlow._inject_resume_handle] Could not set _resume_handle on "
            f"runtime_meta type={type(runtime_meta).__name__}; resume may fall back to fresh pod"
        )
    return runtime_meta


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
        "e2b": ".swe.environment.e2b:E2BEnvBuilder",
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
        "environment": {"name":"e2b"},  # or "k8s"; see example sweagent_config_train.yaml
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

    # The whole stack is async now; `thread_pool` config is accepted for
    # backward-compat but ignored (flow runs on the caller's event loop and
    # relies on asyncio.gather for concurrency).
    if "thread_pool" in config:
        logger.info("[agentflow] `thread_pool` config is deprecated under async flow; ignoring")

    return SWEAgentFlow(*instances, validate_agent=validate_agent)


class SWEAgentFlow(AgentFlow):
    def __init__(
        self,
        env: ContainerEnvBuilder,
        runtime: RuntimeBuilder,
        agent: AgentBuilder = None,  # 可选，validate 模式下可以为 None
        validate_agent: AgentBuilder = None,
    ):
        # 如果 agent 为 None，使用 validate_agent 作为主要的 agent
        self.agent = validate_agent if agent is None else agent
        self.validate_agent = validate_agent
        self.env = env
        self.runtime = runtime

    async def preprocess(self, sample: dict, model: Model, is_validate: bool = False) -> SWESample:
        """把数据集的一条数据 (dict) 处理成 Sample 对象；可能会 raise exception

        Args:
            sample (dict): 数据集的一条数据
            model (Model): 模型实例（每个样本独立）
            is_validate (bool): 是否为 validate 阶段

        Returns:
            Sample: rollout 并 evaluate 的算例
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
        # 存下 is_validate，reward 依此决定 exit_status 过滤规则
        meta.rollout.is_validate = is_validate

        # Store weight_version (training step) from sample dict for eval to use
        if "weight_version" in sample:
            meta.weight_version = sample["weight_version"]
            logger.debug(f"[SWEAgentFlow.preprocess] Set weight_version={sample['weight_version']} for sample")

        # Partial-rollout inbound: if AgentFlowCallable forwarded a
        # ``partial_agent_data`` from the siirl Sample into this dict,
        # record it on the rollout meta. ``_generate_async`` picks it up
        # and branches into the resume path.
        partial = sample.get("partial_agent_data") if isinstance(sample, dict) else None
        if partial:
            meta.rollout.partial_state = partial
            # 置空出口字段，防止和本轮 _generate_async 新写入的 partial 互相污染
            meta.rollout.partial_agent_data = None
            logger.info(
                f"[SWEAgentFlow.preprocess] Resuming from partial "
                f"(rid={partial.get('rid')}, assistant_turns={partial.get('assistant_turns')})"
            )

        logger.debug(f"[SWEAgentFlow.preprocess] is_validate={is_validate}, agent type: {type(meta.agent)}")
        return s

    async def generate(self, sample: SWESample):
        """Dispatch the full rollout to a worker-thread event loop.

        The actual rollout body (``_generate_async``) runs in isolation — its
        SWEEnv/httpx/kubectl timers can't be starved by Sglang or Ray activity
        on the rollout main loop.  See the module header comment for background.
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            _get_flow_executor(), _run_async_in_thread, self._generate_async, sample
        )

    async def reward(self, sample: SWESample):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            _get_flow_executor(), _run_async_in_thread, self._reward_async, sample
        )

    async def _generate_async(self, sample: SWESample):
        """rollout body — runs inside the worker thread's event loop.

        Abort / partial-rollout semantics (aligned with naive_flow):
          - If the agent's ``engine.generate`` sees ``finish_reason=abort``,
            it raises ``SglangGenerationAborted``. Because that's a
            ``BaseException`` subclass, upstream ``except Exception`` paths
            inside ``RLTokenAgent.forward_with_handling`` don't swallow it.
          - The wrapper (``RLTokenAgentWrapper.run``) catches it, snapshots
            agent + model state into ``m.agent.partial_state``, and re-raises.
          - We catch it here, pack the wrapper snapshot + env handle into
            ``m.rollout.partial_agent_data``, then ``detach`` — the remote
            environment stays alive (K8s pod / E2B sandbox) so the next worker's
            resume can attach. The outer
            ``AgentFlowCallable.__call__`` copies ``partial_agent_data``
            onto the siirl Sample and raises ``RolloutGenerationAborted``.
          - If ``get_handle`` or ``detach`` fails, we fall through to
            ``cleanup`` and invalidate ``partial_agent_data`` — a stale pod
            handle would be worse than losing the partial progress.
        """
        m = sample.m
        partial = getattr(m.rollout, "partial_state", None)
        logger.warning(
            "[SWEAgentFlow.generate] Running agent.%s", "resume" if partial else "run"
        )
        logger.debug(f"[SWEAgentFlow.generate] m.agent type: {type(m.agent)}")

        # Inject the resume handle into runtime_meta so the env builder reattaches:
        # K8sEnvAdapter → existing pod; E2BEnvBuilder → ``Sandbox.connect(sandbox_id)``.
        runtime_meta = m.data.runtime_meta
        if partial and partial.get("swe_env_handle"):
            runtime_meta = _inject_resume_handle(runtime_meta, partial["swe_env_handle"])

        env = None
        try:
            env = await self.env.start(m.data.container_args, runtime_meta)
        except Exception as start_exc:
            logger.error(f"[SWEAgentFlow.generate] Failed to start environment: {start_exc}", exc_info=True)
            raise EnvCreateError(f"env start failed: {start_exc}") from start_exc

        aborted = False
        try:
            await m.runtime.bootstrap(env)  # idempotent — resume path is a no-op
            if getattr(self.env, "step_pause", False) and hasattr(env, "enable_step_pause"):
                env.enable_step_pause()
            if partial:
                await m.agent.resume(env, partial)
            else:
                await m.agent.run(env)
            if hasattr(env, "resume_sandbox"):
                await env.resume_sandbox()
            await m.runtime.diff(env)
        except SglangGenerationAborted:
            # RLTokenAgentWrapper.run/resume already did:
            #   1. self.partial_state = self._snapshot()
            #   2. self.partial_state["swe_env_handle"] = env.get_handle()
            #   3. await env.detach()  (pauses sandbox / snapshots state)
            # We just need to pack it into partial_agent_data for the cancel queue.
            wrapper_partial = getattr(m.agent, "partial_state", None) or {}
            env_handle = wrapper_partial.get("swe_env_handle")
            if env_handle is None or not wrapper_partial:
                logger.warning(
                    f"[SWEAgentFlow.generate] Abort without usable partial "
                    f"(env_handle={bool(env_handle)}, wrapper_partial={bool(wrapper_partial)}); "
                    "releasing env"
                )
                m.rollout.partial_agent_data = None
            else:
                m.rollout.partial_agent_data = {
                    **wrapper_partial,
                    "swe_runtime_bootstrapped": True,
                }
                aborted = True
            raise
        finally:
            if env is not None:
                if hasattr(env, "disable_step_pause"):
                    with contextlib.suppress(Exception):
                        env.disable_step_pause()
                if aborted:
                    # Wrapper already paused the sandbox via env.detach().
                    # If detach failed there, env may still be alive — clean up
                    # as a safety net (detach is a no-op if already closed).
                    if not getattr(m.rollout, "partial_agent_data", None):
                        with contextlib.suppress(Exception):
                            await env.cleanup()
                else:
                    try:
                        await env.cleanup()
                    except Exception as e:
                        logger.warning(f"[SWEAgentFlow.generate] cleanup failed: {e}")

        # 按 exit_status 决定是否让该样本进入 eval 阶段（与 Agentic_RL 对齐）
        # 注意：TRUNCATED 样本依然会跑 eval（仅预置 status=TRUNCATED，训练时由上层按 status 排除）。
        # Abort 路径不会到这里（上面 raise 已经离开了函数）。
        do_eval, status_override = _should_eval(m.rollout.exit_status, m.rollout.is_validate)
        if status_override is not None:
            sample.status = status_override
            tag = "Pre-set status" if do_eval else "Skip eval"
            sample.errors.append(f"{tag}: exit_status={m.rollout.exit_status!r}, is_validate={m.rollout.is_validate}")
            logger.info(
                f"[SWEAgentFlow.generate] {tag} (status={status_override.value}, do_eval={do_eval}, "
                f"exit_status={m.rollout.exit_status!r}, is_validate={m.rollout.is_validate})"
            )

    async def _reward_async(self, sample: SWESample):
        """reward body — runs inside the worker thread's event loop."""
        m = sample.m

        # 与 generate 对齐的 exit_status 过滤：不合格样本跳过 eval，但设置 reward=0
        # 这样样本仍然进入 databuffer，保证 GRPO batch size 一致性
        do_eval, _ = _should_eval(m.rollout.exit_status, m.rollout.is_validate)
        if not do_eval:
            logger.info(
                f"[SWEAgentFlow.reward] Skipping eval, setting reward=0 (exit_status={m.rollout.exit_status!r}, "
                f"is_validate={m.rollout.is_validate}, sample.status={sample.status.value})"
            )
            sample.reward = 0.0
            return

        # Check if runtime needs external env (e.g., run_instance_k8s_for_rl creates its own pod)
        needs_external_env = getattr(m.runtime, "needs_external_env", True)
        logger.debug(f"[SWEAgentFlow.reward] needs_external_env={needs_external_env}")

        env = None
        if needs_external_env:
            try:
                # 标记为 eval pod，跳过 reset（eval pod 应该保持镜像中的原始状态）
                runtime_meta = m.data.runtime_meta
                logger.debug(f"[SWEAgentFlow.reward] runtime_meta type: {type(runtime_meta)}")
                if isinstance(runtime_meta, dict):
                    runtime_meta = {**runtime_meta, "_is_eval_pod": True}
                    logger.debug("[SWEAgentFlow.reward] Set _is_eval_pod=True in dict")
                else:
                    # runtime_meta 是 SBSample 对象（dataclass，可能 frozen），绕过 frozen 检查
                    object.__setattr__(runtime_meta, "_is_eval_pod", True)
                    logger.debug("[SWEAgentFlow.reward] Set _is_eval_pod=True on SBSample object")
                env = await self.env.start(m.data.container_args, runtime_meta)
            except Exception as start_exc:
                logger.error(f"[SWEAgentFlow.reward] Failed to start environment: {start_exc}", exc_info=True)
                raise EnvCreateError(f"env start failed: {start_exc}") from start_exc
        else:
            logger.info("[SWEAgentFlow.reward] Skipping external env creation (runtime handles its own pod)")

        try:
            if env is not None:
                await m.runtime.bootstrap(env)
            await m.runtime.eval(env)
        finally:
            if env is not None:
                try:
                    await env.cleanup()
                except Exception as e:
                    logger.warning(f"[SWEAgentFlow.reward] cleanup failed: {e}")
