"""Basic agent class. See https://mini-swe-agent.com/latest/advanced/control_flow/ for visual explanation."""

import re
import subprocess
from dataclasses import asdict, dataclass

from jinja2 import StrictUndefined, Template

from ...base import Model, ModelResponse
from ..base import SWERolloutResult, SWESample
from ..environment import ContainerEnv, ContainerOutput
from .base import Agent, AgentBuilder


@dataclass
class AgentConfig:
    system_template: str = "You are a helpful assistant that can do anything."
    instance_template: str = (
        "Your task: {{task}}. Please reply with a single shell command in triple backticks. "
        "To finish, the first line of the output of the shell command must be 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'."
    )
    timeout_template: str = (
        "The last command <command>{{action['action']}}</command> timed out and has been killed.\n"
        "The output of the command was:\n <output>\n{{output}}\n</output>\n"
        "Please try another command and make sure to avoid those requiring interactive input."
    )
    format_error_template: str = "Please always provide EXACTLY ONE action in triple backticks."
    action_observation_template: str = "<returncode>{{returncode}}</returncode>\n<output>\n{{output}}\n</output>"
    action_regex: str = r"```bash\s*\n(.*?)\n```"
    step_limit: int = 0
    cost_limit: float = 3.0


class MiniSWEAgentBuilder(AgentBuilder):
    def __init__(self, config: dict):
        config.pop("name")
        self.config = AgentConfig(**config)

    def build(self, sample: SWESample) -> Agent:
        return MiniSWEAgent(self.config, sample)


class MiniSWEAgent(Agent):
    """MiniSWEAgent; its sets error name to rollout_result"""

    def __init__(self, config: AgentConfig, sample: SWESample):
        self.config = config
        self.sample = sample
        self.extra_template_vars = {"task": sample.m.data.problem_statement}
        self.n_calls = 0

    @classmethod
    def parse_config(cls, config: dict) -> AgentConfig:
        return AgentConfig(**config)

    async def run(self, env: ContainerEnv):
        """It sets rollout"""
        self.sample.add_message("system", self.render_template(self.config.system_template))
        self.sample.add_message("user", self.render_template(self.config.instance_template))

        while True:
            try:
                query = await self.query()
                await self.get_observation(query, env)
            except NonTerminatingException as e:
                self.sample.add_message("user", str(e))
            except TerminatingException as e:
                self.sample.add_message("user", str(e))
                match type(e).__name__:
                    case "Submitted":
                        result = SWERolloutResult.SUBMIT
                    case "LimitsExceeded":
                        result = SWERolloutResult.EXCEED
                    case _:
                        result = SWERolloutResult.FAILURE
                self.sample.m.rollout.result = result
                return

    async def query(self) -> ModelResponse:
        """Query the model and return the response."""
        if self.n_calls >= self.config.step_limit > 0:
            raise LimitsExceeded()
        input_tokens: list[int] = self.model.tokenizer.apply_chat_template(self.messages)
        # TODO: apply(a+b) != apply(a)+apply(b) ? (tokenin tokenout 问题)
        # assert input_tokens == self.sample.tokens
        response = await self.model.query(input_tokens, self.messages)
        self.n_calls += 1
        self.sample.add_message("assistant", response)
        return response

    async def get_observation(self, response: ModelResponse, env: ContainerEnv) -> ContainerOutput:
        """Execute the action and return the observation."""
        action = self.parse_action(response)
        output = await self.execute_action(action, env)
        observation = self.render_template(
            self.config.action_observation_template,
            output=output,
        )
        self.sample.add_message("user", observation)
        return output

    def render_template(self, template: str, **kwargs) -> str:
        template_vars = asdict(self.config)
        return Template(template, undefined=StrictUndefined).render(
            **kwargs, **template_vars, **self.extra_template_vars
        )

    def parse_action(self, response: ModelResponse) -> str:
        """Parse the action from the message. Returns the action cmd."""
        actions = re.findall(self.config.action_regex, response.output, re.DOTALL)
        if len(actions) == 1:
            return actions[0].strip()
        raise FormatError(self.render_template(self.config.format_error_template, actions=actions))

    async def execute_action(self, cmd: str, env: ContainerEnv) -> ContainerOutput:
        try:
            output = await env.execute(cmd, check=False)
        except (TimeoutError, subprocess.TimeoutExpired) as e:
            output = (
                e.output.decode("utf-8", errors="replace") if isinstance(e, subprocess.TimeoutExpired) else e.strerror
            )
            raise ExecutionTimeoutError(self.render_template(self.config.timeout_template, action=cmd, output=output))
        self.has_finished(output)
        return output

    def has_finished(self, output: ContainerOutput):
        """Raises Submitted exception with final output if the agent has finished its task."""
        lines = output.output.decode("utf-8").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() in [
            "MINI_SWE_AGENT_FINAL_OUTPUT",
            "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
        ]:
            raise Submitted("".join(lines[1:]))

    @property
    def messages(self) -> list[dict]:
        return self.sample.conversations

    @property
    def model(self) -> Model:
        return self.sample.model


class NonTerminatingException(Exception):
    """Raised for conditions that can be handled by the agent."""


class FormatError(NonTerminatingException):
    """Raised when the LM's output is not in the expected format."""


class ExecutionTimeoutError(NonTerminatingException):
    """Raised when the action execution timed out."""


class TerminatingException(Exception):
    """Raised for conditions that terminate the agent."""


class Submitted(TerminatingException):
    """Raised when the LM declares that the agent has finished its task."""


class LimitsExceeded(TerminatingException):
    """Raised when the agent has reached its cost or step limit."""
