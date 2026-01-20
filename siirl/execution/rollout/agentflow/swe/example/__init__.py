import re
from dataclasses import dataclass, field

from ...base import AgentFlow, Model, ModelResponse, Sample


@dataclass
class SimpleAgentMeta:
    sample: dict
    generate_result: dict = field(default_factory=dict)


SimpleSample = Sample[SimpleAgentMeta]


class SimpleAgentFlow:
    """Example AgentFlow implementation with configurable evaluation"""

    def __init__(
        self,
        config: dict,
        model: Model,
    ):
        self.model = model

    def preprocess(self, sample: dict) -> SimpleSample:
        """Default preprocessing: wrap raw data into Sample"""
        return SimpleSample(SimpleAgentMeta(sample), self.model)

    async def generate(self, sample: SimpleSample):
        """Default generation: query model with arithmetic question"""
        message = {
            "role": "user",
            "content": "what is result of 1+1; write your answer in <answer></answer>",
        }
        input_tokens = self.model.tokenizer.apply_chat_template([message], add_generation_prompt=True)
        sample.conversations.append(message)
        sample.append_input_tokens(input_tokens)

        # Query model
        try:
            response: ModelResponse = await self.model.query(
                input_tokens=sample.tokens,
                messages=sample.conversations,
                max_tokens=512,
                timeout=30,
            )
            sample.conversations.append({"role": "assistant", "content": response.output})
            sample.append_output(response)

            # parse output
            match = re.search(r"<answer>\s*(.*?)\s*</answer>", response.output, re.DOTALL)
            if not match:
                sample.status = SimpleSample.Status.ABORTED
                return
            answer_text = match.group(1).strip()

            sample.m.generate_result = {
                "response_raw": response.raw,
                "answer": answer_text,
            }
            sample.status = SimpleSample.Status.ROLLEDOUT

        except Exception as e:
            sample.status = SimpleSample.Status.ABORTED
            sample.errors.append(f"Generate fail: {e}")

    async def reward(self, sample: SimpleSample):
        """Configurable evaluation: injected by config"""
        sample.status = SimpleSample.Status.FAILED
        sample.errors.append("No reward function configured")


# to inject
async def eval_reward(_: AgentFlow, sample: SimpleSample):
    """Example reward function: parse <answer> tags and evaluate correctness"""
    try:
        answer_text: str = sample.m.generate_result["answer"]
        parsed_answer = float(answer_text)
        correct = abs(parsed_answer - 2.0) < 1e-6
        sample.reward = 1.0 if correct else 0.0
    except ValueError:
        sample.reward = 0.0
        sample.errors.append("Reward fail: Cannot parse as number")
    sample.status = SimpleSample.Status.COMPLETED
