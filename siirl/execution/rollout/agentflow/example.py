import asyncio

from . import load_agentflow
from .base import DummyModel

model = DummyModel(["<answer> 2 </answer>"])
agent = load_agentflow(
    {
        "name": "agentflow.swe.example:SimpleAgentFlow",
        "reward_fn": "agentflow.swe.example:eval_reward",
    },
    model,
)
sample = agent.preprocess({})


async def run_agent(sample):
    await agent.generate(sample)
    await agent.reward(sample)


asyncio.run(run_agent(sample))
assert sample.reward == 1.0
print(sample)
