import asyncio
import json
from dotenv import load_dotenv
import os
import yaml
import pathlib
import loguru

__DIR__ = pathlib.Path(__file__).parent.resolve()

load_dotenv()

from .. import agentflow
from .openai import OpenaiModel


def main():
    model = OpenaiModel()
    with open(__DIR__ / "config.yaml", "r") as f:
        config = yaml.safe_load(f)

    agent = agentflow(config, model)

    samples: list[dict] = []
    with open(os.environ["SWEBENCH_VERIFIED_JSONL"], "r") as f:
        for line in f:
            samples.append(json.loads(line)["metadata"])
    samples = samples[:1]

    async def run_agent(sample: dict, sema: asyncio.Semaphore):
        async with sema:
            instance_id = sample["instance_id"]
            try:
                s = agent.preprocess(sample)
                print("Generating instance", instance_id)
                await agent.generate(s)
                print("Rewarding instance", instance_id)
                await agent.reward(s)
                reward = s.reward
                return reward
            except Exception:
                print("Instance", instance_id, "fail")
                raise

    # Run all samples concurrently and collect rewards
    async def run_all():
        sema = asyncio.Semaphore(32)
        tasks = [run_agent(sample, sema) for sample in samples]
        rewards = await asyncio.gather(*tasks)
        reward_1 = sum(1 for r in rewards if r is not None and r >= 0.5)
        reward_0 = sum(1 for r in rewards if r is not None and r < 0.5)
        failed = sum(1 for r in rewards if r is None)
        avg_reward = (
            sum(r for r in rewards if r is not None) / len(rewards) if rewards else 0
        )
        print(
            f"Reward=1: {reward_1}, Reward=0: {reward_0}, Failed: {failed}, Avg reward: {avg_reward:.4f}"
        )

    asyncio.run(run_all())


if __name__ == "__main__":
    main()
