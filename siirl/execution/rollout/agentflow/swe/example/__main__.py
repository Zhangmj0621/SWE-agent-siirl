from .. import agentflow
from ...base import DummyModel
import asyncio


def main():
    model = DummyModel(["<answer> 2 </answer>"])
    agent = agentflow(
        {
            "name": "swe.agentflow",  # name or path; must be specified
            "agent": {
                "name": "swe.agent.minisweagent:MiniSWEAgentBuilder"
            },  # agent config
            "environment": {
                "name": "swe.environment.k8s:K8sEnvBuilder"
            },  # agent config
            "runtime": {
                "name": "swe.runtime.swefactory:SWEFactoryBuiler"
            },  # agent config
            "python_path": ["/root/math/custom_agents"],  # optional import paths
        },
        model,
    )

    sample = agent.preprocess(
        {
            "instance_id": "github/community",
            "problem_statement": "It already succeeds",
            "base_commit": "deadface",
            "image_name": "alpine:latest",
        }
    )

    async def run_agent(sample):
        await agent.generate(sample)
        await agent.reward(sample)

    asyncio.run(run_agent(sample))
    assert sample.reward == 1.0
    print(sample)


if __name__ == "__main__":
    main()
