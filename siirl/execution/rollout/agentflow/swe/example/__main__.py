import asyncio
import json
from dotenv import load_dotenv
import os

load_dotenv()

from .. import agentflow
from .openai import OpenaiModel


FLOW_CONFIG = {
    "name": "swe",  # name or path; must be specified
    "agent": {
        "name": "minisweagent",
        "system_template": "You are a helpful assistant that can interact with a computer.\n\nYour response must contain exactly ONE bash code block with ONE command (or commands connected with && or ||).\nInclude a THOUGHT section before your command where you explain your reasoning process.\nFormat your response as shown in <format_example>.\n\n<format_example>\nYour reasoning and analysis here. Explain why you want to perform the action.\n\n```bash\nyour_command_here\n```\n</format_example>\n\nFailure to follow these rules will cause your response to be rejected.\nTo finish, issue the following command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`\nwithout any other command.\n",
        "instance_template": "Please solve this issue: {{task}}\n\nYou can execute bash commands and edit files to implement the necessary changes.\n\n## Recommended Workflow\n1. Analyze the codebase by finding and reading relevant files\n2. Create a script to reproduce the issue\n3. Edit the source code to resolve the issue\n4. Verify your fix works by running your script again\n5. Test edge cases to ensure your fix is robust\n\n## Important Rules\n\n1. Every response must contain exactly one action\n2. The action must be enclosed in triple backticks\n3. Directory or environment variable changes are not persistent. Every action is executed in a new subshell.\n   However, you can prefix any action with `MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...` or write/load environment variables from files\n4. To finish, issue the following command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.\n   Do not combine it with any other command.\n\n## Formatting your response\n\nHere is an example of a correct response:\n\n<example_response>\nTHOUGHT: I need to understand the structure of the repository first. Let me check what files are in the current directory to get a better understanding of the codebase.\n\n```bash\nls -la\n```\n</example_response>\n\n## Useful command examples\n\n### Create a new file:\n\n```bash\ncat <<'EOF' > newfile.py\nimport numpy as np\nhello = \"world\"\nprint(hello)\nEOF\n```\n\n### Edit files with sed:\n\n```bash\n# Replace all occurrences\nsed -i 's/old_string/new_string/g' filename.py\n\n# Replace only first occurrence\nsed -i 's/old_string/new_string/' filename.py\n\n# Replace first occurrence on line 1\nsed -i '1s/old_string/new_string/' filename.py\n\n# Replace all occurrences in lines 1-10\nsed -i '1,10s/old_string/new_string/g' filename.py\n```\n\n### View file content:\n\n```bash\n# View specific lines with numbers\nnl -ba filename.py | sed -n '10,20p'\n```\n\n### Any other command you want to run\n\n```bash\nanything\n```\n",
        "action_observation_template": "<returncode>{{output.returncode}}</returncode>\n{% if output.output | length < 10000 -%}\n<output>\n{{ output.output -}}\n</output>\n{%- else -%}\n<warning>\nThe output of your last command was too long.\nPlease try a different command that produces less output.\nIf you're looking at a file you can try use head, tail or sed to view a smaller number of lines selectively.\nIf you're using grep or find and it produced too much output, you can use a more selective search pattern.\nIf you really need to see something from the full command's output, you can redirect output to a file and then search in that file.\n</warning>\n{%- set elided_chars = output.output | length - 10000 -%}\n<output_head>\n{{ output.output[:5000] }}\n</output_head>\n<elided_chars>\n{{ elided_chars }} characters elided\n</elided_chars>\n<output_tail>\n{{ output.output[-5000:] }}\n</output_tail>\n{%- endif -%}\n",
        "format_error_template": "Please always provide EXACTLY ONE action in triple backticks, found {{actions|length}} actions.\nIf you want to end the task, please issue the following command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`\nwithout any other command.\nElse, please format your response exactly as follows:\n\n<response_example>\nHere are some thoughts about why you want to perform the action.\n\n```bash\n<action>\n```\n</response_example>\n",
    },
    "environment": {"name": "docker"},
    "runtime": {"name": "swebench_sii", "use_acr": True},
}


def main():
    model = OpenaiModel()
    agent = agentflow(FLOW_CONFIG, model)

    samples: list[dict] = []
    with open(os.environ["SWEBENCH_VERIFIED_JSONL"], "r") as f:
        for line in f:
            samples.append(json.loads(line))

    async def run_agent(sample: dict, sema: asyncio.Semaphore):
        async with sema:
            instance_id = sample["metadata"]["instance_id"]
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
