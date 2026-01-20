# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import datetime
import json
import os
import traceback

import aiohttp

# Global variable to store the path for failed submissions
_failed_submissions_path = os.path.expanduser("~")


def set_failed_submissions_path(path: str):
    """
    Set the path where failed submissions will be saved.

    Args:
        path: The directory path to save failed submissions
    """
    global _failed_submissions_path
    _failed_submissions_path = os.path.expanduser(path)
    # Create directory if it doesn't exist
    os.makedirs(_failed_submissions_path, exist_ok=True)
    print(f"Failed submissions will be saved to: {_failed_submissions_path}")


def get_failed_submissions_path() -> str:
    """
    Get the current path where failed submissions will be saved.

    Returns:
        The current path for saving failed submissions
    """
    return _failed_submissions_path


async def call_one_submission(
    url: str,
    submission: dict,
    session: aiohttp.ClientSession,
    max_retries: int = 4,
    backoff_factor: float = 0.5,
):
    attempt_count = 0
    result = None
    while attempt_count < max_retries:
        attempt_count += 1
        try:
            async with session.post(url, json=submission) as response:
                response.raise_for_status()
                response_json = await response.json()
                result = response_json
                return result
        except aiohttp.ClientResponseError as e:
            print(f"Attempt {attempt_count}: Server responded with {e.status}")
            if attempt_count < max_retries:
                await asyncio.sleep(backoff_factor * (2 ** (attempt_count - 1)))
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            print(f"Attempt {attempt_count}: Caught {type(e).__name__}: {repr(e)}")
            if attempt_count < max_retries:
                await asyncio.sleep(backoff_factor * (2 ** (attempt_count - 1)))
        except Exception as e:
            print(f"call_one_submission Error: {e}")
            if attempt_count < max_retries:
                await asyncio.sleep(backoff_factor * (2 ** (attempt_count - 1)))
            traceback.print_exc()
    return None


async def call_long_batch(
    url: str,
    submissions: list[dict],
    session: aiohttp.ClientSession,
    max_retries: int = 4,
    backoff_factor: float = 0.5,
):

    sub_num = len(submissions)
    results = [None] * sub_num
    duration_time = [0] * sub_num
    sub_ids = list(range(sub_num))
    attempt_count = 0
    while submissions and attempt_count < max_retries:
        attempt_count += 1
        try:
            data = {"type": "batch", "submissions": submissions}
            queue_timeouts = []
            async with session.post(url, json=data) as response:
                response.raise_for_status()
                response_json = await response.json()
                for sub_id, result, duration in zip(sub_ids, response_json["results"], response_json["duration_time"], strict=False):
                    results[sub_id] = result
                    duration_time[sub_id] = duration
            print(
                f"Batch size: {sub_num}, duration time: "
                f'min {f"{min(duration_time):.10f}" if duration_time else "N/A"}, '
                f'max {f"{max(duration_time):.10f}" if duration_time else "N/A"}, '
                f'avg {f"{sum(duration_time)/len(duration_time):.10f}" if duration_time else "N/A"}'
            )
            submissions = [sub for _, sub in queue_timeouts]
            sub_ids = [sub_id for sub_id, _ in queue_timeouts]
        except aiohttp.ClientResponseError as e:
            print(f"Attempt {attempt_count}: Server responded with {e.status}")
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            print(f"Attempt {attempt_count}: Caught {type(e).__name__}: {repr(e)}")
        except Exception as e:
            print(f"run_tool_calls_on_server_async Error: {e}")
            traceback.print_exc()
        finally:
            await asyncio.sleep(backoff_factor * (2 ** (attempt_count - 1)))

    # Save failed submissions to file if any remain after max retries
    if submissions:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        failed_file = os.path.join(_failed_submissions_path, f"failed_submissions_{timestamp}.json")

        failed_data = {
            "timestamp": timestamp,
            "url": url,
            "max_retries": max_retries,
            "failed_submissions": [],
        }

        for sub_id, submission in zip(sub_ids, submissions, strict=False):
            failed_data["failed_submissions"].append({"original_index": sub_id, "submission": submission})

        try:
            with open(failed_file, "w", encoding="utf-8") as f:
                json.dump(failed_data, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(submissions)} failed submissions to: {failed_file}")
        except Exception as e:
            print(f"Failed to save failed submissions: {e}")

    return results


async def run_single_tool_call_on_server_async(
    tool_call: dict,
    max_retries: int = 4,
    backoff_factor: float = 0.5,
    url: str = "http://localhost:8089",
):
    """
    Put the single tool-call task from distributed queue to centralized buffer on master node.

    Args:
        tool_call (Dict): The tool call
        max_retries (int,   optional): The maximum number of retry attempts. Defaults to 4.
        backoff_factor (float, optional): The backoff factor for retries. Defaults to 0.5.
        url (str, optional): The URL of the centralized buffer. Defaults to "http://localhost:8089".
    """

    submission = None
    if tool_call["name"] == "sandbox_fusion":
        from .sandbox_fusion_utils import generate_tool_call_code, generate_tool_call_input

        submission = {
            "name": tool_call["name"],
            "solution": generate_tool_call_code(tool_call),
            "input": generate_tool_call_input(tool_call),
            "start_time": tool_call["start_time"],
            "code_type": tool_call.get("code_type", "python"),
        }
    elif tool_call["name"] == "search":
        submission = {
            "name": tool_call["name"],
            "query_list": tool_call["arguments"]["query_list"],
            "topk": tool_call["arguments"]["topk"],
            "start_time": tool_call["start_time"],
        }
    elif tool_call["name"] == "gsm8k":
        pass
    else:
        raise ValueError(f"Unsupported tool call type: {tool_call['name']}")

    url = f"{url}/submit_request"

    async with aiohttp.ClientSession() as session:
        result = await call_one_submission(url, submission, session, max_retries, backoff_factor)

    if result is None:
        print(f"run_single_tool_call_on_server_async failed for tool call {tool_call} after {max_retries} attempts.")
        return {}

    # only need to contain stdout and stderr here
    # align with verl/tools/utils/sandbox_fusion_tools.py
    metadata = result["metadata"]

    # we should always expect this since we don't have correct answer
    # sandbox_fusion
    if metadata["result_type"] == "sandbox_fusion":
        if metadata["run_status"] == "Finished":
            actual_output = metadata["stdout"] + metadata["stderr"]
            # print(
            #    f"actual_output from sandbox fusion: {actual_output}"
            # )
            result = actual_output
        else:
            result = "no stdout here"
    elif metadata["result_type"] == "search":
        result = metadata
    else:
        pass

    return result


async def run_tool_calls_on_server_async(
    tool_calls: list,
    session: aiohttp.ClientSession,
    max_retries: int = 4,
    backoff_factor: float = 0.5,
    host_addr: str = "localhost",
    host_port: str = "8088",
):
    """
    Put the batch of tool-call tasks from distributed queue to centralized buffer on master node.

    Args:
        tool_calls (List): The list of tool calls
        session (aiohttp.ClientSession): The aiohttp session to centralized buffer
        type (Literal['sandbox_fusion', 'search', "gsm8k"], optional): The type of tool call. Defaults to 'sandbox_fusion'.
        max_retries (int, optional): The maximum number of retry attempts. Defaults to 4.
        backoff_factor (float, optional): The backoff factor for retries. Defaults to 0.5.
        host_addr (str, optional): The host address of the centralized buffer. Defaults to "localhost".
        host_port (str, optional): The host port of the centralized buffer. Defaults to "8088".
    """
    submissions = []
    for tool_call in tool_calls:
        if tool_call["name"] == "search":
            submissions.append(
                {
                    "name": tool_call["name"],
                    "query_list": tool_call["arguments"]["query_list"],
                    "topk": tool_call["arguments"]["topk"],
                    "start_time": tool_call["start_time"],
                }
            )
        elif tool_call["name"] == "sandbox_fusion":
            from .sandbox_fusion_utils import generate_tool_call_code, generate_tool_call_input

            submissions.append(
                {
                    "name": tool_call["name"],
                    "solution": generate_tool_call_code(tool_call),
                    "input": generate_tool_call_input(tool_call),
                    "start_time": tool_call["start_time"],
                    "code_type": tool_call.get("code_type", "python"),
                }
            )
        elif tool_call["name"] == "gsm8k":
            pass
        else:
            raise ValueError(f"Unsupported tool call type: {tool_call['name']}")

    url = f"http://{host_addr}:{host_port}/run/long-batch"
    results = await call_long_batch(url, submissions, session, max_retries, backoff_factor)

    if None in results:
        failed_indices = [i for i, result in enumerate(results) if result is None]
        # throw an error if any tool call failed after max retries
        if len(failed_indices) > 0:
            print(f"run_tool_calls_on_server_async failed for {len(failed_indices)} tool calls after {max_retries} attempts.")

    for i in range(len(results)):
        # only need to contain stdout and stderr here
        # align with verl/tools/utils/sandbox_fusion_tools.py
        metadata = results[i]["metadata"]

        # we should always expect this since we don't have correct answer
        # sandbox_fusion
        if metadata["result_type"] == "sandbox_fusion":
            if metadata["run_status"] == "Finished":
                actual_output = metadata["stdout"] + metadata["stderr"]
                # print(
                #    f"actual_output from sandbox fusion: {actual_output}"
                # )
                results[i] = actual_output
            else:
                results[i] = "no stdout here"
        elif metadata["result_type"] == "search":
            results[i] = metadata
        else:
            pass

    return results
