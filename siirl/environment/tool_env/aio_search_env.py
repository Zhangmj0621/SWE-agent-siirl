# Copyright 2025, Shanghai Innovation Institute. All rights reserved.
#
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
import json
import time
from typing import Any
from uuid import uuid4

import aio.Scheduler.config as scheduler_config
import aiohttp

from siirl.environment import EnvResponse
from siirl.environment.tool_env.utils.schemas import OpenAIFunctionToolSchema
from siirl.environment.tool_env.utils.tool_call import run_single_tool_call_on_server_async

from .base_tool_env import ToolEnv


class AIOSearchEnv(ToolEnv):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.tool_schema = tool_schema
        self.topk = config.get("topk", 3)

    async def create(self, instance_id: str | None = None, **kwargs) -> tuple[str, EnvResponse]:
        if instance_id is None:
            instance_id = str(uuid4())

        self._instance_dict[instance_id] = {
            "response": "",
            "reward": [],
        }
        return instance_id, EnvResponse()

    async def reset(self, instance_id: str | None = None):
        return self.create()

    async def step(self, action: dict[str, Any]) -> EnvResponse:
        query_list_from_params = action.get("query_list")
        # instance_id = action.get("instance_id")
        if not query_list_from_params or not isinstance(query_list_from_params, list):
            error_msg = "Error: 'query_list' is missing, empty, or not a list in parameters."
            print(f"[SearchTool] {error_msg} Received parameters: {action}")
            return EnvResponse(text=json.dumps({"result": error_msg}))

        # add tool_call to queue
        tool_call = {
            "name": "search",
            "arguments": {"query_list": query_list_from_params, "topk": self.topk},
            "start_time": time.time(),
        }

        master_url = f"http://{scheduler_config.PROXY_HOST_IP}:{scheduler_config.PROXY_HOST_PORT}/get_server"
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(master_url, params={"tool_name": "search"}, timeout=30) as resp:
                    if resp.status == 200:
                        server_info = await resp.json()

                        # send tool_call to server
                        metadata = await run_single_tool_call_on_server_async(
                            tool_call=tool_call,
                            url=server_info["server"],
                        )

                        # need to inform master node that this server has completed a task
                        master_url = f"http://{scheduler_config.PROXY_HOST_IP}:{scheduler_config.PROXY_HOST_PORT}/complete_task"

                        try:
                            async with session.post(
                                master_url,
                                json={
                                    "url": server_info["server"],
                                    "tool_name": "search",
                                },
                                timeout=30,
                            ) as complete_resp:
                                if complete_resp.status != 200:
                                    error_msg = (
                                        f"Error: Failed to inform master node of task completion. " f"Status code: {complete_resp.status}"
                                    )
                                    print(f"[AIOSearchTool] {error_msg}")
                                    # Decide if you want to return an error here or just log it
                        except Exception as e:
                            error_msg = f"Error: Exception occurred while informing master node of task completion: {e}"
                            print(f"[AIOSearchTool] {error_msg}")
                            raise e
                            # Decide if you want to return an error here or just log it

                    else:
                        error_msg = f"Error: Failed to get server from master node. Status code: {resp.status}"
                        print(f"[AIOSearchTool] {error_msg}")
                        return EnvResponse(text=json.dumps({"result": error_msg}))
            except Exception as e:
                error_msg = f"Error: Exception occurred while getting server from master node: {e}"
                print(f"[AIOSearchTool] {error_msg}")
                # since run_server may timeout, we still need to complete task to proxy
                async with session.post(
                    master_url,
                    json={"url": server_info["server"], "tool_name": "search"},
                    timeout=30,
                ) as complete_resp:
                    if complete_resp.status != 200:
                        error_msg = f"Error: Failed to inform master node of task completion. Status code: {complete_resp.status}"
                        print(f"[AIOSearchTool] {error_msg}")
                return EnvResponse(text=json.dumps({"result": error_msg}))

        # Store results in instance dictionary
        # self._instance_dict[instance_id]["reward"].append(metadata.get("result_text", "").strip())

        # Convert metadata to metrics
        metrics = {
            "query_count": metadata.get("query_count", 0),
            "status": metadata.get("status", "unknown"),
            "total_results": metadata.get("total_results", 0),
            "api_request_error": metadata.get("api_request_error"),
        }

        return EnvResponse(text=metadata.get("result_text", ""), metrics=metrics)
