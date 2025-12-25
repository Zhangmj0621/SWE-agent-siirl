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

from siirl.environment.base import BasEnvironment
from siirl.environment.tool_env.utils.schemas import OpenAIFunctionToolSchema

class ToolEnv(BasEnvironment):
    def __init__(self, config, tool_schema: OpenAIFunctionToolSchema) -> None:
        super().__init__()
        self.config = config
        self.tool_schema = tool_schema or self.get_openai_tool_schema()
        assert self.tool_schema is not None, "Tool schema is not set!"
        self.name = self.tool_schema.function.name
        self._instance_dict ={}
        print(json.dumps(self.tool_schema.model_dump(exclude_unset=True, exclude_none=True), indent=2))
    
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema
    
    async def release(self, instance_id:str):
        """Release the tool instance.

        Args:
            instance_id: The instance id of the tool.
        """
        pass