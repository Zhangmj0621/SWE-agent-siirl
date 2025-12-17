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

# -*- coding: utf-8 -*-

""" 
This module defines an abstract base class for a Vision-Language-Action (VLA) environment
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
from pydantic import BaseModel

class BasEnvironment(ABC):
    """
    Abstract Base Class for a Vision-Language-Action (VLA) environment.
    It defines the standard asynchronous interface for resetting the environment
    and stepping through it.
    """
    def __init__(self) -> None:
        super().__init__()
    @abstractmethod
    async def reset(self) -> Dict[str, Any]:
        """
        Resets the environment to an initial state.

        Returns:
            Dict[str, Any]: The initial multi-modal observation,
                            e.g., {"image": np.array, "text": "task prompt"}.
        """
        pass

    @abstractmethod
    async def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Runs one timestep of the environment's dynamics.

        Args:
            action (Dict[str, Any]): A dictionary containing the action to be executed.
                                     For example, {"continuous_action": np.array([...])}.

        Returns:
            Tuple[Dict, float, bool, bool, Dict]: A tuple containing:
                - observation (Dict): The next observation.
                - reward (float): The reward received.
                - terminated (bool): Whether the episode has ended.
                - info (Dict): Auxiliary diagnostic information.
        """
        pass
    

class EnvResponse(BaseModel):
    """The response from a tool execution."""

    text: str | None = None
    image: list[Any] | None = None
    video: list[Any] | None = None
    rewards: float | None = None
    complete: bool = False
    metrics: dict | None = None
    
    @classmethod
    def initialize_request(cls, values):
        if "image" in values and not isinstance(values["image"], list):
            raise ValueError(
                f"Image must be a list, but got {type(values['image'])}. Please check the tool.execute(). "
                f"For single images, wrap in a list: [image]. "
                f"Example: {{'image': [img1]}} or {{'image': [img1, img2, ...]}}."
            )
        if "video" in values and not isinstance(values["video"], list):
            raise ValueError(
                f"Video must be a list, but got {type(values['video'])}. Please check the tool.execute(). "
                f"For single videos, wrap in a list: [video]. "
                f"Example: {{'video': [video1]}} or {{'video': [video1, video2, ...]}}."
            )

        return values

    def is_empty(self) -> bool:
        return not self.text and not self.image and not self.video

    def is_text_only(self) -> bool:
        return self.text and not self.image and not self.video