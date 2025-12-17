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
import os
import importlib
import sys
from functools import partial
from loguru import logger
from typing import Optional, Callable

from siirl.params import SiiRLArguments

def load_custom_reward_function(config: SiiRLArguments) -> Optional[Callable]:
    """
    Dynamically loads a custom reward function from a user-specified file.

    This function reads the path and function name from the configuration,
    imports the module, and returns the specified function.

    Args:
        config: The main SiiRLArguments configuration object which contains
                the `custom_reward_function` settings.

    Returns:
        The loaded custom reward function wrapped with its configured keyword
        arguments, or None if no custom function is specified.

    Raises:
        FileNotFoundError: If the specified Python file does not exist.
        AttributeError: If the function is not found within the specified file.
        RuntimeError: If the module cannot be loaded for other reasons.
    """
    reward_fn_config = config.custom_reward_function
    file_path = reward_fn_config.path

    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Custom reward function file not found: '{file_path}'")

    # Dynamically import the module from the given file path.
    module_name = "custom_module"  # A placeholder name for the module.
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create module spec from '{file_path}'")

    module = importlib.util.module_from_spec(spec)
    # This allows the module to be discoverable by other parts of the system
    # if necessary, for instance during deserialization (unpickling).
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Failed to execute module from '{file_path}': {e}") from e

    function_name = reward_fn_config.name
    if not hasattr(module, function_name):
        raise AttributeError(f"Function '{function_name}' not found in custom reward file '{file_path}'.")

    logger.info(f"Using custom reward function '{function_name}' from '{file_path}'")
    raw_fn = getattr(module, function_name)
    reward_kwargs = dict(reward_fn_config.reward_kwargs)

    # Wrap the function to pre-fill the custom keyword arguments.
    return partial(raw_fn, **reward_kwargs)