from dataclasses import dataclass,field
from typing import List

from siirl.params.model_args import MultiturnArguments
from .tool_env.utils.tool_register import initialize_tools_from_config
from .tool_env.utils.tool_parser import ToolParser
from .base import EnvResponse

@dataclass
class EnvManager:
    env_list: List = field(default=None)
    env_name: List = field(default=None)
    tool_schemas: List = field(default=None)
    tool_parser: ToolParser = field(default=None)
    tool_parser_name: str = field(default=None)

def initialize_env(config:MultiturnArguments):
    if config.env_type == 'tool_env':
        env_list,env_name, tool_schemas = initialize_tools_from_config(config.env_path)
        return EnvManager(env_list=env_list, env_name=env_name, tool_schemas=tool_schemas)
    else:
        raise NotImplementedError(f"env_type {config.env_type} has not implement")

__all__ = ["initialize_env","EnvResponse"]