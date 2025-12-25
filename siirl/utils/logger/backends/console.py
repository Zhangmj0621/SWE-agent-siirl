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

"""
Console Logger Backend

Provides formatted console output for training metrics.
"""

from typing import Dict, List, Any
from loguru import logger
from .base import BackendConfig


class ConsoleBackend:
    """
    Console logger backend with formatted output.
    
    Features:
        - Groups metrics by prefix (e.g., actor/xxx, critic/xxx)
        - Formats numbers appropriately (scientific notation for extreme values)
        - Optional rich table support for log_table
    
    Example output:
        Step 100 | actor: loss=0.1234, entropy=2.34 | critic: loss=0.0567, vf=0.89
    """
    
    def __init__(self, config: BackendConfig):
        """
        Initialize console backend.
        
        Args:
            config: Backend configuration (mostly unused for console)
        """
        self._name = "console"
        self._config = config
        self._console = None  # Lazy load rich.Console if available
        self._initialized = True  # Console is always ready
    
    @property
    def name(self) -> str:
        return self._name
    
    def _ensure_initialized(self) -> bool:
        """Console backend is always initialized."""
        return True
    
    def log(self, data: Dict[str, float], step: int) -> None:
        """
        Log metrics to console with grouped formatting.
        
        Args:
            data: Dictionary of metric names to values
            step: Current training step
        """
        formatted = self._format_metrics(data, step)
        logger.info(formatted)
    
    def log_text(self, tag: str, text: str, step: int) -> None:
        """
        Log text content to console.
        
        Args:
            tag: Tag for the text
            text: Text content (truncated if too long)
            step: Current training step
        """
        # Truncate long text
        display_text = text[:500] + "..." if len(text) > 500 else text
        logger.info(f"[Step {step}] {tag}:\n{display_text}")
    
    def log_table(self, tag: str, columns: List[str], data: List[List[Any]], step: int) -> None:
        """
        Log tabular data to console.
        
        Uses rich.Table if available, otherwise falls back to simple text.
        
        Args:
            tag: Tag for the table
            columns: Column names
            data: Rows of data
            step: Current training step
        """
        try:
            from rich.console import Console
            from rich.table import Table
            
            if self._console is None:
                self._console = Console()
            
            table = Table(title=f"{tag} (Step {step})")
            for col in columns:
                table.add_column(col, overflow="fold", max_width=50)
            
            # Limit rows for display
            for row in data[:10]:
                table.add_row(*[str(v)[:100] for v in row])
            
            if len(data) > 10:
                table.add_row(*["..." for _ in columns])
            
            self._console.print(table)
            
        except ImportError:
            # Fallback without rich
            logger.info(f"[Step {step}] {tag}: {len(data)} rows")
            for i, row in enumerate(data[:3]):
                logger.info(f"  Row {i}: {row}")
            if len(data) > 3:
                logger.info(f"  ... and {len(data) - 3} more rows")
    
    def finish(self) -> None:
        """Console backend doesn't need cleanup."""
        pass
    
    def _format_metrics(self, data: Dict[str, float], step: int) -> str:
        """
        Format metrics into a grouped string.
        
        Groups metrics by their prefix (part before "/") and formats values.
        
        Args:
            data: Dictionary of metric names to values
            step: Current training step
            
        Returns:
            Formatted string for logging
        """
        # Group metrics by prefix
        groups: Dict[str, List[str]] = {}
        
        for key, value in sorted(data.items()):
            # Extract group from key
            if "/" in key:
                group = key.split("/")[0]
                name = key.split("/", 1)[1]
            else:
                group = "misc"
                name = key
            
            if group not in groups:
                groups[group] = []
            
            # Format value
            if isinstance(value, float):
                if abs(value) < 0.0001 or abs(value) > 10000:
                    formatted_value = f"{value:.2e}"
                elif abs(value) < 1:
                    formatted_value = f"{value:.4f}"
                else:
                    formatted_value = f"{value:.2f}"
            elif isinstance(value, int):
                formatted_value = str(value)
            else:
                formatted_value = str(value)
            
            groups[group].append(f"{name}={formatted_value}")
        
        # Build output string
        parts = [f"Step {step}"]
        
        # Priority order for groups
        priority_groups = ["training", "actor", "critic", "response", "prompt", "perf"]
        
        # Add priority groups first
        for group in priority_groups:
            if group in groups:
                metrics = groups.pop(group)
                parts.append(f"{group}: {', '.join(metrics)}")
        
        # Add remaining groups
        for group, metrics in sorted(groups.items()):
            parts.append(f"{group}: {', '.join(metrics)}")
        
        return " | ".join(parts)

