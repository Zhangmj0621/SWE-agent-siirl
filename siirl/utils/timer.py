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
Timer utilities for measuring code execution time.

Features:
- Context manager and decorator support
- Cumulative timing with statistics (min/max/avg/count)
- Human-readable time formatting
- Global timer registry
- Compatible with codetiming.Timer API

Examples:
    # Basic usage
    with Timer("operation") as t:
        do_something()
    print(t.elapsed)  # 1.234

    # Cumulative timing
    timer = Timer("loop", cumulative=True)
    for i in range(10):
        with timer:
            do_iteration()
    print(timer.stats)  # TimerStats(count=10, total=5.0, mean=0.5, ...)

    # TimerCollection
    timers = TimerCollection()
    with timers["step1"]:
        do_step1()
    with timers["step2"]:
        do_step2()
    print(timers.to_dict())  # {"step1": 1.0, "step2": 2.0}

    # Global registry
    with Timer.get("global_op"):
        do_global_op()
    print(Timer.get("global_op").elapsed)
"""

from __future__ import annotations

import time
from contextlib import ContextDecorator
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Optional


def format_time(seconds: float) -> str:
    """Format time in human-readable format with appropriate unit."""
    if seconds < 1e-6:
        return f"{seconds * 1e9:.2f}ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.2f}µs"
    elif seconds < 1:
        return f"{seconds * 1e3:.2f}ms"
    elif seconds < 60:
        return f"{seconds:.3f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


@dataclass
class TimerStats:
    """Statistics for cumulative timer measurements."""
    count: int = 0
    total: float = 0.0
    min: float = float("inf")
    max: float = 0.0
    _values: list[float] = field(default_factory=list, repr=False)
    
    @property
    def mean(self) -> float:
        """Average time per measurement."""
        return self.total / self.count if self.count > 0 else 0.0
    
    @property
    def last(self) -> float:
        """Last recorded measurement."""
        return self._values[-1] if self._values else 0.0
    
    def record(self, value: float) -> None:
        """Record a new measurement."""
        self.count += 1
        self.total += value
        self.min = min(self.min, value)
        self.max = max(self.max, value)
        self._values.append(value)
    
    def reset(self) -> None:
        """Reset all statistics."""
        self.count = 0
        self.total = 0.0
        self.min = float("inf")
        self.max = 0.0
        self._values.clear()
    
    def __str__(self) -> str:
        if self.count == 0:
            return "TimerStats(no measurements)"
        return (
            f"TimerStats(count={self.count}, total={format_time(self.total)}, "
            f"mean={format_time(self.mean)}, min={format_time(self.min)}, max={format_time(self.max)})"
        )


class Timer(ContextDecorator):
    """
    High-precision timer for measuring code execution time.
    
    Features:
        - Context manager: `with Timer("name") as t:`
        - Decorator: `@Timer("name")`
        - Manual start/stop
        - Cumulative mode with statistics
        - Global registry for shared timers
    
    Attributes:
        name: Timer name for identification
        elapsed: Time elapsed in seconds (last measurement or running time)
        last: Alias for elapsed (codetiming compatibility)
        stats: TimerStats object (only in cumulative mode)
    """
    
    # Global timer registry
    _registry: ClassVar[dict[str, Timer]] = {}
    
    def __init__(
        self,
        name: Optional[str] = None,
        logger: Optional[Any] = None,
        log_level: str = "info",
        cumulative: bool = False,
    ):
        """
        Initialize timer.
        
        Args:
            name: Optional name for identifying this timer
            logger: Optional logger instance (must have info/debug/etc methods)
            log_level: Log level to use ("debug", "info", "warning", etc.)
            cumulative: If True, accumulate statistics across multiple measurements
        """
        self.name = name
        self.logger = logger
        self.log_level = log_level
        self.cumulative = cumulative
        
        self._start_time: Optional[float] = None
        self._elapsed: Optional[float] = None
        self._stats: Optional[TimerStats] = TimerStats() if cumulative else None
    
    @classmethod
    def get(cls, name: str, **kwargs) -> Timer:
        """
        Get or create a timer from the global registry.
        
        Args:
            name: Timer name
            **kwargs: Arguments passed to Timer() if creating new
            
        Returns:
            Timer instance (shared if already exists)
        """
        if name not in cls._registry:
            cls._registry[name] = cls(name=name, **kwargs)
        return cls._registry[name]
    
    @classmethod
    def clear_registry(cls) -> None:
        """Clear all timers from the global registry."""
        cls._registry.clear()
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self._elapsed is not None:
            return self._elapsed
        if self._start_time is not None:
            return time.perf_counter() - self._start_time
        return 0.0
    
    @property
    def last(self) -> float:
        """Alias for elapsed (codetiming compatibility)."""
        return self.elapsed
    
    @property
    def stats(self) -> Optional[TimerStats]:
        """Get statistics (only available in cumulative mode)."""
        return self._stats
    
    @property
    def formatted(self) -> str:
        """Get elapsed time in human-readable format."""
        return format_time(self.elapsed)
    
    def start(self) -> Timer:
        """Start the timer. Returns self for chaining."""
        self._start_time = time.perf_counter()
        self._elapsed = None
        return self
    
    def stop(self) -> float:
        """Stop the timer and return elapsed time."""
        if self._start_time is None:
            raise RuntimeError("Timer has not been started. Call start() first.")
        
        self._elapsed = time.perf_counter() - self._start_time
        self._start_time = None
        
        # Record to stats if cumulative
        if self._stats is not None:
            self._stats.record(self._elapsed)
        
        # Log if logger configured
        self._log()
        
        return self._elapsed
    
    def reset(self) -> Timer:
        """Reset the timer and statistics. Returns self for chaining."""
        self._start_time = None
        self._elapsed = None
        if self._stats is not None:
            self._stats.reset()
        return self
    
    def _log(self) -> None:
        """Log the elapsed time if logger is configured."""
        if self.logger is None:
            return
        
        name_part = f"{self.name}: " if self.name else ""
        msg = f"{name_part}{self.formatted}"
        
        log_func = getattr(self.logger, self.log_level, None)
        if log_func and callable(log_func):
            log_func(msg)
    
    def __enter__(self) -> Timer:
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
    
    def __repr__(self) -> str:
        parts = []
        if self.name:
            parts.append(f"name={self.name!r}")
        parts.append(f"elapsed={self.formatted}")
        if self._stats and self._stats.count > 0:
            parts.append(f"count={self._stats.count}")
        return f"Timer({', '.join(parts)})"
    
    def __str__(self) -> str:
        if self.name:
            return f"{self.name}: {self.formatted}"
        return self.formatted


class TimerCollection:
    """
    Collection of named timers for tracking multiple operations.
    
    Provides dict-like access to timers with automatic creation.
    
    Examples:
        timers = TimerCollection()
        
        # Dict-like access
        with timers["operation1"]:
            do_op1()
        
        # Method access
        with timers.time("operation2"):
            do_op2()
        
        # Get results
        print(timers["operation1"].elapsed)  # 1.234
        print(timers.to_dict())  # {"operation1": 1.234, "operation2": 0.567}
    """
    
    def __init__(self, cumulative: bool = False):
        """
        Initialize collection.
        
        Args:
            cumulative: If True, all timers in collection use cumulative mode
        """
        self._timers: dict[str, Timer] = {}
        self._cumulative = cumulative
    
    def __getitem__(self, name: str) -> Timer:
        """Get or create a timer by name."""
        if name not in self._timers:
            self._timers[name] = Timer(name=name, cumulative=self._cumulative)
        return self._timers[name]
    
    def __contains__(self, name: str) -> bool:
        """Check if timer exists."""
        return name in self._timers
    
    def __iter__(self):
        """Iterate over timer names."""
        return iter(self._timers)
    
    def __len__(self) -> int:
        """Number of timers."""
        return len(self._timers)
    
    def time(self, name: str) -> Timer:
        """Get or create a timer by name. Alias for __getitem__."""
        return self[name]
    
    def get(self, name: str, default: float = 0.0) -> float:
        """Get elapsed time for a timer, or default if not exists."""
        if name in self._timers:
            return self._timers[name].elapsed
        return default
    
    def to_dict(self) -> dict[str, float]:
        """Get all timer values as dict."""
        return {name: timer.elapsed for name, timer in self._timers.items()}
    
    def get_all(self) -> dict[str, float]:
        """Alias for to_dict() for backward compatibility."""
        return self.to_dict()
    
    def clear(self) -> None:
        """Clear all timers."""
        self._timers.clear()
    
    def reset(self) -> None:
        """Reset all timers (keep them but reset values)."""
        for timer in self._timers.values():
            timer.reset()
    
    def summary(self) -> str:
        """Get a summary string of all timers."""
        if not self._timers:
            return "TimerCollection(empty)"
        
        lines = ["TimerCollection:"]
        for name, timer in sorted(self._timers.items()):
            lines.append(f"  {name}: {timer.formatted}")
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"TimerCollection({self.to_dict()})"
    
    def __str__(self) -> str:
        return self.summary()


# Backward compatibility alias
_CollectionTimer = Timer  # No longer needed as separate class
