from abc import ABC
from typing import Any, Optional, Type, TypeVar, cast
import importlib
from pathlib import Path
from threading import Lock
import sys

MODULE_PATH = Path(__file__).parent.resolve()
PARENT_PATH = str(MODULE_PATH.parent.resolve())
MODULE_NAME = MODULE_PATH.name

# global lock for import_any to support freethread
import_lock = Lock()


def import_any(
    name: Optional[str] = None,
    builtin: Optional[dict[str, Any]] = None,
    path: Optional[list[str]] = None,
) -> Any:
    if name is None:
        return None
    if builtin is not None:
        builtin_name = builtin.get(name)
        if isinstance(builtin_name, str):
            name = builtin_name
    if ":" in name:
        with import_lock:
            module_path, class_name = name.split(":")
            oldpath = sys.path.copy()
            sys.path.append(str(PARENT_PATH))
            if path is not None:
                sys.path = path + oldpath
            try:
                module = importlib.import_module(module_path)
                agent_class = getattr(module, class_name)
            except Exception as e:
                sys.path = oldpath
                raise e
            sys.path = oldpath
        return agent_class
    raise ImportError(name=name)


T = TypeVar("T", bound=ABC)


def null_abc(base_cls: Type[T]) -> Type[T]:
    """Create a concrete subclass of an ABC where all abstract methods raise NotImplementedError."""
    if not issubclass(base_cls, ABC):
        raise TypeError(f"{base_cls} is not an abstract base class")
    abstract_methods = getattr(base_cls, "__abstractmethods__", set())
    methods = {}
    for name in abstract_methods:

        def make_stub(method_name):
            def stub(self, *args, **kwargs):
                raise NotImplementedError(
                    f"Null implementation of abstract method '{method_name}'"
                )

            return stub

        methods[name] = make_stub(name)

    class_name = f"Null{base_cls.__name__}"
    null_class = type(class_name, (base_cls,), methods)
    return cast(Type[T], null_class)
