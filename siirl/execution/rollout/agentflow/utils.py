import importlib
from abc import ABC
from typing import Any, TypeVar, cast


def import_any(
    name: str | None = None,
    builtin: dict[str, Any] | None = None,
    path: list[str] | None = None,
) -> Any:
    if name is None:
        return None
    if builtin is not None:
        builtin_name = builtin.get(name)
        if isinstance(builtin_name, str):
            name = builtin_name
    if ":" in name:
        module_path, class_name = name.split(":")
        module = importlib.import_module(module_path, "siirl.execution.rollout.agentflow")
        agent_class = getattr(module, class_name)
        return agent_class
    raise ImportError(name=name)


T = TypeVar("T", bound=ABC)


def null_abc(base_cls: type[T]) -> type[T]:
    """Create a concrete subclass of an ABC where all abstract methods raise NotImplementedError."""
    if not issubclass(base_cls, ABC):
        raise TypeError(f"{base_cls} is not an abstract base class")
    abstract_methods = getattr(base_cls, "__abstractmethods__", set())
    methods = {}
    for name in abstract_methods:

        def make_stub(method_name):
            def stub(self, *args, **kwargs):
                raise NotImplementedError(f"Null implementation of abstract method '{method_name}'")

            return stub

        methods[name] = make_stub(name)

    class_name = f"Null{base_cls.__name__}"
    null_class = type(class_name, (base_cls,), methods)
    return cast(type[T], null_class)
