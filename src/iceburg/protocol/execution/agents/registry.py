from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, Protocol, Union


# Protocol for class-based agents (back-compat)
class Agent(Protocol):
    def run(self, payload: dict, **kwargs) -> Any: ...


# Agents may be sync or async callables
AgentRunner = Union[Callable[..., Any], Callable[..., Awaitable[Any]]]

# Two registries for back-compat
_RUNNERS: Dict[str, AgentRunner] = {}
_FACTORIES: Dict[str, Callable[[], Agent]] = {}


def register_agent(name: str) -> Callable[[AgentRunner], AgentRunner]:
    """Decorator to register a function-based agent runner by name."""

    def decorator(func: AgentRunner) -> AgentRunner:
        _RUNNERS[name] = func
        return func

    return decorator


def register(name: str) -> Callable[[Callable[[], Agent]], Callable[[], Agent]]:
    """Decorator to register a class-based agent factory (legacy style)."""

    def decorator(factory: Callable[[], Agent]) -> Callable[[], Agent]:
        _FACTORIES[name] = factory
        return factory

    return decorator


def get_agent_runner(name: str) -> AgentRunner | None:
    """Returns a runner for the agent name, adapting legacy factories when needed."""
    if name in _RUNNERS:
        return _RUNNERS[name]
    if name in _FACTORIES:
        def _runner_from_factory(*, cfg=None, **kwargs):
            agent = _FACTORIES[name]()
            return agent.run(kwargs, cfg=cfg)
        return _runner_from_factory
    return None
