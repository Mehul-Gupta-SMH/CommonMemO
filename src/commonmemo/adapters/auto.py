"""Attach a `Memory` to every installed agent SDK in one call (#116, of #6).

    import commonmemo
    commonmemo.instrument(memory)
    commonmemo.uninstrument()

`instrument` patches each SDK's *defaults* so the adapter in this package is
used where the application did not choose a memory itself. Anything passed
explicitly always wins. It is opt-in (nothing happens on import), needs only
the SDKs that are installed, and warns and skips a hook that has moved
instead of raising. Like the adapters it stores and searches; it does not
push context into a prompt.

Hooks: Google ADK (`Runner` memory service), OpenAI Agents (`Runner.run*`
session), LangGraph (`StateGraph.compile` store), AutoGen (`AssistantAgent` memory; its
`update_context` does put findings in the prompt).

`user_id` is a string or a callable returning one, read at each use; without
it everything belongs to one anonymous user. `session_id` (OpenAI Agents
only, the SDK gives none) is likewise a string or callable; runs sharing it
share a conversation.
"""
from __future__ import annotations

import importlib
import importlib.util
import warnings
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, Union

if TYPE_CHECKING:
    from ..memory import Memory

Resolver = Union[str, Callable[[], str]]
ANONYMOUS = "anonymous"

_restore: list[tuple[Any, str, Any]] = []


def _value(resolver: Optional[Resolver], default: str) -> str:
    if resolver is None:
        return default
    return resolver() if callable(resolver) else resolver


def _patch(owner: Any, name: str, replacement: Any) -> None:
    _restore.append((owner, name, owner.__dict__[name]))
    setattr(owner, name, replacement)


def _adk(memory: "Memory", user_id: Optional[Resolver], session_id: Optional[Resolver]) -> None:
    from google.adk.runners import Runner
    from .adk import CommonMemoMemoryService

    service = CommonMemoMemoryService(memory)
    original = Runner.__init__

    def init(self: Any, *args: Any, **kwargs: Any) -> None:
        if kwargs.get("memory_service") is None:
            kwargs["memory_service"] = service
        original(self, *args, **kwargs)

    _patch(Runner, "__init__", init)


def _openai_agents(memory: "Memory", user_id: Optional[Resolver], session_id: Optional[Resolver]) -> None:
    from agents import Runner
    from .openai_agents import CommonMemoSession

    for name in ("run", "run_sync", "run_streamed"):
        original = getattr(Runner, name)

        def make(original: Callable[..., Any]) -> Callable[..., Any]:
            def call(starting_agent: Any, input: Any, *args: Any, **kwargs: Any) -> Any:
                if kwargs.get("session") is None and isinstance(input, (str, list)):
                    kwargs["session"] = CommonMemoSession(
                        memory, _value(session_id, "auto"), user_id=_value(user_id, ANONYMOUS))
                return original(starting_agent, input, *args, **kwargs)
            return call

        _patch(Runner, name, staticmethod(make(original)))


def _langgraph(memory: "Memory", user_id: Optional[Resolver], session_id: Optional[Resolver]) -> None:
    from langgraph.graph import StateGraph
    from .langgraph import CommonMemoStore

    store = CommonMemoStore(memory, user_id=_value(user_id, ANONYMOUS))
    original = StateGraph.compile

    def compile(self: Any, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("store") is None:
            kwargs["store"] = store
        return original(self, *args, **kwargs)

    _patch(StateGraph, "compile", compile)


def _autogen(memory: "Memory", user_id: Optional[Resolver], session_id: Optional[Resolver]) -> None:
    from autogen_agentchat.agents import AssistantAgent
    from .autogen import CommonMemoMemory

    original = AssistantAgent.__init__

    def init(self: Any, *args: Any, **kwargs: Any) -> None:
        if kwargs.get("memory") is None:
            kwargs["memory"] = [CommonMemoMemory(memory, user_id=_value(user_id, ANONYMOUS), agent=kwargs.get("name"))]
        original(self, *args, **kwargs)

    _patch(AssistantAgent, "__init__", init)


def _claude_agent_sdk(memory: "Memory", user_id: Optional[Resolver], session_id: Optional[Resolver]) -> None:
    from claude_agent_sdk import ClaudeAgentOptions
    from .claude_agent_sdk import CommonMemoSessionStore

    original = ClaudeAgentOptions.__init__

    def init(self: Any, *args: Any, **kwargs: Any) -> None:
        original(self, *args, **kwargs)
        if self.session_store is None:
            self.session_store = CommonMemoSessionStore(memory, user_id=_value(user_id, ANONYMOUS))

    _patch(ClaudeAgentOptions, "__init__", init)


HOOKS: dict[str, tuple[str, Callable[..., None]]] = {
    "adk": ("google.adk", _adk),
    "openai_agents": ("agents", _openai_agents),
    "langgraph": ("langgraph", _langgraph),
    "autogen": ("autogen_agentchat", _autogen),
    "claude_agent_sdk": ("claude_agent_sdk", _claude_agent_sdk),
}


def instrument(memory: "Memory", frameworks: Optional[Sequence[str]] = None, *,
               user_id: Optional[Resolver] = None, session_id: Optional[Resolver] = None) -> list[str]:
    """Patch the defaults of the installed SDKs (or just `frameworks`); returns those hooked.

    Calling it again replaces the earlier hooks.
    """
    uninstrument()
    wanted = list(HOOKS) if frameworks is None else list(frameworks)
    unknown = [name for name in wanted if name not in HOOKS]
    if unknown:
        raise ValueError(f"unknown framework {unknown}; choose from {sorted(HOOKS)}")
    hooked = []
    for name in wanted:
        package, hook = HOOKS[name]
        if importlib.util.find_spec(package.split(".")[0]) is None:
            if frameworks is not None:
                warnings.warn(f"commonmemo: {name} is not installed; skipped", stacklevel=2)
            continue
        try:
            importlib.import_module(package)
            hook(memory, user_id, session_id)
        except Exception as exc:  # an SDK that moved a hook must not break the app
            warnings.warn(f"commonmemo: could not hook {name}: {exc!r}; skipped", stacklevel=2)
            continue
        hooked.append(name)
    return hooked


def uninstrument() -> None:
    """Put every patched default back."""
    while _restore:
        owner, name, original = _restore.pop()
        setattr(owner, name, original)
