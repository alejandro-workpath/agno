"""Unit tests for DiscoverableTools."""

import json

import pytest

from agno.tools.discoverable import DiscoverableTools
from agno.tools.function import Function
from agno.tools.toolkit import Toolkit


def search_contacts(query: str) -> str:
    """Search contacts by name or email."""
    return f"searched: {query}"


def send_email(to: str, subject: str, body: str) -> str:
    """Email a recipient with subject and body."""
    return "sent"


def list_calendar_events() -> str:
    """List upcoming calendar events."""
    return "events"


def fetch_weather(city: str) -> str:
    """Fetch current weather for a city."""
    return "sunny"


@pytest.fixture
def dt():
    return DiscoverableTools(
        tools=[search_contacts, send_email, list_calendar_events, fetch_weather],
        max_results=3,
    )


def test_registry_built_from_callables(dt):
    assert set(dt._registry.keys()) == {
        "search_contacts",
        "send_email",
        "list_calendar_events",
        "fetch_weather",
    }


def test_registry_built_from_function_objects():
    func = Function(name="custom", description="Custom tool.", entrypoint=lambda: "ok")
    dt = DiscoverableTools(tools=[func])
    assert "custom" in dt._registry


def test_registry_built_from_toolkit():
    class MyKit(Toolkit):
        def __init__(self):
            super().__init__(name="mykit", tools=[search_contacts, send_email])

    dt = DiscoverableTools(tools=[MyKit()])
    assert "search_contacts" in dt._registry
    assert "send_email" in dt._registry


def test_toolkit_registers_search_meta_only(dt):
    # DiscoverableTools is a Toolkit — inherits get_functions() from parent.
    # It should register exactly the search_tools meta-Function.
    assert list(dt.functions.keys()) == ["search_tools"]
    assert dt.functions["search_tools"].entrypoint == dt._search


def test_bind_resets_active_names(dt):
    dt._active_names.add("send_email")
    dt.bind(tools_list=[])
    assert dt._active_names == set()


def test_toolkit_instructions_include_count(dt):
    # Toolkit's `instructions` (auto-injected via add_instructions=True)
    # replaces the old get_system_prompt_snippet method.
    assert dt.instructions is not None
    assert "4 additional tools" in dt.instructions
    assert "search_tools" in dt.instructions
    assert dt.add_instructions is True


def test_toolkit_instructions_empty_when_no_registry():
    dt = DiscoverableTools(tools=[])
    assert dt.instructions == ""


def test_search_returns_word_matches(dt):
    fake_list: list = []
    dt.bind(tools_list=fake_list)
    result = json.loads(dt._search("email send"))
    names = [t["name"] for t in result["discovered_tools"]]
    assert "send_email" in names


def test_search_appends_matches_to_tools_list(dt):
    fake_list: list = []
    dt.bind(tools_list=fake_list)
    dt._search("calendar")
    assert any(f.name == "list_calendar_events" for f in fake_list)


def test_search_respects_max_results(dt):
    fake_list: list = []
    dt.bind(tools_list=fake_list)
    # broad query that hits multiple tools
    result = json.loads(dt._search("email send calendar weather"))
    # max_results=3, registry has 4 — must cap at 3
    assert len(result["discovered_tools"]) <= 3
    assert len(fake_list) <= 3


def test_search_skips_already_active(dt):
    fake_list: list = []
    dt.bind(tools_list=fake_list)
    dt._search("email")
    before = set(dt._active_names)
    dt._search("email")
    assert dt._active_names == before  # no new activations


def test_search_returns_empty_when_no_match(dt):
    fake_list: list = []
    dt.bind(tools_list=fake_list)
    result = json.loads(dt._search("xyzzy_no_match_token"))
    assert result["discovered_tools"] == []
    assert fake_list == []


def test_search_remaining_count(dt):
    fake_list: list = []
    dt.bind(tools_list=fake_list)
    result = json.loads(dt._search("email"))
    assert result["remaining"] == len(dt._registry) - len(dt._active_names)


def test_inject_warns_when_unbound(dt, caplog):
    # No bind() called
    result = json.loads(dt._search("email"))
    # Should not crash; tools just not appended anywhere
    assert "discovered_tools" in result


def test_substring_fallback_match(dt):
    fake_list: list = []
    dt.bind(tools_list=fake_list)
    # Query token "weath" is substring of "weather" but not exact word match
    result = json.loads(dt._search("weath"))
    names = [t["name"] for t in result["discovered_tools"]]
    assert "fetch_weather" in names


def test_async_mode_flag_switches_registry():
    """Toolkits with async-only Functions should resolve through async registry."""

    async def async_only_fn() -> str:
        """Async-only capability."""
        return "done"

    class AsyncKit(Toolkit):
        def __init__(self):
            super().__init__(name="asynckit", tools=[async_only_fn])

    dt = DiscoverableTools(tools=[AsyncKit()])
    # Sync registry has the entrypoint via Toolkit.functions (async detected and routed)
    # Async registry also has it
    assert "async_only_fn" in dt._async_registry
    fake_list: list = []
    dt.bind(tools_list=fake_list, async_mode=True)
    dt._search("async")
    assert any(f.name == "async_only_fn" for f in fake_list)


def test_approval_sentinel_preserved_on_callable():
    """Raw callables with @approval(type='required') must carry the flag into the registry."""

    def sensitive_action(target: str) -> str:
        """Delete something sensitive."""
        return f"deleted {target}"

    sensitive_action._agno_approval_type = "required"  # type: ignore[attr-defined]

    dt = DiscoverableTools(tools=[sensitive_action])
    func = dt._sync_registry["sensitive_action"]
    assert func.approval_type == "required"
    assert func.requires_confirmation is True


def test_audit_approval_without_hitl_flag_raises():
    """Mirror parse_tools invariant: @approval(type='audit') needs at least one HITL flag."""

    def audited_action(target: str) -> str:
        """Perform an audited action."""
        return f"logged {target}"

    audited_action._agno_approval_type = "audit"  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="audit.*HITL"):
        DiscoverableTools(tools=[audited_action])


def test_function_object_input_registers_in_both_registries():
    """A plain Function passed in should appear in sync + async registries."""
    func = Function(name="plain", description="Plain function.", entrypoint=lambda: "ok")
    dt = DiscoverableTools(tools=[func])
    assert "plain" in dt._sync_registry
    assert "plain" in dt._async_registry


def test_bind_resets_active_names_across_runs(dt):
    """A second run must not inherit activations from the first."""
    fake_list_1: list = []
    dt.bind(tools_list=fake_list_1)
    dt._search("email")
    assert len(dt._active_names) > 0
    # Simulate second run
    fake_list_2: list = []
    dt.bind(tools_list=fake_list_2)
    assert dt._active_names == set()
    dt._search("email")
    assert len(fake_list_2) > 0


def test_discovered_function_gets_run_context_and_media(dt):
    """_inject must propagate run_context and media refs to discovered Function."""
    fake_list: list = []
    dummy_ctx = object()
    dummy_images = ["img1"]
    dt.bind(
        tools_list=fake_list,
        run_context=dummy_ctx,  # type: ignore[arg-type]
        images=dummy_images,  # type: ignore[arg-type]
    )
    dt._search("email")
    injected = fake_list[0]
    assert injected._run_context is dummy_ctx
    assert injected._images is dummy_images


def test_agent_accepts_discoverable_tools_inside_tools_list():
    """DX check: DiscoverableTools slots into the existing tools= param (Toolkit pattern)."""
    from agno.agent import Agent

    def upfront_tool() -> str:
        """Always visible."""
        return "upfront"

    def deferred_tool() -> str:
        """Discoverable."""
        return "deferred"

    discoverable = DiscoverableTools(tools=[deferred_tool])
    agent = Agent(tools=[upfront_tool, discoverable])

    # Both the upfront callable and the DiscoverableTools toolkit survive as tools
    assert agent.tools is not None
    assert any(t is discoverable for t in agent.tools)
    # Registry still holds deferred tool; it is NOT in agent.tools as a top-level entry
    assert "deferred_tool" in discoverable._sync_registry
    assert not any(callable(t) and getattr(t, "__name__", None) == "deferred_tool" for t in agent.tools)


def test_registry_exposes_media_needs_for_host_detection():
    """Host (Agent/Team) must be able to introspect discoverable pool for media params.

    Regression: codex review flagged that needs_media was computed only from upfront
    tools. If the only media-using tool lives in the discoverable pool, the host must
    still collect joint media at run start so the discovered Function sees it.
    """
    from inspect import signature

    def image_analyzer(images: list) -> str:
        """Analyze images."""
        return "analyzed"

    dt = DiscoverableTools(tools=[image_analyzer])
    has_media_tool = any(
        func.entrypoint is not None
        and any(p in signature(func.entrypoint).parameters for p in ("images", "videos", "audios", "files"))
        for func in dt._sync_registry.values()
    )
    assert has_media_tool is True
