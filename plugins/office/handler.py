"""Office plugin event handlers.

Each ``on_<event>`` function is bridged to the runtime event stream by the
plugin loader (see ``events`` in plugin.json). They only update the in-memory
tracker, so they are cheap and never block the agent loop.
"""

try:  # loaded by the plugin manager as a package submodule
    from .state import tracker
except ImportError:  # pragma: no cover - direct import fallback
    from plugins.office.state import tracker


def _make(event_name):
    def _handler(event, sdk=None):
        try:
            tracker.handle(event_name, event or {})
        except Exception:  # never let the view break an agent turn
            pass
    _handler.__name__ = f'on_{event_name}'
    return _handler


on_turn_begin = _make('turn_begin')
on_processing_started = _make('processing_started')
on_agent_busy_changed = _make('agent_busy_changed')
on_llm_thinking = _make('llm_thinking')
on_tool_call_started = _make('tool_call_started')
on_tool_executed = _make('tool_executed')
on_approval_required = _make('approval_required')
on_approval_resolved = _make('approval_resolved')
on_final_answer = _make('final_answer')
on_turn_complete = _make('turn_complete')
