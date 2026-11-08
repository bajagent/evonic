"""Live activity tracker for the Office view.

Turns raw agent-runtime events into one compact *activity* per agent and
keeps a short, sequenced journal the browser can tail over SSE.

An activity looks like::

    {"agent_id": "super", "state": "terminal", "tool_name": "bash",
     "detail": "npm test", "target_agent_id": "", "ts": 1731050000000,
     "seq": 42}

States (the page animates each one differently):

    idle       nothing running — the agent wanders to the lounge and naps
    thinking   the model is reasoning — typing at the desk
    terminal   running a shell / python command
    file       reading or editing a file
    tool       any other tool call
    visiting   messaging/delegating to another agent — walks to its desk
    approval   blocked on a human approval
    asking     blocked on a question to the user
    replying   writing the final answer
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

STATES = (
    'idle', 'thinking', 'terminal', 'file', 'tool',
    'visiting', 'approval', 'asking', 'replying',
)

TERMINAL_TOOLS = {'bash', 'runpy', 'monitor', 'database_query'}
FILE_TOOLS = {
    'read_file', 'write_file', 'str_replace', 'patch', 'delete_file',
    'read_attachment', 'portal_copy', 'save_artifact', 'fetch_artifact',
    'list_artifacts', 'send_file',
}
VISIT_TOOLS = {'send_agent_message', 'resolve_agent_approval'}
ASK_TOOLS = {'ask_user_question', 'escalate_to_user'}

DETAIL_MAX = 120
JOURNAL_SIZE = 500


def _now_ms() -> int:
    return int(time.time() * 1000)


def _short(text: Any, limit: int = DETAIL_MAX) -> str:
    """Collapse whitespace and clip to ``limit`` chars."""
    if text is None:
        return ''
    if not isinstance(text, str):
        text = str(text)
    text = ' '.join(text.split())
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + '…'
    return text


def _first(args: Dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = args.get(key)
        if value not in (None, '', [], {}):
            return value if isinstance(value, str) else str(value)
    return ''


def tool_state(tool_name: str) -> str:
    """Return the visual state for a tool call."""
    name = (tool_name or '').strip()
    base = name.split(':', 1)[-1].split('.')[-1].lower()
    if base in TERMINAL_TOOLS:
        return 'terminal'
    if base in FILE_TOOLS:
        return 'file'
    if base in VISIT_TOOLS:
        return 'visiting'
    if base in ASK_TOOLS:
        return 'asking'
    return 'tool'


def tool_detail(tool_name: str, args: Any) -> str:
    """Pick the most telling argument of a tool call for the speech bubble."""
    if not isinstance(args, dict):
        return _short(args)
    state = tool_state(tool_name)
    if state == 'terminal':
        text = _first(args, 'command', 'cmd', 'script', 'code', 'query')
        if text:
            # Show the first non-empty line of multi-line scripts.
            text = next((ln for ln in text.splitlines() if ln.strip()), text)
        return _short(text)
    if state == 'file':
        return _short(_first(args, 'path', 'file_path', 'filename', 'name',
                             'artifact_id', 'src'))
    if state == 'visiting':
        return _short(_first(args, 'message', 'content'))
    if state == 'asking':
        return _short(_first(args, 'question', 'message', 'reason'))
    return _short(_first(args, 'query', 'url', 'path', 'name', 'message'))


def map_event(event_name: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Map one runtime event to an activity dict, or ``None`` to ignore it."""
    if not isinstance(data, dict):
        return None
    agent_id = str(data.get('agent_id') or '').strip()
    if not agent_id:
        return None

    activity: Dict[str, Any] = {
        'agent_id': agent_id,
        'state': None,
        'tool_name': '',
        'detail': '',
        'target_agent_id': '',
        'session_id': str(data.get('session_id') or ''),
    }

    if event_name in ('turn_begin', 'processing_started'):
        activity['state'] = 'thinking'
    elif event_name == 'agent_busy_changed':
        activity['state'] = 'thinking' if data.get('busy') else 'idle'
    elif event_name == 'llm_thinking':
        activity['state'] = 'thinking'
        activity['detail'] = _short(data.get('thinking'))
    elif event_name == 'tool_call_started':
        name = str(data.get('tool_name') or '')
        args = data.get('tool_args') or {}
        activity['state'] = tool_state(name)
        activity['tool_name'] = name
        activity['detail'] = tool_detail(name, args)
        if activity['state'] == 'visiting' and isinstance(args, dict):
            activity['target_agent_id'] = str(
                args.get('target_agent_id') or ''
            ).strip().lower()
    elif event_name == 'tool_executed':
        # Back to the desk to think about the result.
        activity['state'] = 'thinking'
        activity['tool_name'] = str(data.get('tool_name') or '')
        if data.get('has_error'):
            activity['detail'] = f"{activity['tool_name']} failed"
    elif event_name == 'approval_required':
        name = str(data.get('tool_name') or '')
        activity['state'] = 'approval'
        activity['tool_name'] = name
        detail = tool_detail(name, data.get('tool_args') or {})
        activity['detail'] = _short(f"Needs approval: {name} {detail}".strip())
    elif event_name == 'approval_resolved':
        activity['state'] = 'thinking'
        decision = data.get('decision')
        if decision:
            activity['detail'] = _short(f"Approval {decision}")
    elif event_name == 'final_answer':
        activity['state'] = 'replying'
        activity['detail'] = _short(data.get('answer'))
    elif event_name == 'turn_complete':
        activity['state'] = 'idle'
        activity['detail'] = _short(data.get('response'))
    else:
        return None

    return activity


class OfficeTracker:
    """Thread-safe latest-activity map plus a bounded, sequenced journal."""

    def __init__(self, journal_size: int = JOURNAL_SIZE):
        self._lock = threading.Condition()
        self._latest: Dict[str, Dict[str, Any]] = {}
        self._journal: Deque[Dict[str, Any]] = deque(maxlen=journal_size)
        self._seq = 0

    @property
    def seq(self) -> int:
        with self._lock:
            return self._seq

    def record(self, activity: Dict[str, Any]) -> Dict[str, Any]:
        """Store ``activity`` as the agent's latest state and journal it."""
        with self._lock:
            self._seq += 1
            entry = dict(activity)
            entry['seq'] = self._seq
            entry.setdefault('ts', _now_ms())
            prev = self._latest.get(entry['agent_id'])
            # Keep the last useful text when an event carries none.
            if not entry.get('detail') and prev and entry['state'] in (
                    'thinking', 'idle'):
                entry['last_detail'] = prev.get('detail') or prev.get(
                    'last_detail', '')
            self._latest[entry['agent_id']] = entry
            self._journal.append(entry)
            self._lock.notify_all()
            return entry

    def handle(self, event_name: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        activity = map_event(event_name, data)
        if activity is None:
            return None
        return self.record(activity)

    def latest(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {k: dict(v) for k, v in self._latest.items()}

    def since(self, seq: int) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(e) for e in self._journal if e['seq'] > seq]

    def wait(self, seq: int, timeout: float) -> List[Dict[str, Any]]:
        """Block up to ``timeout`` seconds for entries newer than ``seq``."""
        with self._lock:
            if self._seq <= seq:
                self._lock.wait(timeout)
            return [dict(e) for e in self._journal if e['seq'] > seq]


tracker = OfficeTracker()
