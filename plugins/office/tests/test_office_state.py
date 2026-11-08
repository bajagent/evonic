"""Tests for the Office plugin's event -> activity mapping and tracker."""

import os
import sys
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from plugins.office.state import OfficeTracker, map_event, tool_state  # noqa: E402


def test_tool_states():
    assert tool_state('bash') == 'terminal'
    assert tool_state('runpy') == 'terminal'
    assert tool_state('str_replace') == 'file'
    assert tool_state('read_file') == 'file'
    assert tool_state('send_agent_message') == 'visiting'
    assert tool_state('ask_user_question') == 'asking'
    assert tool_state('kanban:update_status') == 'tool'
    assert tool_state('') == 'tool'


def test_bash_call_shows_first_command_line():
    act = map_event('tool_call_started', {
        'agent_id': 'super', 'session_id': 's1', 'tool_name': 'bash',
        'tool_args': {'command': '\n  cd /workspace\nnpm test'},
    })
    assert act['state'] == 'terminal'
    assert act['tool_name'] == 'bash'
    assert act['detail'] == 'cd /workspace'


def test_file_edit_shows_path():
    act = map_event('tool_call_started', {
        'agent_id': 'a', 'tool_name': 'str_replace',
        'tool_args': {'path': 'src/app.py', 'old_str': 'x', 'new_str': 'y'},
    })
    assert act['state'] == 'file'
    assert act['detail'] == 'src/app.py'


def test_agent_message_sets_visit_target():
    act = map_event('tool_call_started', {
        'agent_id': 'super', 'tool_name': 'send_agent_message',
        'tool_args': {'target_agent_id': ' Hangie ', 'message': 'Please review the PR'},
    })
    assert act['state'] == 'visiting'
    assert act['target_agent_id'] == 'hangie'
    assert act['detail'] == 'Please review the PR'


def test_lifecycle_events():
    assert map_event('turn_begin', {'agent_id': 'a'})['state'] == 'thinking'
    assert map_event('agent_busy_changed', {'agent_id': 'a', 'busy': True})['state'] == 'thinking'
    assert map_event('agent_busy_changed', {'agent_id': 'a', 'busy': False})['state'] == 'idle'
    approval = map_event('approval_required', {
        'agent_id': 'a', 'tool_name': 'bash', 'tool_args': {'command': 'rm -rf build'},
        'approval_id': 'ap-1'})
    assert approval['state'] == 'approval'
    assert approval['approval_id'] == 'ap-1'
    assert 'rm -rf build' in approval['detail']
    assert map_event('approval_resolved', {'agent_id': 'a', 'decision': 'approved'})['state'] == 'thinking'
    reply = map_event('final_answer', {'agent_id': 'a', 'answer': 'All   done\nthanks'})
    assert reply['state'] == 'replying'
    assert reply['detail'] == 'All done thanks'
    assert map_event('turn_complete', {'agent_id': 'a', 'response': 'ok'})['state'] == 'idle'


def test_ignores_unknown_or_anonymous_events():
    assert map_event('message_sent', {'agent_id': 'a'}) is None
    assert map_event('tool_call_started', {'tool_name': 'bash'}) is None
    assert map_event('turn_begin', None) is None


def test_detail_is_clipped():
    act = map_event('llm_thinking', {'agent_id': 'a', 'thinking': 'x' * 500})
    assert len(act['detail']) <= 120
    assert act['detail'].endswith('…')


def test_tracker_sequences_and_keeps_last_detail():
    tr = OfficeTracker(journal_size=3)
    tr.handle('tool_call_started', {'agent_id': 'a', 'tool_name': 'bash', 'tool_args': {'command': 'ls'}})
    tr.handle('tool_executed', {'agent_id': 'a', 'tool_name': 'bash'})
    latest = tr.latest()['a']
    assert latest['state'] == 'thinking'
    assert latest['last_detail'] == 'ls'
    assert tr.seq == 2
    for i in range(3):
        tr.handle('turn_begin', {'agent_id': f'b{i}'})
    assert [e['seq'] for e in tr.since(0)] == [3, 4, 5]   # bounded journal
    assert tr.since(5) == []


def test_tracker_wait_wakes_on_new_activity():
    tr = OfficeTracker()
    result = {}

    def waiter():
        result['entries'] = tr.wait(0, timeout=5)

    th = threading.Thread(target=waiter)
    th.start()
    tr.handle('turn_begin', {'agent_id': 'a'})
    th.join(2)
    assert not th.is_alive()
    assert result['entries'][0]['agent_id'] == 'a'
