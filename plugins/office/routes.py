"""HTTP routes for the Office view: the page, a state snapshot and an SSE tail."""

import json
import os
import time

from flask import Blueprint, Response, jsonify, render_template, request, stream_with_context

try:
    from .state import tracker
except ImportError:  # pragma: no cover - direct import fallback
    from plugins.office.state import tracker

PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))

# How long one SSE connection lives before the browser reconnects. Keeps
# worker threads from being pinned forever by forgotten tabs.
STREAM_MAX_SECONDS = 300
HEARTBEAT_SECONDS = 15


def _agents_snapshot():
    from models.db import db
    agents = []
    for agent in db.get_agents() or []:
        if agent.get('enabled') is not None and not agent.get('enabled'):
            continue
        agents.append({
            'id': agent.get('id'),
            'name': agent.get('name') or agent.get('id'),
            'description': (agent.get('description') or '')[:160],
            'workplace_id': agent.get('workplace_id') or '',
            'is_super': bool(agent.get('is_super')),
            'has_avatar': bool(agent.get('avatar_path')),
        })
    workplaces = []
    try:
        for wp in db.get_workplaces() or []:
            workplaces.append({'id': wp.get('id'), 'name': wp.get('name') or wp.get('id')})
    except Exception:
        pass
    return agents, workplaces


def _busy_agents():
    try:
        from backend.realtime_store import realtime_store
        return realtime_store.busy_agents()
    except Exception:
        return {}


def build_state():
    agents, workplaces = _agents_snapshot()
    latest = tracker.latest()
    busy = _busy_agents()
    now = int(time.time() * 1000)
    activity = {}
    for agent in agents:
        aid = agent['id']
        entry = latest.get(aid)
        if entry is None or (entry['state'] == 'idle' and aid in busy):
            entry = {'agent_id': aid, 'state': 'thinking' if aid in busy else 'idle',
                     'tool_name': '', 'detail': '', 'target_agent_id': '',
                     'ts': now, 'seq': 0}
        elif aid not in busy and entry['state'] not in ('idle', 'approval', 'asking'):
            # The tracker missed the end of a turn (e.g. server restart).
            entry = dict(entry, state='idle')
        activity[aid] = entry
    return {
        'agents': agents,
        'workplaces': workplaces,
        'activity': activity,
        'seq': tracker.seq,
        'server_time': now,
    }


def create_blueprint():
    bp = Blueprint('office', __name__, template_folder=os.path.join(PLUGIN_DIR, 'templates'))

    @bp.route('/board/office')
    def office_page():
        return render_template('office.html')

    @bp.route('/api/office/state')
    def office_state():
        return jsonify(build_state())

    @bp.route('/api/office/stream')
    def office_stream():
        try:
            cursor = int(request.args.get('after') or request.headers.get('Last-Event-ID') or 0)
        except ValueError:
            cursor = 0
        if cursor > tracker.seq:  # server restarted; client cursor is stale
            cursor = 0

        try:  # don't hold a DB connection open for the life of the stream
            from models.db import db
            db.close()
        except Exception:
            pass

        @stream_with_context
        def generate():
            nonlocal cursor
            started = time.time()
            last_beat = started
            yield 'retry: 3000\n\n'
            while time.time() - started < STREAM_MAX_SECONDS:
                entries = tracker.wait(cursor, timeout=1.0)
                for entry in entries:
                    cursor = entry['seq']
                    yield f"id: {cursor}\nevent: activity\ndata: {json.dumps(entry)}\n\n"
                if not entries and time.time() - last_beat >= HEARTBEAT_SECONDS:
                    last_beat = time.time()
                    yield ': ping\n\n'

        return Response(generate(), mimetype='text/event-stream', headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        })

    return bp
