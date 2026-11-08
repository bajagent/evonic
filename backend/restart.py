"""
In-place server restart — single source of truth.

Restarts the running server by replacing the current process image with a fresh
one (os.execv), preserving the PID the run/ lock and PID file point to. This is
the only restart mechanism that works without an external process manager, so it
is correct under the documented `evonic start -d` install path as well as under
systemd/Docker.

Used by the /restart slash command, the super-agent restart tool, the web setup
auto-restart, and the web update "Restart" action so every restart path behaves
identically.
"""

import logging
import os
import subprocess
import sys
import threading
import time
import shutil

log = logging.getLogger(__name__)

# Tests patch this to exercise both platform branches on one host.
_IS_WINDOWS = sys.platform == 'win32'


def restart_in_place(delay: float = 1.5) -> None:
    """Stop channels and the scheduler, free inherited FDs, and re-exec the server.

    Runs in the calling thread and never returns on success (the process image is
    replaced). Intended to be invoked from a daemon thread via schedule_restart().
    """
    time.sleep(delay)  # Let the triggering response/log flush first.

    # Stop all channels cleanly so Telegram releases its long-poll before the
    # new process re-opens them (otherwise the bot token hits a getUpdates
    # Conflict on the next boot).
    try:
        from backend.channels.registry import channel_manager
        channel_manager.stop_all()
        time.sleep(1.0)  # Give Telegram server-side time to release.
    except Exception as e:
        log.error("Error stopping channels during restart: %s", e, exc_info=True)

    # Shut down the scheduler so in-flight jobs are not abandoned mid-execution.
    try:
        from backend.scheduler import scheduler as global_scheduler
        global_scheduler.shutdown()
    except Exception as e:
        log.error("Error shutting down scheduler during restart: %s", e, exc_info=True)

    # Close inherited file descriptors (including Flask's bound socket) so the
    # new process can re-bind the same port. close_range(inheritable=False) keeps
    # the FDs in this process but stops them leaking into the new image; fall back
    # to closerange on older Python. POSIX-only — skip on Windows.
    if not _IS_WINDOWS:
        try:
            import resource
            maxfd = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
            if maxfd == resource.RLIM_INFINITY or maxfd > 65535:
                maxfd = 4096
            try:
                os.close_range(3, maxfd, inheritable=False)
            except AttributeError:
                os.closerange(3, maxfd)
        except Exception:
            pass

    # Flat-repo architecture: project root IS the live directory.
    import config
    target = os.path.realpath(config.BASE_DIR)
    app_py = os.path.join(target, 'app.py')
    if _IS_WINDOWS:
        venv_python = os.path.join(target, '.venv', 'Scripts', 'python.exe')
    else:
        venv_python = os.path.join(target, '.venv', 'bin', 'python')
    python = venv_python if os.path.exists(venv_python) else sys.executable

    os.chdir(target)
    if _IS_WINDOWS:
        _respawn_windows(python, app_py, target)
        return  # unreachable: _respawn_windows exits the process

    log.info("Re-executing server process")
    os.execv(python, [python, app_py])


# Runs in a detached helper process on Windows: wait for the old server to
# exit (releasing its port and PID lock), then start a fresh one.
_WINDOWS_RESPAWN_SCRIPT = r'''
import ctypes, os, subprocess, sys, time
old_pid, python, app_py, log_path = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
k32 = ctypes.windll.kernel32
SYNCHRONIZE = 0x00100000
h = k32.OpenProcess(SYNCHRONIZE, False, old_pid)
if h:
    k32.WaitForSingleObject(h, 30000)
    k32.CloseHandle(h)
time.sleep(0.5)
env = dict(os.environ)
for _k in ("EVONIC_PID_LOCK_HELD", "EVONIC_TESTING", "EVONIC_SMOKE_TEST"):
    env.pop(_k, None)
flags = 0x00000008 | 0x00000200 | 0x08000000  # DETACHED | NEW_PROCESS_GROUP | NO_WINDOW
with open(log_path, "a") as out:
    subprocess.Popen([python, app_py], cwd=os.path.dirname(app_py), env=env,
                     stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                     close_fds=True, creationflags=flags)
'''


def _respawn_windows(python: str, app_py: str, target: str) -> None:
    """Restart on Windows, where os.execv cannot replace the process image.

    On Windows os.execv spawns a *new* process and exits the old one, so the
    old listening socket can outlive it while the new server binds the same
    port (Windows SO_REUSEADDR allows that) — requests then hang on the dead
    socket. Instead, hand off to a detached helper and exit cleanly so the OS
    releases the socket and PID lock before the replacement starts.
    """
    log_dir = os.path.join(target, 'shared', 'run')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'server.log')
    flags = 0x00000008 | 0x00000200 | 0x08000000
    log.info("Restarting server process (Windows respawn)")
    try:
        subprocess.Popen(
            [python, '-I', '-c', _WINDOWS_RESPAWN_SCRIPT,
             str(os.getpid()), python, app_py, log_path],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, close_fds=True, creationflags=flags,
        )
    except Exception:
        log.exception("Failed to spawn restart helper; server keeps running")
        return
    for handler in logging.getLogger().handlers:
        try:
            handler.flush()
        except Exception:
            pass
    os._exit(0)


def _systemd_command(action: str) -> None:
    """Run a systemctl action for the configured Evonic unit."""
    import config

    service_name = config.SYSTEMD_SERVICE_NAME
    if not service_name:
        raise RuntimeError(
            "SYSTEMD_SERVICE_NAME is required when SERVICE_SYSTEM=systemd"
        )
    command = []

    # Check if we are NOT root (UID 0)
    if os.geteuid() != 0:
        # Check if sudo is installed and in the PATH
        if shutil.which("sudo"):
            command.append("sudo")
        else:
            log.warning("Not running as root and 'sudo' is not installed. Command will likely fail.")

    command.extend(["systemctl", action, service_name])

    log.info("Managing Evonic via systemd: %s", " ".join(command))
    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError):
        log.exception("Failed to run systemd service command: %s", action)
        raise


def restart_service(delay: float = 1.5) -> None:
    """Restart using the configured service manager or the in-place fallback."""
    import config

    if config.SERVICE_SYSTEM == "systemd":
        if not config.SYSTEMD_SERVICE_NAME:
            raise RuntimeError(
                "SYSTEMD_SERVICE_NAME is required when SERVICE_SYSTEM=systemd"
            )

        def _restart_systemd():
            time.sleep(delay)
            _systemd_command("restart")

        threading.Thread(target=_restart_systemd, daemon=True).start()
        return

    if config.SERVICE_SYSTEM:
        log.warning(
            "Unsupported SERVICE_SYSTEM=%r; using in-place restart",
            config.SERVICE_SYSTEM,
        )
    schedule_restart(delay)


def stop_service(delay: float = 1.5, fallback=None) -> None:
    """Stop via the configured service manager or invoke the supplied fallback."""
    import config

    if config.SERVICE_SYSTEM == "systemd":
        if not config.SYSTEMD_SERVICE_NAME:
            raise RuntimeError(
                "SYSTEMD_SERVICE_NAME is required when SERVICE_SYSTEM=systemd"
            )

        def _stop_systemd():
            time.sleep(delay)
            _systemd_command("stop")

        threading.Thread(target=_stop_systemd, daemon=True).start()
        return

    if config.SERVICE_SYSTEM:
        log.warning(
            "Unsupported SERVICE_SYSTEM=%r; using process stop fallback",
            config.SERVICE_SYSTEM,
        )
    if fallback is not None:
        threading.Thread(target=fallback, daemon=True).start()


def schedule_restart(delay: float = 1.5) -> None:
    """Run restart_in_place() in a daemon thread and return immediately."""
    threading.Thread(target=restart_in_place, args=(delay,), daemon=True).start()
