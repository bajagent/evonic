"""Cross-platform advisory file locking for Evonic's single-instance guards.

Evonic guards a few singleton resources (the server PID file and the update
lock) with an OS-level advisory lock held for the lifetime of a file
descriptor.  Historically those call sites used ``fcntl.flock`` directly,
which does not exist on Windows and therefore crashed startup there.

This module is a thin, dependency-free abstraction over the platform's native
advisory locking:

* POSIX (Linux/macOS) -> :func:`fcntl.flock` with ``LOCK_EX``.
* Windows             -> :func:`msvcrt.locking` on a one-byte region.

The public API is intentionally tiny:

``open_and_lock(path)``
    Open (creating if needed) and acquire an exclusive, *non-blocking* lock.
    Raises :class:`AlreadyLockedError` when another owner already holds it.

``release(fd)``
    Unlock and close a descriptor previously returned by ``open_and_lock``.

Both platform modules are imported lazily *inside* the helpers so importing
this module never requires ``fcntl`` (absent on Windows) or ``msvcrt`` (absent
on POSIX).
"""

from __future__ import annotations

import errno
import os
import sys

__all__ = ["AlreadyLockedError", "open_and_lock", "release"]

# Selected once at import time so the hot path does not re-check ``sys.platform``.
# Tests monkeypatch this flag to exercise both branches on one host.
_IS_WINDOWS = sys.platform == "win32"

# Windows byte-range locks are mandatory: a locked byte cannot be read by any
# other process. Locking byte 0 therefore hid the PID written at the start of
# the PID file from ``evonic status``/``stop``. Lock a single byte far past any
# real content instead (LockFile allows locking beyond EOF), so the lock still
# guards the singleton while the file's contents stay readable.
_WIN_LOCK_OFFSET = 1 << 30


class AlreadyLockedError(OSError):
    """Raised when a non-blocking lock is already held by another owner.

    Subclasses :class:`OSError` so existing ``except OSError`` / ``except
    IOError`` handlers around the old ``fcntl.flock`` calls keep working.
    """


def _is_contention(exc: OSError) -> bool:
    """Return True when ``exc`` means "already locked" rather than a real error.

    ``fcntl.flock(LOCK_NB)`` reports contention as ``EAGAIN``/``EWOULDBLOCK``
    (sometimes ``EACCES``); ``msvcrt.locking(LK_NBLCK)`` reports it as
    ``EACCES`` or ``EDEADLOCK``.
    """
    if isinstance(exc, BlockingIOError):
        return True
    errnos = {
        errno.EACCES,
        errno.EAGAIN,
        getattr(errno, "EWOULDBLOCK", errno.EAGAIN),
        getattr(errno, "EDEADLK", -1),
        getattr(errno, "EDEADLOCK", -1),
    }
    return exc.errno in errnos


def _ensure_lockable_size(fd: int) -> None:
    """Guarantee the Windows byte-range lock target exists.

    ``msvcrt.locking`` locks ``nbytes`` bytes starting at the current file
    position and cannot reliably lock a zero-length file, so pad the file with
    a single byte when it is empty and rewind to offset 0.  This is a no-op on
    POSIX, where ``flock`` is length-agnostic — POSIX semantics stay unchanged.
    """
    if not _IS_WINDOWS:
        return
    if os.fstat(fd).st_size == 0:
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, b"\n")
        os.fsync(fd)
    os.lseek(fd, 0, os.SEEK_SET)


def _lock(fd: int, blocking: bool) -> None:
    """Acquire an exclusive lock on ``fd``."""
    if _IS_WINDOWS:
        import msvcrt

        _ensure_lockable_size(fd)
        mode = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
        os.lseek(fd, _WIN_LOCK_OFFSET, os.SEEK_SET)
        try:
            msvcrt.locking(fd, mode, 1)
        finally:
            os.lseek(fd, 0, os.SEEK_SET)
    else:
        import fcntl

        flags = fcntl.LOCK_EX
        if not blocking:
            flags |= fcntl.LOCK_NB
        fcntl.flock(fd, flags)


def _unlock(fd: int) -> None:
    """Release the lock held on ``fd`` (without closing it)."""
    if _IS_WINDOWS:
        import msvcrt

        os.lseek(fd, _WIN_LOCK_OFFSET, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)


def open_and_lock(path: str, *, blocking: bool = False) -> int:
    """Open ``path`` and acquire an exclusive advisory lock on it.

    Args:
        path: File to lock. The parent directory must already exist.
        blocking: When ``False`` (default) fail fast with
            :class:`AlreadyLockedError` if the lock is held. When ``True``
            wait for the current owner to release it.

    Returns:
        The open file descriptor, which **must stay open** for as long as the
        lock is meant to be held (the OS drops the lock when it closes or the
        process exits). Pass it to :func:`release` to unlock and close.

    Raises:
        AlreadyLockedError: the lock is held by another owner.
        OSError: the file could not be opened.
    """
    flags = os.O_CREAT | os.O_RDWR
    # Keep the lock fd out of exec'd children and make Windows do no CRLF
    # translation on the descriptor. Both flags are absent on some platforms,
    # hence the getattr fallbacks.
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_BINARY", 0)

    fd = os.open(path, flags)
    try:
        _lock(fd, blocking)
    except OSError as exc:
        os.close(fd)
        if not blocking and _is_contention(exc):
            raise AlreadyLockedError(
                exc.errno, f"File is already locked: {path}"
            ) from exc
        raise
    return fd


def release(fd) -> None:
    """Unlock and close a descriptor returned by :func:`open_and_lock`.

    Safe to call with ``None`` and tolerant of an already-released lock, so it
    can be used directly in ``finally`` blocks.
    """
    if fd is None:
        return
    try:
        _unlock(fd)
    except OSError:
        # The lock may already be gone (e.g. process teardown); closing the
        # descriptor below is what actually matters for cleanup.
        pass
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
