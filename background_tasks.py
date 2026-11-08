"""Bounded, process-local background task registry for expensive API work."""

from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
import hashlib
import json
from threading import Lock


MAX_WORKERS = 2
MAX_RETAINED_TASKS = 64
MAX_IN_FLIGHT_TASKS = 4

_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="artha-ai")
_lock = Lock()
_tasks = OrderedDict()


def make_task_key(namespace: str, *parts) -> str:
    """Return a stable opaque key without exposing document text in logs/state."""
    encoded = json.dumps(
        [namespace, *parts], ensure_ascii=False, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _remove_finished_entries():
    while len(_tasks) > MAX_RETAINED_TASKS:
        key, future = next(iter(_tasks.items()))
        if not future.done():
            break
        _tasks.pop(key, None)


def submit_task(
    key: str, function, *args, retry_failed: bool = False, **kwargs
) -> Future | None:
    """Reuse a task by key, or submit one if the bounded queue has capacity."""
    with _lock:
        existing = _tasks.get(key)
        if existing is not None:
            if retry_failed and existing.done() and existing.exception() is not None:
                _tasks.pop(key, None)
            else:
                _tasks.move_to_end(key)
                return existing

        in_flight = sum(not future.done() for future in _tasks.values())
        if in_flight >= MAX_IN_FLIGHT_TASKS:
            return None

        future = _executor.submit(function, *args, **kwargs)
        _tasks[key] = future
        _tasks.move_to_end(key)
        _remove_finished_entries()
        return future


def forget_task(key: str, future: Future | None = None):
    with _lock:
        if future is None or _tasks.get(key) is future:
            _tasks.pop(key, None)

