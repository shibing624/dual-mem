# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Bounded per-key asyncio.Lock registry. Both the write path (per app/user)
and the System2 drain path (per app/user/agent) need a lock per identity, but a long-running
server with high-cardinality users would otherwise accumulate one Lock per key forever.
LockRegistry caps the number of cached locks and evicts the least-recently-used entries that
are NOT currently held, so an in-flight critical section is never dropped.
"""
import asyncio
from collections import OrderedDict

_DEFAULT_MAX_LOCKS = 4096


class LockRegistry:
    """LRU-bounded registry handing out one asyncio.Lock per key.

    get() returns a stable Lock for a key. When the registry exceeds max_locks, it evicts
    the oldest entries whose lock is currently unlocked; held locks are kept so callers
    already inside `async with registry.get(k)` are never affected.
    """

    def __init__(self, *, max_locks: int = _DEFAULT_MAX_LOCKS):
        self._max_locks = max(1, max_locks)
        self._locks: OrderedDict[str, asyncio.Lock] = OrderedDict()

    def get(self, key: str) -> asyncio.Lock:
        """Return the lock for ``key``, creating it on first use and pruning if over cap."""
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        self._locks.move_to_end(key)
        self._prune()
        return lock

    def _prune(self) -> None:
        """Evict least-recently-used entries until within cap, skipping held locks."""
        if len(self._locks) <= self._max_locks:
            return
        # Iterate oldest-first; drop only unlocked entries so a held critical section
        # keeps its lock identity. Stop once we are back within the cap.
        for key in list(self._locks.keys()):
            if len(self._locks) <= self._max_locks:
                break
            lock = self._locks[key]
            if not lock.locked():
                del self._locks[key]

    def __len__(self) -> int:
        return len(self._locks)
