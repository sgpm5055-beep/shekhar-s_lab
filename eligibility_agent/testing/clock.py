"""FrozenClock: the only time source any test or eval uses.

An injected clock (``now`` is a bound method, so it satisfies ``db.repo.Clock = Callable[[], datetime]``) keeps guards
and the repo pure functions of ``(inputs, now)`` instead of monkeypatching ``datetime`` module-wide.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


class FrozenClock:
    """Holds one aware instant, always normalised to UTC; the test moves it by hand."""

    def __init__(self, start: datetime) -> None:
        if start.tzinfo is None:
            raise ValueError("FrozenClock requires an aware datetime")
        self._now: datetime = start.astimezone(timezone.utc)

    def now(self) -> datetime:
        return self._now

    def advance(self, delta: timedelta) -> None:
        self._now = self._now + delta

    def set_now(self, instant: datetime) -> None:
        if instant.tzinfo is None:
            raise ValueError("aware datetime required")
        self._now = instant.astimezone(timezone.utc)
