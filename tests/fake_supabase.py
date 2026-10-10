"""A minimal in-memory stand-in for a supabase-py sync client: PostgREST query builder, rpc and storage.

Only the builder surface SupabaseRepo uses is implemented. It records every executed query and an ordered
event log so tests can pin call shapes (filters, rpc params, upload-before-insert) without any network.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from datetime import datetime
from typing import Any, cast

from postgrest.exceptions import APIError
from storage3.exceptions import StorageApiError

PRIMARY_KEYS: dict[str, tuple[str, ...]] = {
    "lenders": ("id", "version"),
    "lender_pincodes": ("lender_id", "version", "pincode"),
    "consents": ("id",),
    "pan_records": ("pan_hash",),
    "score_cache": ("pan_hash",),
    "report_cache": ("pan_hash",),
    "checks": ("id",),
    "surepass_calls": ("id",),
    "pan_locks": ("pan_hash", "kind"),
    "review_queue": ("id",),
    "lc_quota_overrides": ("lc_id",),
    "opportunity_pan_events": ("id",),
}


class FakeResponse:
    def __init__(self, data: Any) -> None:
        self.data = data


def _norm(value: Any) -> Any:
    """Compare ISO timestamps as datetimes (the real database compares timestamptz values)."""
    if isinstance(value, str) and "T" in value:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return value
    return value


def _payloads(payload: Any) -> list[dict[str, Any]]:
    items = cast(list[Any], payload) if isinstance(payload, list) else [payload]
    return [cast(dict[str, Any], item) for item in items]


def _sort_key(column: str) -> Callable[[dict[str, Any]], Any]:
    def key(row: dict[str, Any]) -> Any:
        return _norm(row.get(column))

    return key


class FakeQuery:
    def __init__(self, client: FakeSupabase, table: str) -> None:
        self.client = client
        self.table = table
        self.op = "select"
        self.columns = "*"
        self.payload: Any = None
        self.on_conflict: str = ""
        self.ignore_duplicates = False
        self.filters: list[tuple[str, str, Any]] = []
        self.orders: list[tuple[str, bool]] = []
        self.limit_n: int | None = None
        self.range_: tuple[int, int] | None = None

    # ---- verbs
    def select(self, columns: str = "*") -> FakeQuery:
        self.op, self.columns = "select", columns
        return self

    def insert(self, payload: Any) -> FakeQuery:
        self.op, self.payload = "insert", payload
        return self

    def update(self, payload: dict[str, Any]) -> FakeQuery:
        self.op, self.payload = "update", payload
        return self

    def upsert(self, payload: Any, *, on_conflict: str = "", ignore_duplicates: bool = False) -> FakeQuery:
        self.op, self.payload = "upsert", payload
        self.on_conflict, self.ignore_duplicates = on_conflict, ignore_duplicates
        return self

    def delete(self) -> FakeQuery:
        self.op = "delete"
        return self

    # ---- filters / modifiers
    def eq(self, column: str, value: Any) -> FakeQuery:
        self.filters.append(("eq", column, value))
        return self

    def lt(self, column: str, value: Any) -> FakeQuery:
        self.filters.append(("lt", column, value))
        return self

    def gt(self, column: str, value: Any) -> FakeQuery:
        self.filters.append(("gt", column, value))
        return self

    def is_(self, column: str, value: Any) -> FakeQuery:
        self.filters.append(("is", column, value))
        return self

    def in_(self, column: str, values: list[Any]) -> FakeQuery:
        self.filters.append(("in", column, list(values)))
        return self

    def order(self, column: str, *, desc: bool = False) -> FakeQuery:
        self.orders.append((column, desc))
        return self

    def limit(self, n: int) -> FakeQuery:
        self.limit_n = n
        return self

    def range(self, start: int, end: int) -> FakeQuery:
        self.range_ = (start, end)
        return self

    def execute(self) -> FakeResponse:
        return self.client.run(self)

    # ---- helpers for assertions
    def has_filter(self, op: str, column: str, value: Any = ...) -> bool:
        return any(f[0] == op and f[1] == column and (value is ... or f[2] == value) for f in self.filters)


class FakeBucket:
    def __init__(self, client: FakeSupabase, name: str) -> None:
        self.client, self.name = client, name

    def upload(self, path: str, file: bytes, file_options: dict[str, str] | None = None) -> dict[str, str]:
        self.client.events.append(f"upload:{path}")
        self.client.uploads.append((self.name, path, dict(file_options or {})))
        self.client.objects[(self.name, path)] = file
        return {"path": path}

    def download(self, path: str) -> bytes:
        self.client.events.append(f"download:{path}")
        if (self.name, path) not in self.client.objects:
            raise StorageApiError("Object not found", "not_found", 404)
        return self.client.objects[(self.name, path)]

    def remove(self, paths: list[str]) -> list[dict[str, Any]]:
        self.client.events.append(f"remove:{len(paths)}")
        self.client.removals.append(list(paths))
        removed: list[dict[str, Any]] = []
        for path in paths:
            if self.client.objects.pop((self.name, path), None) is not None:
                removed.append({"name": path})
        return removed


class FakeStorage:
    def __init__(self, client: FakeSupabase) -> None:
        self.client = client

    def from_(self, bucket: str) -> FakeBucket:
        return FakeBucket(self.client, bucket)


class FakeRpc:
    def __init__(self, client: FakeSupabase, fn: str, params: dict[str, Any]) -> None:
        self.client, self.fn, self.params = client, fn, params

    def execute(self) -> FakeResponse:
        self.client.events.append(f"rpc:{self.fn}")
        self.client.rpcs.append((self.fn, copy.deepcopy(self.params)))
        return FakeResponse(self.client.rpc_handlers[self.fn](self.params))


class FakeSupabase:
    def __init__(self) -> None:
        self.tables: dict[str, list[dict[str, Any]]] = {name: [] for name in PRIMARY_KEYS}
        self.events: list[str] = []
        self.queries: list[FakeQuery] = []
        self.rpcs: list[tuple[str, dict[str, Any]]] = []
        self.rpc_handlers: dict[str, Callable[[dict[str, Any]], Any]] = {}
        self.objects: dict[tuple[str, str], bytes] = {}
        self.uploads: list[tuple[str, str, dict[str, str]]] = []
        self.removals: list[list[str]] = []
        self.storage = FakeStorage(self)
        self.max_rows = 1000  # PostgREST's default max-rows cap on any response

    # ---- supabase-py surface
    def table(self, name: str) -> FakeQuery:
        return FakeQuery(self, name)

    def rpc(self, fn: str, params: dict[str, Any] | None = None) -> FakeRpc:
        return FakeRpc(self, fn, params or {})

    # ---- query execution
    def _matches(self, row: dict[str, Any], query: FakeQuery) -> bool:
        for op, column, value in query.filters:
            cell: Any = row.get(column)
            expected: Any = value
            if op == "eq":
                ok = _norm(cell) == _norm(expected)
            elif op == "is":
                ok = cell is None if expected in ("null", None) else cell == expected
            elif op == "in":
                ok = cell in cast(list[Any], expected)
            elif op == "lt":
                ok = cell is not None and _norm(cell) < _norm(expected)
            elif op == "gt":
                ok = cell is not None and _norm(cell) > _norm(expected)
            else:
                raise NotImplementedError(op)
            if not ok:
                return False
        return True

    def run(self, query: FakeQuery) -> FakeResponse:
        self.queries.append(query)
        self.events.append(f"{query.op}:{query.table}")
        rows = self.tables[query.table]
        keys = PRIMARY_KEYS[query.table]

        if query.op == "insert":
            inserted: list[dict[str, Any]] = []
            for payload in _payloads(query.payload):
                if any(all(r[k] == payload[k] for k in keys) for r in rows):
                    raise APIError({"message": "duplicate key value", "code": "23505", "hint": None, "details": None})
                rows.append(copy.deepcopy(payload))
                inserted.append(copy.deepcopy(payload))
            return FakeResponse(inserted)

        if query.op == "upsert":
            conflict = tuple(c.strip() for c in query.on_conflict.split(",")) if query.on_conflict else keys
            affected: list[dict[str, Any]] = []
            for payload in _payloads(query.payload):
                existing = next((r for r in rows if all(r[k] == payload[k] for k in conflict)), None)
                if existing is None:
                    rows.append(copy.deepcopy(payload))
                    affected.append(copy.deepcopy(payload))
                elif not query.ignore_duplicates:
                    existing.update(copy.deepcopy(payload))
                    affected.append(copy.deepcopy(existing))
            return FakeResponse(affected)

        matched = [r for r in rows if self._matches(r, query)]
        if query.op == "update":
            for row in matched:
                row.update(copy.deepcopy(query.payload))
            return FakeResponse(copy.deepcopy(matched))
        if query.op == "delete":
            self.tables[query.table] = [r for r in rows if r not in matched]
            return FakeResponse(copy.deepcopy(matched))

        for column, desc in reversed(query.orders):
            matched.sort(key=_sort_key(column), reverse=desc)
        if query.range_ is not None:
            matched = matched[query.range_[0] : query.range_[1] + 1]
        if query.limit_n is not None:
            matched = matched[: query.limit_n]
        matched = matched[: self.max_rows]
        if query.columns != "*":
            wanted = [c.strip() for c in query.columns.split(",")]
            matched = [{c: r.get(c) for c in wanted} for r in matched]
        return FakeResponse(copy.deepcopy(matched))
