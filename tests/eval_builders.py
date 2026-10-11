"""Shared builders for the eval tests: a ``langsmith.Client`` stand-in that never touches the network."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from langsmith.schemas import ExampleCreate


class _Dataset:
    def __init__(self, dataset_id: uuid.UUID) -> None:
        self.id = dataset_id


class RecordingClient:
    """A langsmith.Client stand-in: no network. Records every call; ``existing`` names pretend to exist already."""

    def __init__(self, existing: Sequence[str] = ()) -> None:
        self.existing = set(existing)
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.ids: dict[str, uuid.UUID] = {}

    def has_dataset(self, *, dataset_name: str) -> bool:
        self.calls.append(("has_dataset", {"dataset_name": dataset_name}))
        return dataset_name in self.existing

    def _dataset(self, name: str) -> _Dataset:
        return _Dataset(self.ids.setdefault(name, uuid.uuid4()))

    def create_dataset(self, dataset_name: str, *, description: str) -> _Dataset:
        self.calls.append(("create_dataset", {"dataset_name": dataset_name, "description": description}))
        self.existing.add(dataset_name)
        return self._dataset(dataset_name)

    def read_dataset(self, *, dataset_name: str) -> _Dataset:
        self.calls.append(("read_dataset", {"dataset_name": dataset_name}))
        return self._dataset(dataset_name)

    def delete_dataset(self, *, dataset_id: uuid.UUID) -> None:
        self.calls.append(("delete_dataset", {"dataset_id": dataset_id}))

    def create_examples(self, *, dataset_id: uuid.UUID, examples: Sequence[ExampleCreate]) -> object:
        self.calls.append(("create_examples", {"dataset_id": dataset_id, "examples": list(examples)}))
        return None

    def of(self, name: str) -> list[dict[str, Any]]:
        return [kwargs for call, kwargs in self.calls if call == name]
