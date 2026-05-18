from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AdapterResult:
    url: str
    adapter: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class UrlAdapter(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def can_handle(self, url: str) -> bool:
        ...

    @abstractmethod
    async def extract(self, url: str, query: str = "", timeout: float = 30.0) -> AdapterResult:
        ...
