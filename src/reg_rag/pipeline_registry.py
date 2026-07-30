from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable


PipelineHandler = Callable[..., Any]


class PipelineState(str, Enum):
    ENABLED = "ENABLED"
    READY_GATED = "READY_GATED"
    READY_SHADOW = "READY_SHADOW"
    GATED_OFF = "GATED_OFF"
    NOT_READY = "NOT_READY"
    DISABLED = "DISABLED"


class PipelineUnavailableError(RuntimeError):
    def __init__(self, name: str, state: PipelineState) -> None:
        super().__init__(f"pipeline {name!r} is not enabled: {state.value}")
        self.name = name
        self.state = state


@dataclass(frozen=True)
class PipelineRegistration:
    name: str
    component: str
    state: PipelineState
    description: str
    handler: PipelineHandler | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "component": self.component,
            "state": self.state.value,
            "description": self.description,
            "has_handler": self.handler is not None,
        }


class PipelineRegistry:
    def __init__(self) -> None:
        self._registrations: dict[str, PipelineRegistration] = {}

    def register(self, registration: PipelineRegistration) -> None:
        if registration.name in self._registrations:
            raise ValueError(f"pipeline already registered: {registration.name}")
        self._registrations[registration.name] = registration

    def get(self, name: str) -> PipelineRegistration:
        try:
            return self._registrations[name]
        except KeyError as exc:
            raise KeyError(f"pipeline is not registered: {name}") from exc

    def resolve(
        self,
        name: str,
        *,
        allow_shadow: bool = False,
        allow_gated: bool = False,
    ) -> PipelineHandler:
        registration = self.get(name)
        permitted = registration.state is PipelineState.ENABLED or (
            allow_shadow and registration.state is PipelineState.READY_SHADOW
        ) or (
            allow_gated and registration.state is PipelineState.READY_GATED
        )
        if not permitted or registration.handler is None:
            raise PipelineUnavailableError(name, registration.state)
        return registration.handler

    def describe(self) -> list[dict[str, Any]]:
        return [self._registrations[name].to_dict() for name in sorted(self._registrations)]

    def states(self) -> dict[str, str]:
        return {name: item.state.value for name, item in sorted(self._registrations.items())}
