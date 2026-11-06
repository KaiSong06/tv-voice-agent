from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from .bridge import Screen
from .model import Decision


class Planner(Protocol):
    async def decide(self, command: str, screen: Screen, history: list[dict]) -> Decision: ...


class Bridge(Protocol):
    async def status(self) -> dict: ...
    async def capture(self) -> Screen: ...
    async def press(self, action: str) -> dict: ...
    async def volume(self, level: int) -> dict: ...


@dataclass
class Result:
    status: str
    message: str
    actions: list[dict] = field(default_factory=list)


class TVTask:
    def __init__(
        self,
        bridge: Bridge,
        planner: Planner,
        *,
        max_actions: int = 6,
        max_seconds: float = 120,
        max_screen_age: float = 20,
        settle_seconds: float = 0.7,
        progress: Callable[[str], None] = print,
    ):
        self.bridge, self.planner = bridge, planner
        self.max_actions, self.max_seconds = max_actions, max_seconds
        self.max_screen_age, self.settle_seconds = max_screen_age, settle_seconds
        self.progress = progress

    async def run(self, command: str, *, execute: bool = False) -> Result:
        if not command.strip() or len(command) > 2000:
            raise ValueError("Use a nonempty command of at most 2000 characters.")
        actions: list[dict] = []
        history: list[dict] = []
        started = time.monotonic()
        self.progress("Checking TV controls…")
        status = await self.bridge.status()
        if execute and status.get("connected") is not True:
            return Result("unavailable", "TV controls need Samsung sign-in. No action was sent.")
        available = set(status.get("actions", []))
        self.progress("Reading the TV screen…")
        screen = await self.bridge.capture()
        while True:
            if time.monotonic() - started >= self.max_seconds:
                return Result("limit", "Stopped at the task time limit.", actions)
            if time.monotonic() - screen.captured_at > self.max_screen_age:
                return Result(
                    "stale", "The screenshot is too old. Stopped without another action.", actions
                )
            self.progress("Choosing the next step…")
            decision = await asyncio.wait_for(
                self.planner.decide(command, screen, history),
                timeout=min(15, self.max_seconds - (time.monotonic() - started)),
            )
            if time.monotonic() - started >= self.max_seconds:
                return Result("limit", "Stopped at the task time limit.", actions)
            if time.monotonic() - screen.captured_at > self.max_screen_age:
                return Result(
                    "stale", "The screenshot became stale while planning. Stopped.", actions
                )
            if decision.kind in {"done", "ask"}:
                return Result(decision.kind, decision.message, actions)
            if len(actions) >= self.max_actions:
                return Result(
                    "limit",
                    "Stopped at the action limit. Give another command to continue.",
                    actions,
                )
            if decision.kind == "press" and decision.button not in available:
                return Result(
                    "unsupported", "That remote button is not available for this TV.", actions
                )
            if not execute:
                return Result(
                    "preview",
                    "Preview only: " + decision.message,
                    [decision.model_dump(exclude_none=True)],
                )
            self.progress(decision.message)
            # One call only. An exception or ambiguous delivery ends the entire task.
            receipt = (
                await self.bridge.press(decision.button)
                if decision.kind == "press"
                else await self.bridge.volume(decision.volume)
            )
            action = decision.model_dump(exclude_none=True)
            actions.append(action)
            if (
                receipt.get("ok") is not True
                or receipt.get("dry_run") is True
                or not receipt.get("results")
                or any(item.get("accepted") is not True for item in receipt["results"])
            ):
                return Result(
                    "uncertain",
                    "Command delivery is uncertain. Stopped; it will not be retried.",
                    actions,
                )
            history.append({"action": action, "delivery": "accepted, awaiting visual verification"})
            await asyncio.sleep(self.settle_seconds)
            if time.monotonic() - started >= self.max_seconds:
                return Result(
                    "limit",
                    "Command sent; the task deadline was reached before screen verification.",
                    actions,
                )
            self.progress("Checking the result on screen…")
            after = await self.bridge.capture()
            if screen.sha256 == after.sha256:
                return Result(
                    "unchanged",
                    "Command sent, but the screen did not change. Stopped without repeating it.",
                    actions,
                )
            screen = after
