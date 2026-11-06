import asyncio
import time

import pytest
from pydantic import ValidationError

from tv_voice.agent import TVTask
from tv_voice.bridge import Screen
from tv_voice.model import Decision


def screen(digest="a", age=0):
    return Screen(b"test", digest, time.monotonic() - age, 720, 540)


def press(button="right"):
    return Decision(
        kind="press", button=button, message="Moving right.", observation="Home is visible."
    )


def done():
    return Decision(kind="done", message="Netflix is open.", observation="Netflix home is visible.")


class FakeTV:
    def __init__(self, frames=None, receipt=None, connected=True):
        self.frames = list(frames or [screen(), screen("b")])
        self.receipt = (
            receipt if receipt is not None else {"ok": True, "results": [{"accepted": True}]}
        )
        self.connected = connected
        self.buttons = []
        self.events = []

    async def status(self):
        self.events.append("status")
        return {"connected": self.connected, "actions": ["right", "ok", "home"]}

    async def capture(self):
        self.events.append("capture")
        return self.frames.pop(0)

    async def press(self, action):
        self.events.append("press")
        self.buttons.append(action)
        return self.receipt

    async def volume(self, level):
        self.events.append("volume")
        self.buttons.append(level)
        return self.receipt


class ScriptedPlanner:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.seen = []

    async def decide(self, command, screenshot, history):
        self.seen.append((screenshot.sha256, list(history)))
        return self.decisions.pop(0)


def task(tv, planner, **kwargs):
    return TVTask(tv, planner, settle_seconds=0, progress=lambda _: None, **kwargs)


async def test_each_action_is_followed_by_a_new_observation():
    tv, planner = FakeTV(), ScriptedPlanner([press(), done()])
    result = await task(tv, planner).run("Open Netflix", execute=True)
    assert result.status == "done"
    assert tv.events == ["status", "capture", "press", "capture"]
    assert [entry[0] for entry in planner.seen] == ["a", "b"]
    assert len(planner.seen[1][1]) == 1


async def test_preview_never_sends_a_button():
    tv = FakeTV(connected=False)
    result = await task(tv, ScriptedPlanner([press()])).run("Open Netflix")
    assert result.status == "preview"
    assert not tv.buttons


async def test_unavailable_control_stops_before_capture_and_model():
    tv, planner = FakeTV(connected=False), ScriptedPlanner([])
    assert (await task(tv, planner).run("Open Netflix", execute=True)).status == "unavailable"
    assert tv.events == ["status"] and not planner.seen


@pytest.mark.parametrize(
    "receipt",
    [
        {"ok": False},
        {"ok": True},
        {"ok": True, "results": [{"accepted": False}]},
        {"ok": True, "dry_run": True, "results": [{"accepted": True}]},
    ],
)
async def test_uncertain_delivery_is_never_retried(receipt):
    tv = FakeTV(receipt=receipt)
    result = await task(tv, ScriptedPlanner([press(), press()])).run("Open Netflix", execute=True)
    assert result.status == "uncertain"
    assert tv.buttons == ["right"] and tv.events.count("capture") == 1


async def test_unchanged_frame_stops_without_repeating_action():
    tv = FakeTV(frames=[screen(), screen()])
    result = await task(tv, ScriptedPlanner([press(), press()])).run("Open Netflix", execute=True)
    assert result.status == "unchanged" and tv.buttons == ["right"]


async def test_stale_frame_never_reaches_model():
    tv, planner = FakeTV(frames=[screen(age=30)]), ScriptedPlanner([])
    assert (await task(tv, planner).run("Open Netflix", execute=True)).status == "stale"
    assert not planner.seen and not tv.buttons


async def test_frame_that_ages_during_reasoning_never_drives_action():
    class SlowPlanner:
        async def decide(self, *args):
            await asyncio.sleep(0.03)
            return press()

    tv = FakeTV()
    result = await task(tv, SlowPlanner(), max_screen_age=0.02).run("Open Netflix", execute=True)
    assert result.status == "stale" and not tv.buttons


async def test_action_budget_still_allows_verification():
    tv = FakeTV()
    result = await task(tv, ScriptedPlanner([press(), done()]), max_actions=1).run(
        "Open Netflix", execute=True
    )
    assert result.status == "done" and len(tv.buttons) == 1
    tv = FakeTV()
    result = await task(tv, ScriptedPlanner([press(), press()]), max_actions=1).run(
        "Open Netflix", execute=True
    )
    assert result.status == "limit" and len(tv.buttons) == 1


async def test_unknown_profile_button_is_not_sent():
    tv = FakeTV()
    result = await task(tv, ScriptedPlanner([press("left")])).run("Move left", execute=True)
    assert result.status == "unsupported" and not tv.buttons


async def test_capture_failure_after_delivery_does_not_replay():
    tv = FakeTV(frames=[screen()])
    with pytest.raises(IndexError):
        await task(tv, ScriptedPlanner([press(), press()])).run("Open Netflix", execute=True)
    assert tv.buttons == ["right"]


async def test_control_timeout_does_not_replay():
    class TimeoutTV(FakeTV):
        async def press(self, action):
            self.buttons.append(action)
            raise TimeoutError

    tv = TimeoutTV()
    with pytest.raises(asyncio.TimeoutError):
        await task(tv, ScriptedPlanner([press(), press()])).run("Open Netflix", execute=True)
    assert tv.buttons == ["right"]


async def test_cancel_during_delivery_stops_the_sequence():
    entered = asyncio.Event()

    class WaitingTV(FakeTV):
        async def press(self, action):
            self.buttons.append(action)
            entered.set()
            await asyncio.Event().wait()

    tv = WaitingTV()
    running = asyncio.create_task(
        task(tv, ScriptedPlanner([press(), press()])).run("Open Netflix", execute=True)
    )
    await entered.wait()
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert tv.buttons == ["right"]


async def test_volume_uses_validated_absolute_value():
    tv = FakeTV()
    plan = Decision(
        kind="volume", volume=15, message="Setting volume to 15.", observation="TV home."
    )
    await task(tv, ScriptedPlanner([plan, done()])).run("Volume 15", execute=True)
    assert tv.buttons == [15]


@pytest.mark.parametrize(
    "fields",
    [
        {"kind": "press", "button": "power_off"},
        {"kind": "press", "button": "right; rm -rf /"},
        {"kind": "press", "button": "right", "volume": 1},
        {"kind": "done", "button": "ok"},
        {"kind": "volume", "volume": True},
        {"kind": "volume", "volume": "10"},
        {"kind": "volume", "volume": 101},
        {"kind": "press", "button": "ok", "shell": "whoami"},
    ],
)
def test_action_schema_rejects_unbounded_or_ambiguous_actions(fields):
    with pytest.raises(ValidationError):
        Decision(**fields, message="Test", observation="Test")
