import hashlib
import struct
from pathlib import Path

import pytest

from tv_voice.bridge import TVBridge
from tv_voice.cli import task_lock
from tv_voice.config import Config


def config(root):
    return Config(root, "192.0.2.10", "192.0.2.11")


async def test_capture_validates_receipt_and_removes_temporary_pixels(tmp_path):
    bridge = TVBridge(config(tmp_path))
    png = b"\x89PNG\r\n\x1a\n" + bytes(8) + struct.pack(">II", 720, 540)
    paths = []

    async def fake_run(args, timeout):
        path = Path(args[-1])
        paths.append(path)
        path.write_bytes(png)
        return {
            "success": True,
            "staging_cleaned": True,
            "capture": {"sha256": hashlib.sha256(png).hexdigest(), "width": 720, "height": 540},
        }

    bridge._run = fake_run
    screen = await bridge.capture()
    assert screen.png == png and not paths[0].exists()


async def test_bad_capture_digest_stops_and_discards_pixels(tmp_path):
    bridge = TVBridge(config(tmp_path))
    paths = []

    async def fake_run(args, timeout):
        path = Path(args[-1])
        paths.append(path)
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes(8) + struct.pack(">II", 1, 1))
        return {"success": True, "staging_cleaned": True, "capture": {"sha256": "wrong"}}

    bridge._run = fake_run
    with pytest.raises(RuntimeError, match="receipt"):
        await bridge.capture()
    assert not paths[0].exists()


def test_concurrent_voice_sessions_are_rejected(tmp_path):
    with task_lock(tmp_path), pytest.raises(RuntimeError, match="Another TV"), task_lock(tmp_path):
        pytest.fail("Second session acquired the lock")
