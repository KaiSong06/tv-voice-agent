"""Invoke the existing reviewed TV adapter; no generated shell commands."""

from __future__ import annotations

import asyncio
import hashlib
import json
import signal
import struct
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from .config import Config


@dataclass(frozen=True)
class Screen:
    png: bytes
    sha256: str
    captured_at: float
    width: int
    height: int


class TVBridge:
    def __init__(self, config: Config):
        self.config = config
        self.python = config.root / ".venv/bin/python"

    def prerequisites(self) -> dict[str, bool]:
        return {
            name: (self.config.root / path).exists()
            for name, path in {
                "tv_python": ".venv/bin/python",
                "capture_helper": "tools/native_screen.py",
                "capture_binary": ".runtime/native-capture/NativeCapture.dll",
                "sdb": ".runtime/tizen-sdb/data/tools/sdb",
                "sdk_transport": ".runtime/quiet-tizen-source/vendor/samsung_tv_root/sdb.py",
                "controller": "tvdiag/agent.py",
            }.items()
        }

    async def _run(self, args: list[str], timeout: float = 35) -> dict:
        process = await asyncio.create_subprocess_exec(
            str(self.python),
            *args,
            cwd=self.config.root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout)
        except BaseException:
            # Never replay a control after cancellation/timeout: it may have arrived.
            if process.returncode is None:
                # The screenshot helper has a finally block that removes TV staging files.
                # Let it run on interruption before falling back to a hard stop.
                process.send_signal(signal.SIGINT)
                try:
                    await asyncio.wait_for(process.communicate(), 20)
                except TimeoutError:
                    if process.returncode is None:
                        process.kill()
                    await process.communicate()
            raise
        try:
            value = json.loads(stdout)
        except (ValueError, UnicodeError):
            raise RuntimeError("TV adapter returned no JSON result; no automatic retry.") from None
        if not isinstance(value, dict):
            raise RuntimeError("TV adapter returned an invalid result.")  # noqa: TRY004 - remote protocol error
        if process.returncode and not (value.get("ok") is False or value.get("connected") is False):
            raise RuntimeError("TV adapter failed; see local TV diagnostics. No retry was sent.")
        return value

    async def status(self) -> dict:
        return await self._run(["-m", "tvdiag.agent", "status"])

    async def capture(self) -> Screen:
        # Keep screen pixels and the helper receipt out of the repository and delete on exit.
        with tempfile.TemporaryDirectory(prefix="tv-voice-") as folder:
            path = Path(folder) / "screen.png"
            started = time.monotonic()
            receipt = await self._run(
                [
                    str(self.config.root / "tools/native_screen.py"),
                    "--host",
                    self.config.host,
                    "--callback",
                    self.config.callback,
                    "--output",
                    str(path),
                ],
                timeout=60,
            )
            if receipt.get("success") is not True or receipt.get("staging_cleaned") is not True:
                raise RuntimeError("Screen capture or TV helper cleanup failed; stopped.")
            png = path.read_bytes()
            if not 24 <= len(png) <= 8 * 1024 * 1024 or png[:8] != b"\x89PNG\r\n\x1a\n":
                raise RuntimeError("Invalid TV screenshot.")
            width, height = struct.unpack(">II", png[16:24])
            digest = hashlib.sha256(png).hexdigest()
            metadata = receipt.get("capture", {})
            if digest != metadata.get("sha256") or (width, height) != (
                metadata.get("width"),
                metadata.get("height"),
            ):
                raise RuntimeError("Screenshot does not match its capture receipt.")
            # Conservative age includes capture/transfer time, not only receipt time.
            return Screen(png, digest, started, width, height)

    async def press(self, action: str) -> dict:
        return await self._run(["-m", "tvdiag.agent", "press", action])

    async def volume(self, level: int) -> dict:
        return await self._run(["-m", "tvdiag.agent", "volume", str(level)])
