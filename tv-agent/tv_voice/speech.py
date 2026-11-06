# Adapted from pipecat-ai/pipecat-examples/local-input-select-stt/bot.py
# Copyright (c) 2024–2025, Daily. SPDX-License-Identifier: BSD-2-Clause
"""Explicit recording turns and Pipecat's local Whisper speech service."""

from __future__ import annotations

import asyncio
import select
import sys
import time
import wave
from pathlib import Path

SAMPLE_RATE = 16000
MAX_SECONDS = 20


def record(device: int | None = None) -> bytes:
    import sounddevice as sd

    chunks: list[bytes] = []
    errors = []
    size = 0
    limit = SAMPLE_RATE * 2 * MAX_SECONDS

    def callback(data, frames, timing, status):
        nonlocal size
        if status:
            errors.append(str(status))
        if size < limit:
            chunk = bytes(data)[: limit - size]
            chunks.append(chunk)
            size += len(chunk)

    print("Recording. Press Enter to send (20-second maximum).", flush=True)
    with sd.RawInputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="int16", device=device, callback=callback
    ):
        deadline = time.monotonic() + MAX_SECONDS
        while time.monotonic() < deadline:
            if select.select([sys.stdin], [], [], 0.1)[0]:
                if not sys.stdin.readline():
                    raise EOFError
                break
    if errors:
        raise RuntimeError("Microphone input overflow/interruption; recording discarded.")
    return b"".join(chunks)


def read_wav(path: Path) -> bytes:
    with wave.open(str(path), "rb") as source:
        if (source.getframerate(), source.getnchannels(), source.getsampwidth()) != (16000, 1, 2):
            raise ValueError("Use a mono, 16 kHz, 16-bit PCM WAV file.")
        if source.getnframes() > SAMPLE_RATE * MAX_SECONDS:
            raise ValueError("Audio is longer than the 20-second command limit.")
        return source.readframes(source.getnframes())


class LocalSpeech:
    def __init__(self, model: str):
        from pipecat.services.whisper.stt import WhisperSTTService
        from pipecat.transcriptions.language import Language

        self.stt = WhisperSTTService(
            device="cpu",
            compute_type="int8",
            settings=WhisperSTTService.Settings(
                model=model, language=Language.EN, no_speech_prob=0.3
            ),
        )

    async def transcribe(self, pcm: bytes) -> str:
        import numpy as np
        from pipecat.frames.frames import ErrorFrame, TranscriptionFrame

        if not pcm or len(pcm) > SAMPLE_RATE * 2 * MAX_SECONDS or len(pcm) % 2:
            raise ValueError("Invalid or empty audio command.")
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768
        if float(np.sqrt(np.mean(samples**2))) < 0.002:
            return ""
        # Same local STT service as upstream. Only explicit, bounded recordings enter it.
        texts = []
        async for frame in self.stt.run_stt(pcm):
            if isinstance(frame, ErrorFrame):
                raise RuntimeError("Local transcription failed; no command was sent.")  # noqa: TRY004 - STT error frame
            if isinstance(frame, TranscriptionFrame):
                texts.append(frame.text.strip())
        return " ".join(texts)


async def speak(text: str):
    # macOS system voice; text is an argv value, never a shell command.
    if sys.platform == "darwin":
        process = await asyncio.create_subprocess_exec("/usr/bin/say", "--", text)
        try:
            await asyncio.wait_for(process.wait(), 20)
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
