from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from .agent import TVTask
from .bridge import TVBridge
from .config import Config
from .model import GeminiPlanner


@contextmanager
def task_lock(root: Path):
    folder = root / ".runtime"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "voice-agent.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another TV voice session is running.") from None
        yield


async def run(args) -> int:
    if args.command == "devices":
        import sounddevice as sd

        print(sd.query_devices())
        return 0
    config = Config.from_env()
    bridge = TVBridge(config)
    if args.command == "doctor":
        report = {
            "model": config.model,
            "local_stt": config.whisper_model,
            "model_key_configured": bool(
                os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            ),
            "prerequisites": bridge.prerequisites(),
        }
        if all(report["prerequisites"].values()):
            state = await bridge.status()
            report["controls_connected"] = state.get("connected") is True
            if not report["controls_connected"]:
                report["control_help"] = (
                    "Run the existing TV project's python -m tvdiag.agent login."
                )
            if args.capture:
                with task_lock(config.root):
                    screen = await bridge.capture()
                report["capture"] = {
                    "verified": True,
                    "width": screen.width,
                    "height": screen.height,
                }
        print(json.dumps(report, indent=2))
        return (
            0
            if (
                all(report["prerequisites"].values())
                and report.get("controls_connected")
                and report["model_key_configured"]
            )
            else 1
        )
    if args.command == "transcribe":
        from .speech import LocalSpeech, read_wav

        print(await LocalSpeech(config.whisper_model).transcribe(read_wav(args.wav)))
        return 0
    planner = GeminiPlanner(config.model)
    try:
        with task_lock(config.root):
            task = TVTask(bridge, planner, progress=lambda message: print(message, file=sys.stderr))
            if args.command == "run":
                result = await task.run(args.text, execute=args.execute)
                print(json.dumps(asdict(result), indent=2))
                return 0 if result.status in {"done", "ask", "preview"} else 1
            from .speech import LocalSpeech, record, speak

            speech = LocalSpeech(config.whisper_model)
            print("TV voice ready. Audio stays on this Mac. Ctrl+C quits.")
            print(
                "Controls enabled." if args.execute else "Preview mode: no remote buttons are sent."
            )
            while True:
                # No microphone is open while idle, transcribing, planning, or speaking.
                input("Press Enter to start a voice command… ")
                pcm = record(args.device)
                try:
                    transcript = await speech.transcribe(pcm)
                finally:
                    del pcm
                if not transcript:
                    print("No speech detected.")
                    continue
                print("You:", transcript)
                if transcript.lower().strip(" .!?") in {"cancel", "quit", "exit", "never mind"}:
                    print("Cancelled.")
                    continue
                result = await task.run(transcript, execute=args.execute)
                print(result.message)
                if not args.quiet:
                    await speak(result.message)
    finally:
        await planner.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Local voice commands and visual TV control.")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser(
        "doctor", help="Check configuration, credentials, and TV control readiness"
    )
    doctor.add_argument(
        "--capture", action="store_true", help="Also take and discard a live screenshot"
    )
    sub.add_parser("devices", help="List microphone device indexes")
    transcribe = sub.add_parser("transcribe", help="Test local ASR only; never control the TV")
    transcribe.add_argument("wav", type=Path)
    text = sub.add_parser("run", help="Run a typed command with the same TV agent")
    text.add_argument("text")
    text.add_argument(
        "--execute", action="store_true", help="Send TV actions instead of previewing"
    )
    voice = sub.add_parser("voice", help="Press Enter to start/stop each recording")
    voice.add_argument(
        "--execute", action="store_true", help="Send TV actions instead of previewing"
    )
    voice.add_argument("--device", type=int, help="Microphone index from devices")
    voice.add_argument("--quiet", action="store_true", help="Disable spoken responses")
    try:
        return asyncio.run(run(parser.parse_args()))
    except (KeyboardInterrupt, EOFError):
        print("\nStopped. Any in-flight command will not be retried.", file=sys.stderr)
        return 130
    except Exception as error:  # noqa: BLE001 - sanitize errors at the CLI boundary
        # API exception bodies can contain request data; don't echo them to shared terminals.
        print(f"Stopped ({type(error).__name__}). No automatic retry was sent.", file=sys.stderr)
        if isinstance(error, (ValueError, FileNotFoundError)):
            print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
