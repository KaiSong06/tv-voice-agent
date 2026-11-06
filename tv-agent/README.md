# TV voice agent

A Samsung TV adaptation of [Pipecat Examples](https://github.com/pipecat-ai/pipecat-examples).
Press Enter to start recording, speak a command, then press Enter to send it. Local Whisper
transcribes the audio. Gemini receives the transcript and a fresh TV screenshot, chooses one
remote action, and checks another screenshot before continuing.

The initial interface is a macOS terminal. It also accepts typed commands for debugging.

## Setup

Requires Python 3.11–3.13, [uv](https://docs.astral.sh/uv/), and the separately provisioned
`tvdiag` project with working Samsung controls and native capture. This adapter does not install
TV firmware, enable Developer Mode, or redistribute the capture transport. It invokes the existing
`tools/native_screen.py` and `python -m tvdiag.agent` interfaces in that project's virtual environment.

```sh
cd tv-agent
uv sync --python 3.12
cp env.example .env
chmod 600 .env
```

Set `TV_DIAG_ROOT`, `TV_HOST`, and `TV_CALLBACK` in `.env` to the existing TV project and current
private-network addresses. `TV_CALLBACK` is the Mac address configured in the TV's Developer Mode.
Add a [Gemini API key](https://aistudio.google.com/apikey) to `GEMINI_API_KEY` locally.
Do not commit this file or paste the key into a chat.

```sh
uv run tv-voice doctor --capture
uv run tv-voice devices
uv run tv-voice run "Open Netflix"             # inspect and preview one step
uv run tv-voice run "Open Netflix" --execute   # bounded visual navigation
uv run tv-voice voice --execute               # Enter starts/stops recording; Ctrl+C quits
```

The first voice run downloads Whisper `base.en`; subsequent transcription runs locally.
Use `--device INDEX` to select a microphone or `--quiet` to disable spoken responses.
macOS may request microphone access for the terminal hosting the command. Recording ends after
20 seconds, and the microphone is closed while idle, transcribing, controlling the TV, or speaking.

If `doctor` reports controls disconnected, sign in through the existing TV project's
`python -m tvdiag.agent login`, then rerun `doctor`. A working screenshot alone does not establish
that remote buttons can be delivered.

## Defaults and boundaries

- Pipecat's `WhisperSTTService`, CPU int8 `base.en`, English. Override `TV_WHISPER_MODEL` as needed.
- `gemini-3.5-flash-lite`, structured JSON output, minimal thinking, 768 output-token limit.
  Override `TV_MODEL` with a compatible vision model that supports the same settings.
- Local macOS `say` for short replies. No paid speech service, Daily account, or LiveKit server.
- Fixed semantic remote buttons and volume 0–100. No model-authored code, coordinates, or shell.
- At most six actions per task and a two-minute task deadline. Each operation also has a timeout;
  a capture already in progress may finish cleanup after the task deadline.
- Screenshots older than 20 seconds cannot drive actions. A control failure or unchanged image
  stops the loop. Commands with uncertain delivery are never automatically replayed.
- Preview is the default. `--execute` enables controls for the command/session.
- Power controls, app installation, arbitrary text entry, and account changes are not implemented.
  Navigation and selection are allowed; the planner is instructed to stop at purchase, sign-in,
  deletion, and account/settings screens. This prompt constraint is not a formal screen classifier.
- Ctrl+C stops the session. Spoken cancellation works at command entry; voice interruption during
  execution is not implemented because the microphone is closed during actions.
- No cross-command conversational memory yet. State and previous actions are retained within a task.

Audio stays in process memory and is not saved by this adapter. Only the command transcript and TV
screenshots are sent to Gemini. Screenshots and native-capture receipts use a temporary directory
and are removed after reading. The existing `tvdiag` control service retains its normal diagnostic
receipts. Provider-side retention depends on the API account/tier.

Google's [published standard price](https://ai.google.dev/gemini-api/docs/pricing#gemini-3.5-flash-lite)
on 2026-11-06 was $0.30 per million input tokens and $2.50 per million output tokens, including
thinking. Actual task cost depends on screenshots, prompt length, and action count. No live
latency, cost, or navigation-quality benchmark has been completed yet.

## Validation

```sh
uv run pytest
uv run ruff check tv_voice tests
uv run tv-voice transcribe /path/to/mono-16khz-pcm16.wav  # ASR only; no TV action
```

Unit tests exercise new-frame verification, limits, stale frames, unavailable controls, uncertain
delivery, unchanged screens, invalid model actions, temporary-image cleanup, and session locking.
These tests do not establish real microphone accuracy or live model/TV navigation success.

Verified on macOS on 2026-11-06:

- 31 tests passed; Ruff and Git whitespace checks passed.
- The new bridge captured a 720×540 image from the real TV and verified its digest/receipt.
- Pipecat/Whisper `base.en` transcribed a generated “Open Netflix” WAV correctly; inference took
  about 0.4 seconds after model loading. This is a synthetic smoke test, not human-ASR validation.
- Core Audio enumerated the Mac microphone and iPhone microphone; neither was recorded by the test.
- `doctor` correctly reported the missing Gemini key and disconnected Samsung controls. Live API
  navigation and a human push-to-talk session remain unverified until those connections are ready.

## Upstream and license

Fork base: `pipecat-ai/pipecat-examples` at `7875d02bc0714d59528b0c8797664308af960a3a`.
The adaptation reuses Pipecat's local Whisper service and the explicit-turn approach from
`local-input-select-stt/` and `push-to-talk/`. Unlike the upstream web demo, it closes the microphone
between explicit recording turns and uses the existing TV adapter for actions and vision.
Upstream examples remain available unchanged. The BSD-2-Clause license and Daily attribution are
preserved in the repository's [LICENSE](../LICENSE).

Pipecat was selected because it provides local ASR and interchangeable model services without
requiring a hosted voice transport. LiveKit Agents is also Apache-2.0 but introduces a LiveKit
server/session layer. OpenInterpreter 01 is AGPL-3.0 and centers on general computer/code execution;
this task only needs a small TV action surface.
