import wave
from pathlib import Path

import pytest

from tv_voice.speech import read_wav


def write_wave(path: Path, rate: int):
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        output.writeframes(bytes(3200))


def test_audio_format_is_checked_instead_of_silently_misinterpreted(tmp_path):
    path = tmp_path / "test.wav"
    write_wave(path, 48000)
    with pytest.raises(ValueError, match="16 kHz"):
        read_wav(path)
    write_wave(path, 16000)
    assert len(read_wav(path)) == 3200
