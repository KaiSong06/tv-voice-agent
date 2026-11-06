from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    root: Path
    host: str
    callback: str
    model: str = "gemini-3.5-flash-lite"
    whisper_model: str = "base.en"

    @classmethod
    def from_env(cls) -> Config:
        # Load only this application's local configuration; never search parent projects.
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
        root = os.environ.get("TV_DIAG_ROOT", "")
        if not root:
            raise ValueError("Set TV_DIAG_ROOT in tv-agent/.env to the existing TV project.")
        host, callback = os.getenv("TV_HOST", ""), os.getenv("TV_CALLBACK", "")
        for label, address in (("TV_HOST", host), ("TV_CALLBACK", callback)):
            try:
                ipaddress.IPv4Address(address)
            except ValueError:
                raise ValueError(f"Set {label} to its current IPv4 address.") from None
        return cls(
            Path(root).expanduser().resolve(),
            host,
            callback,
            os.getenv("TV_MODEL", "gemini-3.5-flash-lite"),
            os.getenv("TV_WHISPER_MODEL", "base.en"),
        )
