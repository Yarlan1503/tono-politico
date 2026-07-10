"""DTOs y protocolos del dominio de clips actor-only."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class ClipWindow:
    """Mapeo entre una ventana editada y el timeline original."""

    clip_start: float
    clip_end: float
    source_start: float
    source_end: float


@dataclass(frozen=True)
class ClipTranscriptSegment:
    """Segmento Whisper relativo al inicio del clip."""

    text: str
    t_start: float
    t_end: float


class ClipTranscriber(Protocol):
    def transcribir_clip(
        self,
        audio_path: Path,
        *,
        t_start: float,
        t_end: float,
        modelo: str,
        idioma: str,
    ) -> list[ClipTranscriptSegment]: ...
