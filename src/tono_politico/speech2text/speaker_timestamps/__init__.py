"""speaker_timestamps: quién habla cuándo + match del actor (sin ASR)."""

from __future__ import annotations

from .models import PerfilVozActor, SpeakerMatch, TurnoOrador
from .service import SpeakerTimestampsService

__all__ = [
    "SpeakerTimestampsService",
    "TurnoOrador",
    "PerfilVozActor",
    "SpeakerMatch",
]
