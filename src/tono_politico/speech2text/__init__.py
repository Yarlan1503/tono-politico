"""Umbrella speech2text: audio → ActorTranscript turn-level (actor-only).

Subpaquetes:
    audio_fetcher/         — playlist + descarga .wav
    speaker_timestamps/    — pyannote + match del actor
    transcribe_speech/     — Whisper clips actor-only

Orquestador: SpeechToTextService.
"""

from __future__ import annotations

from .audio_fetcher import (
    AudioFetcherService,
    AudioVideo,
    DownloadResult,
    PlaylistInfo,
    VideoMeta,
)
from .models import (
    ActorTranscript,
    ActorTranscriptSegment,
    AsrMetadata,
    PerfilVozActor,
    SpeakerMatch,
    TurnoOrador,
)
from .service import SpeechToTextService
from .speaker_timestamps import SpeakerTimestampsService
from .transcribe_speech import TranscribeSpeechService

__all__ = [
    "SpeechToTextService",
    "AudioFetcherService",
    "SpeakerTimestampsService",
    "TranscribeSpeechService",
    "VideoMeta",
    "AudioVideo",
    "DownloadResult",
    "PlaylistInfo",
    "ActorTranscript",
    "ActorTranscriptSegment",
    "AsrMetadata",
    "PerfilVozActor",
    "SpeakerMatch",
    "TurnoOrador",
]
