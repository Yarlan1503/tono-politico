"""audio_fetcher: playlist + descarga de audio (sin Whisper).

API pública:
    AudioFetcherService — discover / fetch_one
    VideoMeta, AudioVideo, DownloadResult — DTOs
"""

from __future__ import annotations

from .models import AudioVideo, DownloadResult, VideoMeta
from .service import AudioFetcherService

__all__ = [
    "AudioFetcherService",
    "AudioVideo",
    "DownloadResult",
    "VideoMeta",
]
