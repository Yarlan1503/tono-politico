"""audio_fetcher: playlist + descarga de audio (sin Whisper).

API pública:
    AudioFetcherService — discover / fetch_one
    VideoMeta, AudioVideo, DownloadResult, PlaylistInfo, VideoInfo — DTOs
"""

from __future__ import annotations

from .models import AudioVideo, DownloadResult, PlaylistInfo, VideoInfo, VideoMeta
from .service import AudioFetcherService

__all__ = [
    "AudioFetcherService",
    "AudioVideo",
    "DownloadResult",
    "PlaylistInfo",
    "VideoInfo",
    "VideoMeta",
]
