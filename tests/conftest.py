"""Fixtures compartidas entre todos los tests de ingesta."""

from __future__ import annotations

import json

import pytest

from tono_politico.models import (
    PlaylistInfo,
    SegmentoRaw,
    VideoInfo,
    VideoTranscript,
    WordTimestamp,
)


@pytest.fixture
def playlist_mock() -> PlaylistInfo:
    """Playlist simulada con 3 videos de prueba."""
    return PlaylistInfo(
        nombre="TestPlaylist",
        url="https://youtube.com/playlist?list=FAKE",
        videos=[
            VideoInfo(
                id="vid001",
                titulo="Video Uno",
                url="https://www.youtube.com/watch?v=vid001",
                duracion=120.0,
                fecha="20260101",
            ),
            VideoInfo(
                id="vid002",
                titulo="Video Dos",
                url="https://www.youtube.com/watch?v=vid002",
                duracion=300.0,
                fecha="20260201",
            ),
            VideoInfo(
                id="vid003",
                titulo="Video Tres",
                url="https://www.youtube.com/watch?v=vid003",
                duracion=60.0,
                fecha="20260301",
            ),
        ],
    )


@pytest.fixture
def playlist_vacia_mock() -> PlaylistInfo:
    """Playlist simulada sin videos."""
    return PlaylistInfo(
        nombre="PlaylistVacia",
        url="https://youtube.com/playlist?list=EMPTY",
        videos=[],
    )


@pytest.fixture
def yt_dlp_json_output() -> list[str]:
    """Simula la salida JSON línea-por-línea de yt-dlp --flat-playlist -j."""
    return [
        json.dumps(
            {
                "_type": "url",
                "ie_key": "Youtube",
                "id": "vid001",
                "title": "Video Uno",
                "url": "https://www.youtube.com/watch?v=vid001",
                "duration": 120.0,
                "upload_date": "20260101",
                "playlist": "TestPlaylist",
            }
        ),
        json.dumps(
            {
                "_type": "url",
                "ie_key": "Youtube",
                "id": "vid002",
                "title": "Video Dos",
                "url": "https://www.youtube.com/watch?v=vid002",
                "duration": 300.0,
                "upload_date": "20260201",
                "playlist": "TestPlaylist",
            }
        ),
    ]


@pytest.fixture
def transcript_mock() -> VideoTranscript:
    """VideoTranscript simulado con 2 segmentos y words."""
    return VideoTranscript(
        video_id="vid001",
        url="https://www.youtube.com/watch?v=vid001",
        titulo="Video Uno",
        fecha="20260101",
        raw_segments=[
            SegmentoRaw(
                texto="Hola mundo.",
                t_start=0.0,
                t_end=1.5,
                pausa_antes=0.0,
                words=[
                    WordTimestamp(word="Hola", start=0.0, end=0.6, probability=0.91),
                    WordTimestamp(word="mundo.", start=0.6, end=1.5, probability=0.88),
                ],
            ),
            SegmentoRaw(
                texto="Segundo segmento.",
                t_start=5.0,
                t_end=7.0,
                pausa_antes=3.5,
                words=[
                    WordTimestamp(word="Segundo", start=5.0, end=5.5, probability=None),
                ],
            ),
        ],
    )
