"""Convenciones de rutas para cache de audios (.wav).

Un solo lugar que define dónde viven los audios de cada playlist::

    data/
    └── <playlist>/
        └── videos-<playlist>/              # audios .wav

No hay rutas de transcripción aquí (ASR vive en ``transcribe_speech``).
"""

from __future__ import annotations

from pathlib import Path

# Directorio raíz por defecto (backward compatible con ingesta)
DATA_DIR = Path("data")


def ruta_dir_videos(nombre_playlist: str, base_dir: Path | None = None) -> Path:
    """Devuelve la ruta al directorio de audios de una playlist."""
    base = base_dir or DATA_DIR
    return base / nombre_playlist / f"videos-{nombre_playlist}"


def ruta_audio(nombre_playlist: str, video_id: str, base_dir: Path | None = None) -> Path:
    """Devuelve la ruta al archivo .wav de un video."""
    return ruta_dir_videos(nombre_playlist, base_dir) / f"{video_id}.wav"
