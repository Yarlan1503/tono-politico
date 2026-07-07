"""Convenciones de rutas y directorios para cache local.

Un solo lugar que define dónde viven los datos de cada playlist:
    data/
    └── <playlist>/
        ├── videos-<playlist>/              # audios .wav
        └── transcripciones-<playlist>/     # transcripciones .json

Todas las funciones reciben base_dir opcional (default = DATA_DIR).
IngestaService pasa self.data_dir para controlar la ubicación del cache
sin mutar el global.
"""

from __future__ import annotations

from pathlib import Path

# Directorio raíz por defecto (backward compatible)
DATA_DIR = Path("data")


def ruta_dir_videos(nombre_playlist: str, base_dir: Path | None = None) -> Path:
    """Devuelve la ruta al directorio de audios de una playlist."""
    base = base_dir or DATA_DIR
    return base / nombre_playlist / f"videos-{nombre_playlist}"


def ruta_dir_transcripciones(nombre_playlist: str, base_dir: Path | None = None) -> Path:
    """Devuelve la ruta al directorio de transcripciones de una playlist."""
    base = base_dir or DATA_DIR
    return base / nombre_playlist / f"transcripciones-{nombre_playlist}"


def ruta_audio(nombre_playlist: str, video_id: str, base_dir: Path | None = None) -> Path:
    """Devuelve la ruta al archivo .wav de un video."""
    return ruta_dir_videos(nombre_playlist, base_dir) / f"{video_id}.wav"


def ruta_transcripcion(nombre_playlist: str, video_id: str, base_dir: Path | None = None) -> Path:
    """Devuelve la ruta al archivo .json de transcripción de un video."""
    return ruta_dir_transcripciones(nombre_playlist, base_dir) / f"{video_id}.json"
