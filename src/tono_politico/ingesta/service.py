"""Componente 1: Ingesta — service OOP.

Encapsula config (data_dir, whisper_model, idioma) y orquesta
las funciones puras de playlist/audio/transcripcion.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..models import VideoTranscript
from .audio import descargar_audio, verificar_cache_videos
from .cache import ruta_audio, ruta_transcripcion
from .playlist import obtener_info_playlist
from .transcripcion import (
    cargar_transcripcion,
    guardar_transcripcion,
    transcribir,
    verificar_cache_transcripciones,
)

logger = logging.getLogger(__name__)


class IngestaService:
    """Service del Componente 1: de URL de playlist a transcripciones.

    Toda la configuración (data_dir, modelo Whisper, idioma) vive en self.
    Las funciones puras de los módulos auxiliares reciben base_dir=self.data_dir.

    Attributes:
        data_dir: Directorio raíz para cache de audios y transcripciones.
        whisper_model: Modelo de Whisper a usar (default: large-v3-turbo).
        idioma: Idioma para Whisper (default: es).
    """

    def __init__(
        self,
        data_dir: Path = Path("data"),
        whisper_model: str = "large-v3-turbo",
        idioma: str = "es",
    ) -> None:
        self.data_dir = data_dir
        self.whisper_model = whisper_model
        self.idioma = idioma

    def procesar(self, url_playlist: str) -> list[VideoTranscript]:
        """Procesa una playlist completa: descarga, transcribe y cachea.

        Flujo:
            1. Obtener metadata de la playlist.
            2. Verificar cache de transcripciones.
            3. Para los faltantes: descargar audios → transcribir → guardar.
            4. Cargar todas las transcripciones desde disco.
            5. Devolver en el orden original de la playlist.

        Args:
            url_playlist: URL completa de la playlist de YouTube.

        Returns:
            Lista de VideoTranscript en el orden original de la playlist.
        """
        # 1. Metadata
        info = obtener_info_playlist(url_playlist)

        if not info.videos:
            logger.info("Playlist vacía, no hay nada que procesar")
            return []

        # 2. Cache de transcripciones
        estado_t = verificar_cache_transcripciones(
            info.nombre, info.videos, self.data_dir
        )
        faltantes = estado_t["faltantes"]

        logger.info(
            f"Transcripciones: {len(estado_t['existentes'])} en cache, "
            f"{len(faltantes)} por procesar"
        )

        # 3. Procesar los faltantes
        if faltantes:
            self._procesar_faltantes(info.nombre, faltantes)

        # 4. Cargar todas las transcripciones desde disco
        resultados: list[VideoTranscript] = []
        for video in info.videos:
            ruta = ruta_transcripcion(info.nombre, video.id, self.data_dir)
            resultados.append(cargar_transcripcion(ruta))

        logger.info(f"Procesamiento completo: {len(resultados)} transcripciones")
        return resultados

    def _procesar_faltantes(
        self, nombre_playlist: str, faltantes: list
    ) -> None:
        """Descarga audios faltantes, transcribe y guarda en cache."""
        # Cache de audios
        estado_audios = verificar_cache_videos(
            nombre_playlist, faltantes, self.data_dir
        )
        audios_en_cache = estado_audios["existentes"]
        audios_faltantes = estado_audios["faltantes"]

        logger.info(
            f"Audios: {len(audios_en_cache)} en cache, "
            f"{len(audios_faltantes)} por descargar"
        )

        # Mapear rutas de audio (cache + recién descargados)
        rutas_audio: dict[str, Path] = {}
        for video in audios_en_cache:
            rutas_audio[video.id] = ruta_audio(
                nombre_playlist, video.id, self.data_dir
            )
        for video in audios_faltantes:
            rutas_audio[video.id] = descargar_audio(
                video, nombre_playlist, self.data_dir
            )

        # Transcribir y guardar
        for video in faltantes:
            segmentos = transcribir(
                rutas_audio[video.id],
                modelo=self.whisper_model,
                idioma=self.idioma,
            )
            transcript = VideoTranscript(
                video_id=video.id,
                url=video.url,
                titulo=video.titulo,
                fecha=video.fecha,
                raw_segments=segmentos,
            )
            guardar_transcripcion(
                transcript, nombre_playlist, self.data_dir
            )
