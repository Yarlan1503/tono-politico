"""DTOs del Componente 1: Ingesta."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DownloadResult:
    """Resultado estructurado de la descarga de un video.

    Attributes:
        video_id: ID del video de YouTube.
        path: Ruta al archivo de audio si la descarga fue exitosa, None si falló.
        ok: True si el archivo existe y es válido.
        error: Mensaje de error truncado si ok=False, None si ok=True.
    """

    video_id: str
    path: Path | None
    ok: bool
    error: str | None = None
