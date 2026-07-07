"""Contratos de ejecución del pipeline completo.

Estos DTOs describen el estado observable de una corrida sin depender de logs:
resultado global, fases ejecutadas, videos omitidos/fallidos y artefactos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias

RunStatus: TypeAlias = Literal["ok", "partial", "failed"]
PhaseName: TypeAlias = Literal[
    "ingesta",
    "diarizacion",
    "segmentacion",
    "temas",
    "filtrado",
    "tono",
    "salida",
]


@dataclass(frozen=True)
class VideoRunStatus:
    """Estado de un video individual dentro de una corrida."""

    video_id: str
    titulo: str = ""
    descargado: bool = False
    transcrito: bool = False
    diarizado: bool = False
    segmentos_actor: int = 0
    omitido: bool = False
    error: str | None = None


@dataclass(frozen=True)
class PhaseRunStatus:
    """Estado de una fase del pipeline dentro de una corrida."""

    phase: PhaseName
    ok: bool
    elapsed_seconds: float = 0.0
    message: str = ""


@dataclass
class RunManifest:
    """Manifest observable de una ejecución del pipeline."""

    run_id: str
    playlist_url: str
    playlist_name: str
    status: RunStatus
    videos: list[VideoRunStatus] = field(default_factory=list)
    phases: list[PhaseRunStatus] = field(default_factory=list)
    artifacts_dir: Path | None = None
    cache_dir: Path | None = None


@dataclass
class RunResult:
    """Resultado de alto nivel devuelto por un runner/CLI."""

    manifest: RunManifest
    exit_code: int = 0
    informe_path: Path | None = None
