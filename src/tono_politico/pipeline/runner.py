"""Runner testeable del pipeline completo.

La clase de este módulo mueve la orquestación fuera del CLI para que las fases
puedan probarse con services inyectables y sin cargar modelos pesados.
"""

from __future__ import annotations

import logging
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tono_politico.models import PlaylistInfo, VideoTranscript
from tono_politico.temas.models import ResultadoTemas

from .models import PhaseName, PhaseRunStatus, RunManifest, RunResult, VideoRunStatus

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ServiceFactories:
    """Factories inyectables para construir los services del pipeline."""

    build_ingesta: Callable[[dict[str, Any]], Any]
    build_diarizacion: Callable[[dict[str, Any]], Any]
    build_segmentacion: Callable[[dict[str, Any]], Any]
    build_temas: Callable[[dict[str, Any]], Any]
    build_filtrado: Callable[[dict[str, Any], int], Any]
    build_tono: Callable[[dict[str, Any], str, str], Any]
    build_salida: Callable[[dict[str, Any], str | None], Any]
    get_playlist_info: Callable[[str], PlaylistInfo]


@dataclass
class Fase1Resultado:
    """Resultado interno de Fase 1 usado por discover/analyze."""

    resultado_temas: ResultadoTemas
    manifest: RunManifest


@dataclass
class PipelineRunner:
    """Orquesta el pipeline sin depender de argparse ni sys.exit."""

    cfg: dict[str, Any]
    factories: ServiceFactories
    keep_cache: bool = False
    last_resultado_temas: ResultadoTemas | None = field(default=None, init=False)

    def discover(self, playlist_url: str) -> RunResult:
        """Ejecuta Fase 1: Ingesta → Diarización → Segmentación → Temas."""
        fase_1 = self._ejecutar_fase_1(playlist_url)
        self._limpiar_cache(fase_1.manifest.playlist_name)
        return RunResult(manifest=fase_1.manifest, exit_code=0)

    def analyze(
        self,
        playlist_url: str,
        topico_id: int,
        tema: str,
        output_path: str | None,
    ) -> RunResult:
        """Ejecuta Fase 1 + Filtrado → Tono → Salida."""
        fase_1 = self._ejecutar_fase_1(playlist_url)
        manifest = fase_1.manifest

        svc_filtrado = self.factories.build_filtrado(self.cfg, topico_id)
        resultado_filtrado = self._run_phase(
            manifest,
            "filtrado",
            lambda: svc_filtrado.procesar(fase_1.resultado_temas),
        )

        if resultado_filtrado.total_segmentos_filtrados == 0:
            manifest.status = "failed"
            manifest.phases[-1] = PhaseRunStatus(
                phase="filtrado",
                ok=False,
                elapsed_seconds=manifest.phases[-1].elapsed_seconds,
                message=f"No hay segmentos para el tópico {topico_id}",
            )
            self._limpiar_cache(manifest.playlist_name)
            return RunResult(manifest=manifest, exit_code=1)

        actor = self.cfg.get("diarizacion", {}).get("actor_objetivo", "Lilly Téllez")
        svc_tono = self.factories.build_tono(self.cfg, actor, tema)
        resultado_tono = self._run_phase(
            manifest,
            "tono",
            lambda: svc_tono.procesar(resultado_filtrado),
        )

        svc_salida = self.factories.build_salida(self.cfg, output_path)
        self._run_phase(manifest, "salida", lambda: svc_salida.procesar(resultado_tono))

        self._limpiar_cache(manifest.playlist_name)
        informe_path = getattr(svc_salida, "output_path", None)
        return RunResult(
            manifest=manifest,
            exit_code=0,
            informe_path=Path(informe_path) if informe_path is not None else None,
        )

    def _ejecutar_fase_1(self, playlist_url: str) -> Fase1Resultado:
        info = self.factories.get_playlist_info(playlist_url)
        manifest = self._crear_manifest(playlist_url, info)

        svc_ingesta = self.factories.build_ingesta(self.cfg)
        transcripts = self._run_phase(
            manifest,
            "ingesta",
            lambda: svc_ingesta.procesar(playlist_url),
        )

        svc_diarizacion = self.factories.build_diarizacion(self.cfg)
        transcripts_actor = self._run_phase(
            manifest,
            "diarizacion",
            lambda: svc_diarizacion.procesar(transcripts, info.nombre),
        )

        manifest.videos = _video_statuses(transcripts, transcripts_actor)

        svc_segmentacion = self.factories.build_segmentacion(self.cfg)
        segmentos = self._run_phase(
            manifest,
            "segmentacion",
            lambda: svc_segmentacion.procesar(transcripts_actor),
        )

        svc_temas = self.factories.build_temas(self.cfg)
        resultado_temas = self._run_phase(
            manifest,
            "temas",
            lambda: svc_temas.procesar(segmentos),
        )
        self.last_resultado_temas = resultado_temas

        return Fase1Resultado(resultado_temas=resultado_temas, manifest=manifest)

    def _crear_manifest(self, playlist_url: str, info: PlaylistInfo) -> RunManifest:
        return RunManifest(
            run_id=_run_id(),
            playlist_url=playlist_url,
            playlist_name=info.nombre,
            status="ok",
            cache_dir=self._cache_dir(info.nombre),
        )

    def _run_phase(
        self,
        manifest: RunManifest,
        phase: PhaseName,
        fn: Callable[[], Any],
    ) -> Any:
        started = time.perf_counter()
        result = fn()
        manifest.phases.append(
            PhaseRunStatus(
                phase=phase,
                ok=True,
                elapsed_seconds=time.perf_counter() - started,
            )
        )
        return result

    def _cache_dir(self, playlist_name: str) -> Path:
        data_dir = Path(self.cfg.get("project", {}).get("data_dir", "data"))
        return data_dir / playlist_name

    def _limpiar_cache(self, playlist_name: str) -> None:
        playlist_dir = self._cache_dir(playlist_name)
        if not playlist_dir.exists():
            logger.debug("Cache no encontrado: %s", playlist_dir)
            return
        if self.keep_cache:
            logger.info("Cache conservado (--keep-cache): %s", playlist_dir)
            return
        logger.info("Limpiando cache runtime: %s", playlist_dir)
        shutil.rmtree(playlist_dir)


def _run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _video_statuses(
    transcripts: list[VideoTranscript],
    transcripts_actor: list[VideoTranscript],
) -> list[VideoRunStatus]:
    actor_by_id = {transcript.video_id: transcript for transcript in transcripts_actor}
    statuses: list[VideoRunStatus] = []
    for transcript in transcripts:
        actor_transcript = actor_by_id.get(transcript.video_id)
        segmentos_actor = len(actor_transcript.raw_segments) if actor_transcript else 0
        statuses.append(
            VideoRunStatus(
                video_id=transcript.video_id,
                titulo=transcript.titulo,
                descargado=True,
                transcrito=True,
                diarizado=actor_transcript is not None,
                segmentos_actor=segmentos_actor,
                omitido=segmentos_actor == 0,
            )
        )
    return statuses
